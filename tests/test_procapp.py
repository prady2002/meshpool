"""Unit tests for the dynamic (process) hosting path.

These lock down the pieces that let MeshPool host an app *with its own backend* (not
just static files): launching the deployer's server on the injected ``$PORT``
(:func:`meshpool.agent.runtime.start_proc_app`), the relaxed "any response = up"
health check that supports servers without a ``/healthz`` route
(:meth:`HostingManager._healthy`), the ``index.html``-optional bundle validation, and
the CLI's Procfile/``meshpool.json`` start-command auto-detection.
"""
from __future__ import annotations

import asyncio
import base64
import time
import types

import httpx
import pytest
from fastapi.testclient import TestClient

from meshpool.agent import runtime
from meshpool.controlplane import app as cp_app
from meshpool.cli.main import _detect_start_cmd
from meshpool.common.models import Deployment, Replica
from meshpool.common.transport import HttpTransport
from meshpool.hosting.frameworks import infer_app, prepare_app
from meshpool.hosting import site
from meshpool.hosting.engine import HostingManager


def b64(s: str) -> str:
    return base64.b64encode(s.encode()).decode()


# a tiny, self-contained backend that honours $PORT and echoes the injected env,
# proving it is a real *process* (not static file serving)
_SERVER_PY = (
    "import os\n"
    "from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer\n"
    "PORT = int(os.environ['PORT'])\n"
    "class H(BaseHTTPRequestHandler):\n"
    "    def do_GET(self):\n"
    "        body = ('PORT=%s RID=%s' % (PORT, os.environ.get('MESHPOOL_REPLICA_ID',''))).encode()\n"
    "        self.send_response(200)\n"
    "        self.send_header('Content-Length', str(len(body)))\n"
    "        self.end_headers()\n"
    "        self.wfile.write(body)\n"
    "    def log_message(self, *a):\n"
    "        pass\n"
    "ThreadingHTTPServer(('127.0.0.1', PORT), H).serve_forever()\n"
)


# --------------------------- start_proc_app -------------------------------- #
def test_start_proc_app_launches_a_real_server_on_injected_port(tmp_path):
    (tmp_path / "server.py").write_text(_SERVER_PY, encoding="utf-8")
    port = runtime.free_port()
    try:
        proc = runtime.start_proc_app(str(tmp_path), "python server.py", port,
                                      replica_id="rtest", node_name="node-x")
    except OSError:
        pytest.skip("no 'python' on PATH for the spawned child")
    try:
        body = ""
        deadline = time.time() + 8
        while time.time() < deadline:
            if proc.poll() is not None:
                raise AssertionError("server process exited before binding")
            try:
                r = httpx.get(f"http://127.0.0.1:{port}/", timeout=0.5)
                if r.status_code == 200:
                    body = r.text
                    break
            except httpx.HTTPError:
                time.sleep(0.1)
        assert f"PORT={port}" in body, f"server did not bind injected $PORT: {body!r}"
        assert "RID=rtest" in body, f"replica id env not injected: {body!r}"
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except Exception:  # noqa: BLE001
            proc.kill()


def test_start_proc_app_substitutes_port_token_and_injects_env(tmp_path, monkeypatch):
    captured = {}

    def fake_popen(args, **kwargs):
        captured["args"] = args
        captured["env"] = kwargs.get("env", {})
        return types.SimpleNamespace(poll=lambda: None, terminate=lambda: None)

    monkeypatch.setattr(runtime.subprocess, "Popen", fake_popen)
    runtime.start_proc_app(str(tmp_path), "python server.py --port {port}", 54321,
                           replica_id="r7", node_name="node-9")
    assert "54321" in captured["args"], captured["args"]
    assert "{port}" not in " ".join(captured["args"])
    env = captured["env"]
    assert env.get("PORT") == "54321" and env.get("MESHPOOL_APP_PORT") == "54321"
    assert env.get("MESHPOOL_REPLICA_ID") == "r7" and env.get("MESHPOOL_NODE_NAME") == "node-9"


def test_start_proc_app_applies_user_env_but_reserved_vars_win(tmp_path, monkeypatch):
    captured = {}

    def fake_popen(args, **kwargs):
        captured["env"] = kwargs.get("env", {})
        return types.SimpleNamespace(poll=lambda: None, terminate=lambda: None)

    monkeypatch.setattr(runtime.subprocess, "Popen", fake_popen)
    runtime.start_proc_app(
        str(tmp_path), "python server.py", 45000, replica_id="r1", node_name="n1",
        env_extra={"DATABASE_URL": "postgres://db/app", "PORT": "1",
                   "MESHPOOL_REPLICA_ID": "hacked"})
    env = captured["env"]
    # the deployer's own config reaches the app ...
    assert env.get("DATABASE_URL") == "postgres://db/app"
    # ... but the reserved platform vars always win over user-supplied values
    assert env.get("PORT") == "45000"
    assert env.get("MESHPOOL_REPLICA_ID") == "r1"


def test_start_proc_app_rejects_empty_command(tmp_path):
    with pytest.raises(ValueError, match="empty start command"):
        runtime.start_proc_app(str(tmp_path), "   ", 8000)


# --------------------------- log capture ----------------------------------- #
def test_log_buffer_is_bounded_and_returns_ordered_tail():
    buf = runtime.LogBuffer(maxlen=500)
    for i in range(600):
        buf.append(f"line-{i}")
    all_lines = buf.tail(1000)
    assert len(all_lines) == 500                 # oldest 100 dropped by the ring buffer
    assert all_lines[0] == "line-100"            # order preserved, tail kept
    assert all_lines[-1] == "line-599"
    assert buf.tail(3) == ["line-597", "line-598", "line-599"]


def test_start_proc_app_captures_child_stdout_and_stderr(tmp_path):
    (tmp_path / "logger.py").write_text(
        "import sys\n"
        "print('hello-from-app-stdout', flush=True)\n"
        "print('warning-on-stderr', file=sys.stderr, flush=True)\n"
        "print('app-ready-marker', flush=True)\n",
        encoding="utf-8")
    buf = runtime.LogBuffer()
    try:
        proc = runtime.start_proc_app(str(tmp_path), "python -u logger.py", 8000,
                                      replica_id="rlog", log_buffer=buf)
    except OSError:
        pytest.skip("no 'python' on PATH for the spawned child")
    try:
        deadline = time.time() + 6
        while time.time() < deadline and len(buf.tail(50)) < 3:
            time.sleep(0.1)
        lines = buf.tail(50)
        assert "hello-from-app-stdout" in lines           # stdout captured
        assert "warning-on-stderr" in lines               # stderr merged in too
        assert "app-ready-marker" in lines
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except Exception:  # noqa: BLE001
            proc.kill()


# --------------------------- clean_env validation -------------------------- #
def test_clean_env_accepts_valid_and_coerces_values():
    out = site.clean_env({"DATABASE_URL": "x", "DEBUG": 1, "EMPTY": None})
    assert out == {"DATABASE_URL": "x", "DEBUG": "1", "EMPTY": ""}


def test_clean_env_none_or_empty_is_empty_dict():
    assert site.clean_env(None) == {}
    assert site.clean_env({}) == {}


@pytest.mark.parametrize("bad", ["1BAD", "has space", "DASH-KEY", "", "a.b"])
def test_clean_env_rejects_bad_key(bad):
    with pytest.raises(ValueError, match="invalid env var name"):
        site.clean_env({bad: "x"})


def test_clean_env_enforces_count_cap(monkeypatch):
    monkeypatch.setattr(site, "MAX_ENV_VARS", 3)
    with pytest.raises(ValueError, match="too many env vars"):
        site.clean_env({f"K{i}": "v" for i in range(4)})


def test_clean_env_enforces_byte_cap(monkeypatch):
    monkeypatch.setattr(site, "MAX_ENV_BYTES", 10)
    with pytest.raises(ValueError, match="exceeds"):
        site.clean_env({"KEY": "a value that is definitely longer than ten bytes"})


# --------------------------- _healthy relaxation --------------------------- #
def _manager_with_transport(behavior) -> HostingManager:
    class FakeTransport:
        async def proxy(self, node_id, method, url, **kwargs):
            return behavior(url)

    mgr = object.__new__(HostingManager)   # bypass __init__: _healthy only needs transport
    mgr.transport = FakeTransport()
    return mgr


def _replica() -> Replica:
    return Replica(id="r1", node_id="n1", node_name="node-1", url="http://10.0.0.1:9000")


def test_healthy_true_when_app_answers_even_without_healthz():
    # a process app with no /healthz returns 404 there -> still "up" (any response)
    mgr = _manager_with_transport(lambda url: types.SimpleNamespace(status_code=404))
    assert asyncio.run(mgr._healthy(_replica())) is True


def test_healthy_true_on_200():
    mgr = _manager_with_transport(lambda url: types.SimpleNamespace(status_code=200))
    assert asyncio.run(mgr._healthy(_replica())) is True


def test_healthy_false_when_connection_refused():
    def boom(url):
        raise httpx.ConnectError("connection refused")

    mgr = _manager_with_transport(boom)
    assert asyncio.run(mgr._healthy(_replica())) is False


def test_healthy_false_when_server_errors_on_every_path():
    mgr = _manager_with_transport(lambda url: types.SimpleNamespace(status_code=502))
    assert asyncio.run(mgr._healthy(_replica())) is False


# --------------------------- index-optional bundles ------------------------ #
def test_normalize_site_allows_no_index_for_process_apps():
    bundle = {"server.py": b64("print('hi')"), "requirements.txt": b64("flask\n")}
    out = site.normalize_site(bundle, require_index=False)
    assert set(out) == {"server.py", "requirements.txt"}


def test_normalize_site_still_requires_index_by_default():
    with pytest.raises(ValueError, match="index.html"):
        site.normalize_site({"server.py": b64("print('hi')")})


def test_pack_dir_no_index_when_not_required(tmp_path):
    (tmp_path / "server.py").write_text("print('hi')", encoding="utf-8")
    (tmp_path / "Procfile").write_text("web: python server.py\n", encoding="utf-8")
    bundle = site.pack_dir(str(tmp_path), require_index=False)
    assert "server.py" in bundle and "Procfile" in bundle


# --------------------------- CLI start-command detection ------------------- #
def test_detect_start_cmd_from_procfile(tmp_path):
    (tmp_path / "Procfile").write_text("web: gunicorn app:app --port {port}\n", encoding="utf-8")
    assert _detect_start_cmd(tmp_path) == "gunicorn app:app --port {port}"


def test_detect_start_cmd_from_manifest(tmp_path):
    (tmp_path / "meshpool.json").write_text('{"start": "node server.js"}', encoding="utf-8")
    assert _detect_start_cmd(tmp_path) == "node server.js"


def test_detect_start_cmd_procfile_wins_over_manifest(tmp_path):
    (tmp_path / "Procfile").write_text("web: python server.py\n", encoding="utf-8")
    (tmp_path / "meshpool.json").write_text('{"start": "node server.js"}', encoding="utf-8")
    assert _detect_start_cmd(tmp_path) == "python server.py"


def test_detect_start_cmd_static_site_returns_empty(tmp_path):
    (tmp_path / "index.html").write_text("<h1>hi</h1>", encoding="utf-8")
    assert _detect_start_cmd(tmp_path) == ""


def test_framework_detection_writes_vite_static_config(tmp_path, monkeypatch):
    (tmp_path / "package.json").write_text(
        '{"scripts":{"build":"vite build"},"devDependencies":{"vite":"latest"}}',
        encoding="utf-8")
    (tmp_path / "dist").mkdir()
    (tmp_path / "dist" / "index.html").write_text("<h1>built</h1>", encoding="utf-8")
    calls = []

    def fake_run(*args, **kwargs):
        calls.append((args, kwargs))
        return types.SimpleNamespace(returncode=0)

    monkeypatch.setattr("meshpool.hosting.frameworks.subprocess.run", fake_run)
    plan = prepare_app(tmp_path)
    assert plan.kind == "site"
    assert plan.framework == "vite"
    assert plan.deploy_dir == tmp_path / "dist"
    assert calls and calls[0][0][0] == "npm run build"
    manifest = (tmp_path / "meshpool.json").read_text(encoding="utf-8")
    assert '"framework": "vite"' in manifest
    assert '"output": "dist"' in manifest


def test_framework_detection_fastapi_start_cmd(tmp_path):
    (tmp_path / "requirements.txt").write_text("fastapi\nuvicorn\n", encoding="utf-8")
    (tmp_path / "app.py").write_text("from fastapi import FastAPI\napp = FastAPI()\n",
                                     encoding="utf-8")
    plan = infer_app(tmp_path)
    assert plan.kind == "proc"
    assert plan.framework == "fastapi"
    assert "uvicorn app:app" in plan.start_cmd


@pytest.mark.parametrize(
    ("files", "expected"),
    [
        (
            {
                "package.json": (
                    '{"scripts":{"build":"vite build"},"devDependencies":{"vite":"latest"}}'
                ),
                "dist/index.html": "<h1>vite</h1>",
            },
            ("site", "vite", "dist", ""),
        ),
        (
            {
                "package.json": (
                    '{"scripts":{"build":"react-scripts build"},'
                    '"dependencies":{"react-scripts":"latest"}}'
                ),
                "build/index.html": "<h1>cra</h1>",
            },
            ("site", "create-react-app", "build", ""),
        ),
        (
            {
                "package.json": (
                    '{"scripts":{"build":"next build","start":"next start"},'
                    '"dependencies":{"next":"latest"}}'
                ),
            },
            ("proc", "next", ".", "npm run start"),
        ),
        (
            {"package.json": '{"scripts":{"start":"node server.js"}}', "server.js": ""},
            ("proc", "node", ".", "npm run start"),
        ),
        (
            {"index.html": "<h1>static</h1>"},
            ("site", "static", ".", ""),
        ),
        (
            {"package.json": '{"scripts":{"build":"vite build"}}'},
            ("site", "node-static", "dist", ""),
        ),
    ],
)
def test_framework_auto_detection_matrix_is_discriminating(tmp_path, files, expected):
    for rel, text in files.items():
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    plan = infer_app(tmp_path)
    kind, framework, deploy_rel, start_cmd = expected
    assert plan.kind == kind
    assert plan.framework == framework
    assert plan.deploy_dir == (tmp_path if deploy_rel == "." else tmp_path / deploy_rel)
    assert plan.start_cmd == start_cmd


# --------------------------- session affinity ------------------------------ #
def _manager_with_deployment(dep) -> HostingManager:
    class FakeStore:
        def get_deployment(self, dep_id):
            return dep if (dep is not None and dep.id == dep_id) else None

    mgr = object.__new__(HostingManager)   # bypass __init__: sticky only needs the store
    mgr.store = FakeStore()
    mgr._rr = {}                           # round-robin cursor, used by pick_replica()
    return mgr


def _dep_with(replicas) -> Deployment:
    return Deployment(id="dep-1", name="app", app="proc", site=None,
                      start_cmd="python server.py", replicas_desired=2,
                      replicas=replicas, owner="o", status="running")


def test_sticky_replica_returns_pinned_replica_when_healthy():
    r = Replica(id="rA", node_id="n1", node_name="node-1", url="http://x", healthy=True)
    mgr = _manager_with_deployment(_dep_with([r]))
    assert mgr.sticky_replica("dep-1", "rA") is r


def test_sticky_replica_none_when_pinned_replica_unhealthy():
    r = Replica(id="rA", node_id="n1", url="http://x", healthy=False)
    mgr = _manager_with_deployment(_dep_with([r]))
    assert mgr.sticky_replica("dep-1", "rA") is None   # -> caller round-robins instead


def test_sticky_replica_none_when_pinned_replica_already_tried():
    r = Replica(id="rA", node_id="n1", url="http://x", healthy=True)
    mgr = _manager_with_deployment(_dep_with([r]))
    assert mgr.sticky_replica("dep-1", "rA", exclude={"rA"}) is None   # failover skips it


def test_sticky_replica_none_when_id_unknown_or_blank():
    r = Replica(id="rA", node_id="n1", url="http://x", healthy=True)
    mgr = _manager_with_deployment(_dep_with([r]))
    assert mgr.sticky_replica("dep-1", "rZ") is None
    assert mgr.sticky_replica("dep-1", "") is None


def test_sticky_replica_none_when_deployment_missing():
    mgr = _manager_with_deployment(None)
    assert mgr.sticky_replica("dep-1", "rA") is None


def test_pick_replica_never_delivers_to_unhealthy_replica():
    healthy = Replica(id="healthy", node_id="n1", node_name="node-1",
                      url="http://healthy", healthy=True)
    unhealthy = Replica(id="unhealthy", node_id="n2", node_name="node-2",
                        url="http://unhealthy", healthy=False)
    mgr = _manager_with_deployment(_dep_with([healthy, unhealthy]))
    delivered = {"healthy": 0, "unhealthy": 0}
    for _ in range(25):
        picked = mgr.pick_replica("dep-1")
        assert picked is not None
        delivered[picked.id] += 1
    assert delivered["healthy"] == 25
    assert delivered["unhealthy"] == 0


# --------------------------- deployment removal ---------------------------- #
def _manager_for_remove(dep):
    """Wire a HostingManager with just enough fakes to exercise remove()."""
    stopped: list[str] = []
    deleted: list[str] = []

    class FakeStore:
        def get_deployment(self, dep_id):
            return dep if (dep is not None and dep.id == dep_id) else None

        def delete_deployment(self, dep_id):
            deleted.append(dep_id)

    class FakeRegistry:
        def get(self, node_id):
            return types.SimpleNamespace(id=node_id)

    class FakeTransport:
        async def call(self, node_id, method, payload, timeout=None):
            if method == "stop_replica":
                stopped.append(payload["replica_id"])
            return {}

    class FakeEvents:
        def publish(self, *a, **k):
            pass

    mgr = object.__new__(HostingManager)   # bypass __init__: remove only needs these attrs
    mgr.store = FakeStore()
    mgr.registry = FakeRegistry()
    mgr.transport = FakeTransport()
    mgr.events = FakeEvents()
    mgr._rr = {"dep-1": 3}
    mgr._lock = asyncio.Lock()
    return mgr, stopped, deleted


def test_remove_stops_every_replica_then_deletes_deployment():
    r1 = Replica(id="rA", node_id="n1", node_name="node-1", url="http://x", healthy=True)
    r2 = Replica(id="rB", node_id="n2", node_name="node-2", url="http://y", healthy=True)
    mgr, stopped, deleted = _manager_for_remove(_dep_with([r1, r2]))

    assert asyncio.run(mgr.remove("dep-1")) is True
    assert sorted(stopped) == ["rA", "rB"]     # both replicas told to stop across the pool
    assert deleted == ["dep-1"]                 # store row removed
    assert "dep-1" not in mgr._rr               # round-robin cursor cleaned up


def test_remove_unknown_deployment_returns_false_and_touches_nothing():
    mgr, stopped, deleted = _manager_for_remove(None)
    assert asyncio.run(mgr.remove("dep-1")) is False
    assert stopped == [] and deleted == []


# --------------------------- blue/green + health --------------------------- #
def _manager_for_rollout(old_dep):
    records: dict[str, Deployment] = {old_dep.id: old_dep}
    stopped: list[str] = []
    started: list[str] = []

    class FakeStore:
        def get_deployment(self, dep_id):
            return records.get(dep_id)

        def put_deployment(self, dep):
            records[dep.id] = dep.model_copy(deep=True)

        def list_deployments(self):
            return list(records.values())

    class FakeRegistry:
        def schedulable_nodes(self):
            return [
                types.SimpleNamespace(id="n1", name="node-1",
                                      provides=lambda svc: svc == "hosting",
                                      donatable_cpu_slots=lambda: 4),
                types.SimpleNamespace(id="n2", name="node-2",
                                      provides=lambda svc: svc == "hosting",
                                      donatable_cpu_slots=lambda: 4),
            ]

        def get(self, node_id):
            return types.SimpleNamespace(id=node_id)

        def online(self):
            return self.schedulable_nodes()

    class FakeTransport:
        async def call(self, node_id, method, payload, timeout=None):
            if method == "start_replica":
                started.append(payload["replica_id"])
                return {"replica_id": payload["replica_id"], "url": f"http://{node_id}"}
            if method == "stop_replica":
                stopped.append(payload["replica_id"])
            return {}

    class FakeEvents:
        def publish(self, *a, **k):
            pass

    mgr = object.__new__(HostingManager)
    mgr.store = FakeStore()
    mgr.registry = FakeRegistry()
    mgr.transport = FakeTransport()
    mgr.events = FakeEvents()
    mgr.storage = None
    mgr._rr = {}
    mgr._lock = asyncio.Lock()
    mgr._traffic = {}
    mgr._autoscale = {}
    return mgr, records, started, stopped


def test_rollout_keeps_old_replicas_until_new_version_is_healthy(monkeypatch):
    old_replica = Replica(id="old", node_id="n1", node_name="node-1",
                          url="http://old", healthy=True)
    old_dep = _dep_with([old_replica])
    old_dep.replicas_desired = 1
    mgr, records, started, stopped = _manager_for_rollout(old_dep)
    seen_during_health = {}

    async def healthy(replica):
        seen_during_health["replicas"] = [r.id for r in records["dep-1"].replicas]
        return True

    monkeypatch.setattr(mgr, "_healthy", healthy)
    monkeypatch.setenv("MESHPOOL_ROLLOUT_DRAIN_S", "0")
    dep = asyncio.run(mgr.rollout("dep-1", "app", "site", 1, site={"index.html": b64("v2")}))
    assert dep is not None
    assert seen_during_health["replicas"] == ["old"]
    assert records["dep-1"].replicas[0].id in started
    assert stopped == ["old"]


def test_rollout_failure_restores_old_replicas(monkeypatch):
    old_replica = Replica(id="old", node_id="n1", node_name="node-1",
                          url="http://old", healthy=True)
    old_dep = _dep_with([old_replica])
    old_dep.replicas_desired = 1
    mgr, records, _started, stopped = _manager_for_rollout(old_dep)

    async def unhealthy(_replica):
        return False

    monkeypatch.setattr(mgr, "_healthy", unhealthy)
    dep = asyncio.run(mgr.rollout("dep-1", "app", "site", 1, site={"index.html": b64("v2")}))
    assert dep is not None
    assert records["dep-1"].replicas == [old_replica]
    assert records["dep-1"].status == "running"
    assert stopped and stopped != ["old"]


def test_autoscale_raises_desired_replicas_on_slow_real_requests():
    old_dep = _dep_with([Replica(id="r1", node_id="n1", node_name="node-1",
                                 url="http://one", healthy=True)])
    old_dep.replicas_desired = 1
    mgr, records, _started, _stopped = _manager_for_rollout(old_dep)
    mgr._set_autoscale("dep-1", replicas=1, min_replicas=1, max_replicas=3)
    for _ in range(12):
        mgr.record_request("dep-1", 200, 400.0)
    changed = asyncio.run(mgr._autoscale_deployment(records["dep-1"]))
    assert changed is True
    assert records["dep-1"].replicas_desired == 2


def test_rollout_requires_all_desired_new_replicas_before_cutover(monkeypatch):
    old_replicas = [
        Replica(id="old1", node_id="n1", node_name="node-1", url="http://old1", healthy=True),
        Replica(id="old2", node_id="n2", node_name="node-2", url="http://old2", healthy=True),
    ]
    old_dep = _dep_with(old_replicas)
    old_dep.replicas_desired = 2
    mgr, records, _started, stopped = _manager_for_rollout(old_dep)

    async def healthy(_replica):
        return True

    monkeypatch.setattr(mgr, "_healthy", healthy)
    monkeypatch.setenv("MESHPOOL_ROLLOUT_DRAIN_S", "0")
    dep = asyncio.run(mgr.rollout("dep-1", "app", "site", 2, site={"index.html": b64("v2")}))
    assert dep is not None
    assert len(records["dep-1"].replicas) == 2
    assert all(r.id not in {"old1", "old2"} for r in records["dep-1"].replicas)
    assert sorted(stopped) == ["old1", "old2"]


def test_proxy_retries_5xx_replica_before_serving_response(monkeypatch):
    bad = Replica(id="bad", node_id="n1", node_name="bad-node", url="http://bad", healthy=True)
    good = Replica(id="good", node_id="n2", node_name="good-node", url="http://good", healthy=True)
    picks = [bad, good]
    recorded: list[int] = []

    class FakeHosting:
        def sticky_replica(self, dep_id, replica_id, exclude=None):
            return None

        def pick_replica(self, dep_id, exclude=None):
            while picks:
                candidate = picks.pop(0)
                if exclude and candidate.id in exclude:
                    continue
                return candidate
            return None

        def record_request(self, dep_id, status_code, duration_ms):
            recorded.append(status_code)

    class FakeTransport:
        async def proxy_open(self, node_id, method, url, **kwargs):
            if node_id == "n1":
                return httpx.Response(503, content=b"bad")
            return httpx.Response(200, content=b"good")

    monkeypatch.setattr(cp_app, "CP", types.SimpleNamespace(
        hosting=FakeHosting(), transport=FakeTransport(),
        settings=types.SimpleNamespace(auth_mode="open")))
    client = TestClient(cp_app.app)
    response = client.get("/app/dep-1/")
    assert response.status_code == 200
    assert response.text == "good"
    assert response.headers["x-meshpool-served-by"] == "good-node"
    assert recorded == [200]


# --------------------------- header passthrough ---------------------------- #
def test_http_transport_proxy_forwards_request_and_response_headers():
    captured = {}

    class FakeClient:
        async def request(self, method, url, **kwargs):
            captured["method"] = method
            captured["url"] = url
            captured["headers"] = kwargs.get("headers")
            captured["content"] = kwargs.get("content")
            return httpx.Response(200, headers={"set-cookie": "sid=abc; Path=/"},
                                  content=b"ok")

    t = HttpTransport(registry=None, client=FakeClient())
    res = asyncio.run(t.proxy("n1", "POST", "http://10.0.0.1:9000/login",
                              headers={"cookie": "sid=abc", "content-type": "application/json"},
                              content=b"{}"))
    # the client's real headers reach the app (so logins / sessions work)
    assert captured["headers"] == {"cookie": "sid=abc", "content-type": "application/json"}
    assert captured["content"] == b"{}"
    # and the app's own response headers (Set-Cookie!) come back for the browser
    assert res.status_code == 200
    assert res.headers.get("set-cookie") == "sid=abc; Path=/"
