"""Acceptance gate: Mesh Apps blue/green rollout keeps serving traffic.

The gate deploys real process apps, drives continuous HTTP traffic through the
control-plane proxy during successful and failed rollouts, verifies unhealthy
replicas receive no traffic, and self-mutates the classic "kill old before new
healthy" bug to prove the zero-downtime assertion is non-vacuous.
"""
from __future__ import annotations

import base64
import hashlib
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import httpx

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from sim.harness import Harness
from sim import mutation_guard  # noqa: E402

CP_PORT = 8567
DATA_ROOT = Path(__file__).resolve().parent / "_data_app_zero_downtime"
_INNER_RUN = os.environ.get("MESHPOOL_APP_ROLLOUT_MUTATION") == "1"


def _fail(message: str) -> None:
    print(f"  FAIL: {message}")
    raise SystemExit(1)


def _ok(message: str) -> None:
    print(f"  PASS: {message}")


def _b64(text: str) -> str:
    return base64.b64encode(text.encode()).decode()


def _app_bundle(version: str, startup_delay: float, exit_immediately: bool = False) -> dict[str, str]:
    if exit_immediately:
        return {
            "server.py": _b64("import sys\nprint('boom', flush=True)\nsys.exit(42)\n"),
            "Procfile": _b64("web: python -u server.py\n"),
        }
    server = f"""
import os
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

time.sleep({startup_delay!r})
port = int(os.environ["PORT"])
version = {version!r}

class H(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.startswith("/healthz"):
            body = b"ok"
        else:
            body = ("VERSION=" + version).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
    def log_message(self, *args):
        pass

ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()
""".lstrip()
    return {
        "server.py": _b64(server),
        "Procfile": _b64("web: python -u server.py\n"),
    }


def _deploy(cp: str, name: str, version: str, startup_delay: float,
            dep_id: str | None = None, replicas: int = 1,
            exit_immediately: bool = False) -> str:
    body = {
        "name": name,
        "app": "proc",
        "replicas": replicas,
        "site": _app_bundle(version, startup_delay, exit_immediately=exit_immediately),
        "start_cmd": "python -u server.py",
    }
    if dep_id:
        r = httpx.put(f"{cp}/api/v1/deployments/{dep_id}", json=body, timeout=90)
    else:
        r = httpx.post(f"{cp}/api/v1/deployments", json=body, timeout=90)
    r.raise_for_status()
    got = r.json()
    if got.get("status") != "running" or got.get("replicas") != replicas:
        _fail(f"deployment did not become running: {got}")
    return got["id"]


def _drive_traffic(url: str, stop: threading.Event,
                   seen: list[tuple[float, str]], failures: list[str]) -> None:
    with httpx.Client(timeout=3.0) as client:
        while not stop.is_set():
            try:
                r = client.get(url)
                if r.status_code != 200:
                    failures.append(f"HTTP {r.status_code}: {r.text[:80]}")
                else:
                    seen.append((time.time(), r.text.strip()))
            except Exception as exc:  # noqa: BLE001
                failures.append(f"{type(exc).__name__}: {exc}")
            time.sleep(0.02)


def verify_rollout(cp: str) -> None:
    print("\n== deploy v1 and roll out v2 under traffic ==")
    dep_id = _deploy(cp, "zero-downtime-app", "v1", startup_delay=0.0, replicas=2)
    url = f"{cp}/app/{dep_id}/"
    stop = threading.Event()
    seen: list[tuple[float, str]] = []
    failures: list[str] = []

    t = threading.Thread(target=_drive_traffic, args=(url, stop, seen, failures), daemon=True)
    t.start()
    deadline = time.time() + 15
    while len(seen) < 10 and time.time() < deadline:
        time.sleep(0.05)
    switch_start = time.time()
    _deploy(cp, "zero-downtime-app", "v2", startup_delay=2.0, dep_id=dep_id, replicas=2)
    switch_end = time.time()
    deadline = time.time() + 15
    while "VERSION=v2" not in [body for _ts, body in seen] and time.time() < deadline:
        time.sleep(0.05)
    time.sleep(0.5)
    stop.set()
    t.join(timeout=5)

    bodies = [body for _ts, body in seen]
    during = [body for ts, body in seen if switch_start <= ts <= switch_end]
    if len(seen) < 50:
        _fail(f"non-vacuity request floor not met: only {len(seen)} successful requests")
    if len(during) < 25:
        _fail(f"switch-window request floor not met: only {len(during)} requests during rollout")
    if failures:
        _fail(f"traffic saw {len(failures)} dropped request(s), first={failures[0]}")
    if "VERSION=v1" not in bodies or "VERSION=v2" not in bodies:
        _fail(f"traffic must observe both versions; got {sorted(set(bodies))}")
    if "VERSION=v1" not in during or "VERSION=v2" not in during:
        _fail(f"traffic must hit both old and new during switch; got {sorted(set(during))}")
    first_new = bodies.index("VERSION=v2")
    if "VERSION=v1" in bodies[first_new:]:
        _fail("responses regressed from v2 back to v1 after the cutover")
    _ok(f"{len(seen)} requests flowed ({len(during)} during switch) with zero failures and clean v1->v2 cutover")


def verify_failed_rollout_rollback(cp: str) -> None:
    print("\n== failed rollout rolls back while old keeps serving ==")
    dep_id = _deploy(cp, "rollback-app", "v1", startup_delay=0.0, replicas=1)
    url = f"{cp}/app/{dep_id}/"
    stop = threading.Event()
    seen: list[tuple[float, str]] = []
    failures: list[str] = []
    t = threading.Thread(target=_drive_traffic, args=(url, stop, seen, failures), daemon=True)
    t.start()
    deadline = time.time() + 10
    while len(seen) < 10 and time.time() < deadline:
        time.sleep(0.05)
    start = time.time()
    _deploy(cp, "rollback-app", "v2", startup_delay=0.0, dep_id=dep_id,
            replicas=1, exit_immediately=True)
    end = time.time()
    time.sleep(0.5)
    stop.set()
    t.join(timeout=5)
    during = [body for ts, body in seen if start <= ts <= end]
    bodies = [body for _ts, body in seen]
    if failures:
        _fail(f"failed rollout dropped traffic, first={failures[0]}")
    if len(during) < 5:
        _fail(f"failed-rollout non-vacuity floor not met: only {len(during)} during failure")
    if set(bodies) != {"VERSION=v1"}:
        _fail(f"failed rollout served something other than old version: {sorted(set(bodies))}")
    dep = httpx.get(f"{cp}/api/v1/deployments/{dep_id}", timeout=10).json()
    if dep.get("status") != "running" or len(dep.get("replicas", [])) != 1:
        _fail(f"old deployment was not restored after failed rollout: {dep}")
    _ok(f"{len(seen)} old-version requests flowed ({len(during)} during failed rollout)")


def verify_unhealthy_replica_gets_no_traffic(cp: str) -> None:
    print("\n== unhealthy replica receives zero routed traffic ==")
    dep_id = _deploy(cp, "health-routing-app", "v1", startup_delay=0.0, replicas=2)
    dep = httpx.get(f"{cp}/api/v1/deployments/{dep_id}", timeout=10).json()
    victim = dep["replicas"][0]
    nodes = httpx.get(f"{cp}/api/v1/nodes", timeout=10).json()["nodes"]
    node_url = next(n["url"] for n in nodes if n["id"] == victim["node_id"])
    httpx.post(f"{node_url}/stop_replica", json={"replica_id": victim["id"]}, timeout=10).raise_for_status()
    deadline = time.time() + 12
    unhealthy_seen = False
    while time.time() < deadline:
        dep = httpx.get(f"{cp}/api/v1/deployments/{dep_id}", timeout=10).json()
        if any(r.get("id") == victim["id"] and not r.get("healthy", True) for r in dep.get("replicas", [])):
            unhealthy_seen = True
            break
        time.sleep(0.2)
    if not unhealthy_seen:
        _fail("non-vacuity failed: stopped replica was never observed unhealthy")
    failures: list[str] = []
    ok = 0
    with httpx.Client(timeout=3.0) as client:
        for _ in range(60):
            r = client.get(f"{cp}/app/{dep_id}/")
            if r.status_code != 200:
                failures.append(f"HTTP {r.status_code}: {r.text[:80]}")
            else:
                ok += 1
    if ok < 50:
        _fail(f"health-routing non-vacuity floor not met: only {ok} successes")
    if failures:
        _fail(f"unhealthy replica received traffic, first failure={failures[0]}")
    _ok(f"{ok} requests avoided observed-unhealthy replica {victim['id']}")


def run_nonvacuity_check() -> None:
    print("\n== non-vacuity: killing old before healthy must drop traffic ==")
    target = REPO_ROOT / "meshpool" / "hosting" / "engine.py"
    original = target.read_bytes()
    before = hashlib.sha256(original).hexdigest()
    nl = b"\r\n" if b"\r\n" in original else b"\n"
    anchor = nl.join((
        b'            old.status = "rolling_out"',
        b"            self.store.put_deployment(old)",
        b"            new_replicas = await self._start_replicas(candidate, desired, persist=False)",
    ))
    replacement = nl.join((
        b"            for r in old_replicas:",
        b"                await self._stop_replica(r)",
        b"            old.replicas = []",
        b'            old.status = "rolling_out"',
        b"            self.store.put_deployment(old)",
        b"            new_replicas = await self._start_replicas(candidate, desired, persist=False)",
    ))
    mutated = original.replace(anchor, replacement, 1)
    if mutated == original:
        _fail("mutation anchor not found in hosting engine")
    _mg = mutation_guard.journal(target, original)
    target.write_bytes(mutated)
    try:
        env = dict(os.environ)
        env["MESHPOOL_APP_ROLLOUT_MUTATION"] = "1"
        result = subprocess.run(
            [sys.executable, str(Path(__file__).resolve())],
            cwd=str(REPO_ROOT), capture_output=True,
            text=True, errors="replace", timeout=420, env=env,
        )
    finally:
        target.write_bytes(original)
        mutation_guard.clear(_mg)
    after = hashlib.sha256(target.read_bytes()).hexdigest()
    if after != before:
        _fail(f"hosting engine was not restored byte-identically: {before} != {after}")
    print(f"  restored engine.py sha256={before}")
    if result.returncode == 0:
        _fail("mutated immediate-kill rollout still passed; stdout:\n" + result.stdout[-800:])
    if "dropped request" not in result.stdout and "HTTP 502" not in result.stdout:
        _fail("mutated rollout failed, but not because traffic was dropped:\n"
              + result.stdout[-1000:] + result.stderr[-500:])
    _ok(f"mutated rollout failed as expected with exit {result.returncode}")


def main() -> int:
    shutil.rmtree(DATA_ROOT, ignore_errors=True)
    h = Harness(n_agents=2, cp_port=CP_PORT, data_dir=str(DATA_ROOT), scale="home")
    try:
        print("== starting MeshPool cluster ==")
        h.start()
        verify_rollout(h.cp_url())
        verify_failed_rollout_rollback(h.cp_url())
        verify_unhealthy_replica_gets_no_traffic(h.cp_url())
    except SystemExit as exc:
        return int(exc.code or 0)
    except Exception as exc:  # noqa: BLE001
        print(f"\nFAIL: unexpected {type(exc).__name__}: {exc}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        h.stop()
    if not _INNER_RUN:
        run_nonvacuity_check()
    print("\nPASS: Mesh Apps zero-downtime rollout verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
