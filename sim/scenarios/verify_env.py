"""End-to-end scenario: a hosted app is CONFIGURED with the deployer's env vars.

Real multi-page apps are configured the 12-factor way -- ``DATABASE_URL``, API keys,
feature flags -- through environment variables, not code. This proves MeshPool delivers
that config to every replica of a dynamic app and keeps it after failover:
  * the control plane REJECTS an invalid env var name at the API boundary (400),
  * a proc app deployed with ``env={...}`` actually sees those vars (read back through
    the load-balancing proxy from the app's own ``/env`` route),
  * the reserved platform vars win: a deployer who sets ``PORT`` cannot break the
    launcher -- the app still binds the injected port, never the user's value,
  * every worker process carries the same config (not just one replica),
  * killing a node keeps the app served WITH its config intact (failover + self-heal).
This is the "host any real app that needs configuration" path.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sim.harness import Harness  # noqa: E402
from meshpool.hosting import site as sitelib  # noqa: E402

TOKEN = "MESHPOOL-ENV-4T7K"

# The deployer's configuration. PORT is set deliberately WRONG (1): the platform must
# override it so the app still binds the injected port -- proving reserved vars win.
DEPLOY_ENV = {
    "APP_GREETING": "hello-from-the-mesh",
    "APP_SECRET_TOKEN": "s3cr3t-9QwErTy",
    "PORT": "1",
}

# A genuine backend that reports the config it was handed (and the port it truly bound),
# so a static file server could never fake it. Launched per replica on the injected $PORT.
SERVER_PY = '''\
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PORT = int(os.environ.get("PORT", "8000"))
NODE = os.environ.get("MESHPOOL_NODE_NAME", "")
TOKEN = "MESHPOOL-ENV-4T7K"


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype):
        blob = body if isinstance(body, bytes) else body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(blob)))
        self.end_headers()
        self.wfile.write(blob)

    def do_GET(self):
        path = self.path.split("?", 1)[0].rstrip("/")
        if path == "":
            self._send(200, f"<!doctype html><h1>{TOKEN} home</h1>", "text/html; charset=utf-8")
        elif path == "/env":
            self._send(200, json.dumps({
                "token": TOKEN,
                "greeting": os.environ.get("APP_GREETING", "<unset>"),
                "secret": os.environ.get("APP_SECRET_TOKEN", "<unset>"),
                "port": os.environ.get("PORT", "<unset>"),
                "node": NODE,
                "pid": os.getpid(),
            }), "application/json")
        else:
            self._send(404, f"{TOKEN} no route {path}", "text/plain; charset=utf-8")

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
'''


def build_app(root: Path) -> None:
    """Write a real config-driven backend (its own server + a Procfile) to disk."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "server.py").write_text(SERVER_PY, encoding="utf-8")
    (root / "Procfile").write_text("web: python server.py\n", encoding="utf-8")


def get(cp: str, dep_id: str, path: str):
    return httpx.get(f"{cp}/app/{dep_id}/{path}", timeout=10)


def _wait_healthy(cp: str, dep_id: str, want: int, timeout: float = 35):
    """Poll until ``want`` replicas are healthy on distinct nodes."""
    deadline = time.time() + timeout
    reps: list = []
    nodes: set = set()
    while time.time() < deadline:
        dep = httpx.get(f"{cp}/api/v1/deployments/{dep_id}", timeout=10).json()
        reps = dep["replicas"]
        healthy = [x for x in reps if x["healthy"]]
        nodes = {x["node_name"] for x in healthy}
        if len(healthy) >= want and len(nodes) >= want:
            return reps, nodes
        time.sleep(1)
    return reps, nodes


def main() -> int:
    base = Path(__file__).resolve().parent / "_data_env"
    h = Harness(n_agents=3, cp_port=8116, data_dir=str(base / "cluster"))
    ok = True
    try:
        print("== building a config-driven server app on disk ==")
        app_root = base / "myapp"
        build_app(app_root)
        bundle = sitelib.pack_dir(str(app_root), require_index=False)
        print(f"  packed {len(bundle)} files (server.py + Procfile)")

        print("\n== starting cluster ==")
        h.start()
        cp = h.cp_url()

        print("\n== test 1: the control plane REJECTS an invalid env var name (boundary) ==")
        bad = httpx.post(f"{cp}/api/v1/deployments",
                         json={"name": "bad", "app": "proc", "replicas": 1,
                               "site": bundle, "start_cmd": "python server.py",
                               "env": {"1BAD NAME": "x"}}, timeout=40)
        assert bad.status_code == 400, f"expected 400 for bad env, got {bad.status_code}"
        print(f"  PASS: bad env rejected with 400 ({bad.json().get('detail')})")

        print("\n== test 2: deploy WITH env config (proc, 2 replicas) ==")
        r = httpx.post(f"{cp}/api/v1/deployments",
                       json={"name": "myapp", "app": "proc", "replicas": 2,
                             "site": bundle, "start_cmd": "python server.py",
                             "env": DEPLOY_ENV}, timeout=40)
        r.raise_for_status()
        dep_id = r.json()["id"]
        reps, nodes = _wait_healthy(cp, dep_id, 2)
        assert len(nodes) >= 2, f"expected 2 healthy replicas on distinct nodes: {reps}"
        print(f"  PASS: {len(reps)} replicas healthy on {sorted(nodes)}")

        print("\n== test 3: the app RECEIVED the deployer's env through the proxy ==")
        env = get(cp, dep_id, "env")
        assert env.status_code == 200, f"/env not served: {env.status_code}"
        j = env.json()
        assert j["greeting"] == DEPLOY_ENV["APP_GREETING"], f"greeting not injected: {j}"
        assert j["secret"] == DEPLOY_ENV["APP_SECRET_TOKEN"], f"secret not injected: {j}"
        print(f"  PASS: app sees APP_GREETING={j['greeting']!r} APP_SECRET_TOKEN={j['secret']!r}")

        print("\n== test 4: reserved PORT beat the user's PORT=1; every worker got the config ==")
        seen_pids: set = set()
        seen_ports: set = set()
        for _ in range(20):
            jj = get(cp, dep_id, "env").json()
            assert jj["greeting"] == DEPLOY_ENV["APP_GREETING"], f"a worker missed the env: {jj}"
            seen_pids.add(jj["pid"])
            seen_ports.add(jj["port"])
        assert "1" not in seen_ports, f"user PORT=1 leaked through (reserved not enforced): {seen_ports}"
        assert len(seen_pids) >= 2, f"config not verified on >=2 workers: {seen_pids}"
        print(f"  PASS: {len(seen_pids)} workers all carry the config; "
              f"bound ports={sorted(seen_ports)} (never '1')")

        print("\n== test 5: kill a node -> config still applied on the survivor ==")
        victim = [x for x in reps if x["healthy"]][0]["node_name"]
        victim_idx = int(victim.split("-")[-1]) - 1
        print(f"  killing {victim} ...")
        h.stop_agent(victim_idx)
        good = 0
        for _ in range(12):
            try:
                resp = get(cp, dep_id, "env")
                if resp.status_code == 200 and resp.json().get("greeting") == DEPLOY_ENV["APP_GREETING"]:
                    good += 1
            except Exception:  # noqa: BLE001
                pass
        assert good == 12, f"env-configured app dropped requests / lost config on failover: {good}/12"
        print("  PASS: 12/12 served with config intact after node kill")

        print("\nALL HOSTED-APP ENV / CONFIG E2E CHECKS PASSED")
    except Exception as exc:  # noqa: BLE001
        ok = False
        print(f"\nFAILED: {type(exc).__name__}: {exc}")
        import traceback
        traceback.print_exc()
    finally:
        h.stop()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
