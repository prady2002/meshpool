"""End-to-end scenario: deploy a REAL dynamic server app (its own process) across the mesh.

Where ``verify_realapp`` ships a static multi-page *folder*, this proves the "run any
server app" path: a genuine Python HTTP server -- its own routes, plus live JSON that
reports the worker's pid/node -- is shipped to 2 nodes, launched on each as a child
process listening on the injected ``$PORT``, and load-balanced through the control-plane
proxy with:
  * dynamic routes served (``/``, ``/about``, ``/api/info`` as live JSON),
  * requests genuinely spread across >= 2 worker *processes* on distinct nodes
    (proven by distinct live pids, not just distinct static replicas),
  * unknown routes answered by the app itself (its own 404, not the proxy's),
and it keeps serving after a hosting node is killed (failover + self-heal relaunches
the process on a survivor). This is the "host any large multi-page application that has
its own backend" path -- not just static files.
"""
from __future__ import annotations

import sys
import time
from collections import Counter
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sim.harness import Harness  # noqa: E402
from meshpool.hosting import site as sitelib  # noqa: E402

TOKEN = "MESHPOOL-PROCAPP-7X2Q"          # marker proving OUR server process answered

# A genuine, self-contained backend: reads its port from $PORT (as MeshPool injects it),
# computes routes at request time, and returns live JSON with this worker's pid -- so a
# static file server could never produce it. Written to disk and launched per replica.
SERVER_PY = '''\
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PORT = int(os.environ.get("PORT", "8000"))
NODE = os.environ.get("MESHPOOL_NODE_NAME", "")
REPLICA = os.environ.get("MESHPOOL_REPLICA_ID", "")
TOKEN = "MESHPOOL-PROCAPP-7X2Q"


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
            self._send(200, f"<!doctype html><h1>{TOKEN} home</h1><p>on {NODE}</p>",
                       "text/html; charset=utf-8")
        elif path == "/about":
            self._send(200, f"<!doctype html><h1>{TOKEN} about</h1>",
                       "text/html; charset=utf-8")
        elif path == "/api/info":
            self._send(200, json.dumps({"token": TOKEN, "node": NODE,
                                        "replica": REPLICA, "pid": os.getpid()}),
                       "application/json")
        else:
            self._send(404, f"{TOKEN} no route {path}", "text/plain; charset=utf-8")

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
'''


def build_app(root: Path) -> None:
    """Write a real dynamic backend (its own server + a Procfile) to disk."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "server.py").write_text(SERVER_PY, encoding="utf-8")
    (root / "Procfile").write_text("web: python server.py\n", encoding="utf-8")


def get(cp: str, dep_id: str, path: str):
    return httpx.get(f"{cp}/app/{dep_id}/{path}", timeout=10)


def served_all(cp: str, dep_id: str, n: int) -> Counter:
    """Hit all three routes ``n`` times; count how many were answered by OUR app."""
    c: Counter = Counter()
    routes = ["", "about", "api/info"]
    for i in range(n):
        try:
            r = get(cp, dep_id, routes[i % len(routes)])
            if r.status_code == 200 and TOKEN in r.text:
                c[r.headers.get("X-MeshPool-Served-By", "?")] += 1
            else:
                c[f"HTTP {r.status_code}"] += 1
        except Exception as exc:  # noqa: BLE001
            c[f"ERR {type(exc).__name__}"] += 1
    return c


def _wait_healthy(cp: str, dep_id: str, want: int, timeout: float = 35):
    """Poll until ``want`` replicas are healthy on distinct nodes (dynamic apps take a
    moment to bind their own port after launch)."""
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
    base = Path(__file__).resolve().parent / "_data_procapp"
    h = Harness(n_agents=3, cp_port=8113, data_dir=str(base / "cluster"))
    ok = True
    try:
        print("== building a real dynamic server app on disk ==")
        app_root = base / "myapp"
        build_app(app_root)
        bundle = sitelib.pack_dir(str(app_root), require_index=False)
        print(f"  packed {len(bundle)} files (server.py + Procfile), no index.html needed")
        assert "server.py" in bundle, "server.py not packed"

        print("\n== starting cluster ==")
        h.start()
        cp = h.cp_url()

        print("\n== test 1: deploy the server app (proc, 2 replicas, distinct nodes) ==")
        r = httpx.post(f"{cp}/api/v1/deployments",
                       json={"name": "myapp", "app": "proc", "replicas": 2,
                             "site": bundle, "start_cmd": "python server.py"},
                       timeout=40)
        r.raise_for_status()
        dep_id = r.json()["id"]
        reps, nodes = _wait_healthy(cp, dep_id, 2)
        assert len(nodes) >= 2, f"expected 2 healthy replicas on distinct nodes: {reps}"
        print(f"  PASS: {len(reps)} process replicas, healthy on {sorted(nodes)}")

        print("\n== test 2: dynamic routes served through the proxy ==")
        home = get(cp, dep_id, "")
        assert home.status_code == 200 and f"{TOKEN} home" in home.text, \
            f"home not served: {home.status_code} {home.text[:80]}"
        assert "text/html" in home.headers.get("content-type", ""), "home wrong content-type"
        about = get(cp, dep_id, "about")
        assert about.status_code == 200 and f"{TOKEN} about" in about.text, "about route not served"
        info = get(cp, dep_id, "api/info")
        assert info.status_code == 200 and "application/json" in info.headers.get("content-type", ""), \
            f"/api/info not JSON: {info.status_code} {info.headers.get('content-type')}"
        j = info.json()
        assert j["token"] == TOKEN and j.get("pid"), f"live JSON wrong: {j}"
        print(f"  ok  /  /about  /api/info (live pid={j['pid']}, node={j.get('node')})")
        missing = get(cp, dep_id, "nope")
        assert missing.status_code == 404 and TOKEN in missing.text, \
            f"app's own 404 not returned: {missing.status_code}"
        print("  PASS: dynamic routes + live JSON served; app's own 404 returned")

        print("\n== test 3: load-balanced across >= 2 worker processes (distinct pids) ==")
        workers: Counter = Counter()
        served = 0
        for _ in range(24):
            resp = get(cp, dep_id, "api/info")
            if resp.status_code == 200:
                served += 1
                jj = resp.json()
                workers[(jj.get("node"), jj.get("pid"))] += 1
        assert served == 24, f"dropped requests: {served}/24"
        assert len(workers) >= 2, f"not load-balanced across worker processes: {dict(workers)}"
        print(f"  PASS: 24/24 served across {len(workers)} live workers: {dict(workers)}")

        print("\n== test 4: kill a node -> app still served (failover + self-heal relaunch) ==")
        victim = [x for x in reps if x["healthy"]][0]["node_name"]
        victim_idx = int(victim.split("-")[-1]) - 1
        print(f"  killing {victim} ...")
        h.stop_agent(victim_idx)
        dist = served_all(cp, dep_id, 21)
        good = sum(v for k, v in dist.items() if k.startswith("sim-node"))
        print(f"  21 requests across 3 routes after kill: {dict(dist)}")
        assert good == 21, f"dynamic app dropped requests during failover: {dist}"

        deadline = time.time() + 40
        healed = False
        live: set = set()
        while time.time() < deadline:
            dep = httpx.get(f"{cp}/api/v1/deployments/{dep_id}", timeout=10).json()
            healthy = [x for x in dep["replicas"] if x["healthy"]]
            live = {x["node_name"] for x in healthy}
            if len(healthy) >= 2 and victim not in live:
                healed = True
                break
            time.sleep(1)
        assert healed, "manager did not self-heal (relaunch) the dynamic app"
        final = served_all(cp, dep_id, 9)
        assert sum(v for k, v in final.items() if k.startswith("sim-node")) == 9, \
            f"app not served after self-heal/relaunch: {final}"
        print(f"  PASS: self-healed onto {sorted(live)}; app process relaunched + served")

        print("\nALL DYNAMIC PROC-APP HOSTING E2E CHECKS PASSED")
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
