"""End-to-end scenario: session affinity + real cookie passthrough for a stateful app.

``verify_procapp`` proves we can *run* an arbitrary backend across the mesh. This proves
the piece that makes a real multi-page app actually usable once it has >1 replica: a
logged-in user's server-side session must not bounce between workers on every click.

It deploys a genuinely **stateful** Python server -- it keeps an in-memory per-session
counter keyed by a cookie *it* sets -- across 2 nodes, then verifies through the
control-plane proxy that:
  * the app's own ``Set-Cookie`` is relayed back to the client, AND the client's
    ``Cookie`` is forwarded to the app (proven by the counter surviving across requests),
  * one client is **pinned to a single worker** (same live pid every request) so its
    session count climbs 1..N instead of resetting -- session affinity,
  * *different* clients still spread across both workers -- affinity doesn't defeat the
    load balancer,
  * killing the pinned worker **re-pins** the client to a survivor with no dropped
    request -- affinity degrades gracefully into failover.
This is the "any large multi-page application, reliably, for real users" path.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sim.harness import Harness  # noqa: E402
from meshpool.hosting import site as sitelib  # noqa: E402

TOKEN = "MESHPOOL-STICKY-9K4Z"          # marker proving OUR server process answered

# A genuinely stateful backend: it mints a session id in a cookie it sets, and keeps a
# per-session hit counter in this worker's memory. A monotonic counter across requests
# therefore PROVES both that the client's cookie reached the app and that the client
# kept landing on the SAME worker process (session affinity).
SERVER_PY = '''\
import json
import os
import uuid
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PORT = int(os.environ.get("PORT", "8000"))
NODE = os.environ.get("MESHPOOL_NODE_NAME", "")
REPLICA = os.environ.get("MESHPOOL_REPLICA_ID", "")
TOKEN = "MESHPOOL-STICKY-9K4Z"
SESSIONS = {}


class Handler(BaseHTTPRequestHandler):
    def _sid(self):
        raw = self.headers.get("Cookie", "")
        jar = SimpleCookie()
        if raw:
            jar.load(raw)
        if "app_sid" in jar:
            return jar["app_sid"].value, False
        return uuid.uuid4().hex, True

    def _send(self, code, body, ctype, set_sid=None):
        blob = body if isinstance(body, bytes) else body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(blob)))
        if set_sid is not None:
            self.send_header("Set-Cookie", "app_sid=%s; Path=/" % set_sid)
        self.end_headers()
        self.wfile.write(blob)

    def do_GET(self):
        path = self.path.split("?", 1)[0].rstrip("/")
        if path in ("", "/count"):
            sid, fresh = self._sid()
            SESSIONS[sid] = SESSIONS.get(sid, 0) + 1
            payload = json.dumps({"token": TOKEN, "node": NODE, "replica": REPLICA,
                                  "pid": os.getpid(), "sid": sid, "count": SESSIONS[sid]})
            self._send(200, payload, "application/json", set_sid=sid if fresh else None)
        elif path == "/whoami":
            payload = json.dumps({"token": TOKEN, "node": NODE, "replica": REPLICA,
                                  "pid": os.getpid()})
            self._send(200, payload, "application/json")
        else:
            self._send(404, "%s no route %s" % (TOKEN, path), "text/plain; charset=utf-8")

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
'''


def build_app(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "server.py").write_text(SERVER_PY, encoding="utf-8")
    (root / "Procfile").write_text("web: python server.py\n", encoding="utf-8")


def hit(client: httpx.Client, cp: str, dep_id: str, path: str) -> httpx.Response:
    return client.get(f"{cp}/app/{dep_id}/{path}")


def _wait_healthy(cp: str, dep_id: str, want: int, timeout: float = 35):
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


def main() -> int:  # noqa: C901 - a linear e2e script; readability beats decomposition here
    base = Path(__file__).resolve().parent / "_data_sticky"
    h = Harness(n_agents=3, cp_port=8114, data_dir=str(base / "cluster"))
    ok = True
    try:
        print("== building a real STATEFUL server app on disk ==")
        app_root = base / "myapp"
        build_app(app_root)
        bundle = sitelib.pack_dir(str(app_root), require_index=False)
        assert "server.py" in bundle, "server.py not packed"
        print(f"  packed {len(bundle)} files (stateful server.py + Procfile)")

        print("\n== starting cluster ==")
        h.start()
        cp = h.cp_url()

        print("\n== test 1: deploy the stateful app (proc, 2 replicas, distinct nodes) ==")
        r = httpx.post(f"{cp}/api/v1/deployments",
                       json={"name": "sticky", "app": "proc", "replicas": 2,
                             "site": bundle, "start_cmd": "python server.py"},
                       timeout=40)
        r.raise_for_status()
        dep_id = r.json()["id"]
        reps, nodes = _wait_healthy(cp, dep_id, 2)
        assert len(nodes) >= 2, f"expected 2 healthy replicas on distinct nodes: {reps}"
        print(f"  PASS: {len(reps)} process replicas, healthy on {sorted(nodes)}")

        print("\n== test 2: one client is PINNED to one worker; its session survives ==")
        client = httpx.Client(timeout=10)
        pids: set = set()
        served_by: set = set()
        counts: list = []
        last = {}
        for _ in range(10):
            resp = hit(client, cp, dep_id, "count")
            assert resp.status_code == 200 and TOKEN in resp.text, \
                f"stateful request failed: {resp.status_code} {resp.text[:80]}"
            last = resp.json()
            pids.add(last["pid"])
            counts.append(last["count"])
            served_by.add(resp.headers.get("X-MeshPool-Served-By", "?"))
        # the app's own Set-Cookie AND our affinity cookie both survived the proxy hop
        assert any(c.name == "app_sid" for c in client.cookies.jar), \
            "app's Set-Cookie was not relayed back to the client"
        assert any(c.name == "__mesh_aff" for c in client.cookies.jar), \
            "affinity cookie was not set by the proxy"
        assert len(pids) == 1, f"client bounced across worker processes (not sticky): {pids}"
        assert len(served_by) == 1, f"client bounced across replicas (not sticky): {served_by}"
        assert counts == list(range(1, 11)), \
            f"server-side session not preserved across requests: {counts}"
        pinned_pid = next(iter(pids))
        pinned_node = last["node"]
        print(f"  PASS: pinned to pid={pinned_pid} on {pinned_node}; "
              f"session count climbed {counts[0]}..{counts[-1]} (state preserved)")

        print("\n== test 3: DIFFERENT clients still spread across both workers ==")
        spread: set = set()
        for _ in range(6):
            fresh = httpx.Client(timeout=10)
            resp = hit(fresh, cp, dep_id, "whoami")
            assert resp.status_code == 200 and TOKEN in resp.text, "fresh client not served"
            spread.add(resp.json()["pid"])
            fresh.close()
        assert len(spread) >= 2, \
            f"affinity broke the load balancer -- new clients stuck on one worker: {spread}"
        print(f"  PASS: 6 new clients spread across {len(spread)} workers: {sorted(spread)}")

        print("\n== test 4: kill the pinned worker -> client re-pins to a survivor ==")
        victim_idx = int(pinned_node.split("-")[-1]) - 1
        print(f"  killing {pinned_node} (held pid={pinned_pid}) ...")
        h.stop_agent(victim_idx)
        post_pids: set = set()
        for _ in range(8):
            resp = hit(client, cp, dep_id, "count")
            assert resp.status_code == 200 and TOKEN in resp.text, \
                f"dropped request during failover: {resp.status_code}"
            post_pids.add(resp.json()["pid"])
        assert pinned_pid not in post_pids, \
            f"still routing the same client to the dead worker pid {pinned_pid}: {post_pids}"
        assert len(post_pids) == 1, \
            f"client not re-pinned to a single survivor after failover: {post_pids}"
        print(f"  PASS: re-pinned to survivor pid={next(iter(post_pids))}; "
              f"8/8 served, none hit the dead worker")

        print("\nALL STICKY-SESSION / COOKIE-PASSTHROUGH E2E CHECKS PASSED")
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
