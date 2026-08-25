"""End-to-end scenario: incremental (streaming / SSE) response passthrough.

The proxy used to buffer a replica's ENTIRE response before returning it -- fatal for
Server-Sent Events, chunked token streams, progress feeds and long-poll endpoints (the
user would see nothing until the whole response finished). This proves the proxy now
relays bytes **as they are produced**: a real server that emits 6 SSE chunks 0.3s apart
is deployed across the mesh, and a client reading through the control-plane proxy sees
the chunks arrive spread out over time (not all at the end), with every chunk intact and
in order -- while ordinary (non-streaming) routes and failover keep working.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sim.harness import Harness  # noqa: E402
from meshpool.hosting import site as sitelib  # noqa: E402

TOKEN = "MESHPOOL-STREAM-5R8W"
N_CHUNKS = 6
GAP = 0.3   # seconds the server waits between flushing each chunk

# A real streaming backend: it flushes each SSE chunk then sleeps, so the bytes leave the
# server spread out over (N_CHUNKS-1)*GAP seconds. Content-Length is sent (total known) so
# we avoid hand-rolling chunked encoding, but the bytes still arrive incrementally.
SERVER_PY = '''\
import os
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PORT = int(os.environ.get("PORT", "8000"))
NODE = os.environ.get("MESHPOOL_NODE_NAME", "")
TOKEN = "MESHPOOL-STREAM-5R8W"
N_CHUNKS = 6
GAP = 0.3


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        path = self.path.split("?", 1)[0].rstrip("/")
        if path == "/stream":
            chunks = [("data: chunk-%d %s\\n\\n" % (i, TOKEN)).encode() for i in range(N_CHUNKS)]
            total = sum(len(c) for c in chunks)
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(total))
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            for i, c in enumerate(chunks):
                self.wfile.write(c)
                self.wfile.flush()
                if i < len(chunks) - 1:
                    time.sleep(GAP)
        elif path in ("", "/hello"):
            body = ("%s home on %s" % (TOKEN, NODE)).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            body = ("%s no route" % TOKEN).encode()
            self.send_response(404)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
'''


def build_app(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "server.py").write_text(SERVER_PY, encoding="utf-8")
    (root / "Procfile").write_text("web: python server.py\n", encoding="utf-8")


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


def main() -> int:  # noqa: C901 - a linear e2e script
    base = Path(__file__).resolve().parent / "_data_stream"
    h = Harness(n_agents=3, cp_port=8115, data_dir=str(base / "cluster"))
    ok = True
    try:
        print("== building a real STREAMING (SSE) server app ==")
        app_root = base / "myapp"
        build_app(app_root)
        bundle = sitelib.pack_dir(str(app_root), require_index=False)
        assert "server.py" in bundle, "server.py not packed"

        print("\n== starting cluster ==")
        h.start()
        cp = h.cp_url()

        print("\n== test 1: deploy the streaming app (proc, 2 replicas) ==")
        r = httpx.post(f"{cp}/api/v1/deployments",
                       json={"name": "streamer", "app": "proc", "replicas": 2,
                             "site": bundle, "start_cmd": "python server.py"}, timeout=40)
        r.raise_for_status()
        dep_id = r.json()["id"]
        reps, nodes = _wait_healthy(cp, dep_id, 2)
        assert len(nodes) >= 2, f"expected 2 healthy replicas on distinct nodes: {reps}"
        print(f"  PASS: {len(reps)} replicas healthy on {sorted(nodes)}")

        print("\n== test 2: a plain route still works through the proxy ==")
        hello = httpx.get(f"{cp}/app/{dep_id}/hello", timeout=10)
        assert hello.status_code == 200 and f"{TOKEN} home" in hello.text, \
            f"plain route broke: {hello.status_code} {hello.text[:80]}"
        print(f"  PASS: /hello served by {hello.headers.get('X-MeshPool-Served-By')}")

        print("\n== test 3: SSE chunks arrive INCREMENTALLY (not buffered to the end) ==")
        arrivals: list = []
        collected = bytearray()
        start = time.perf_counter()
        with httpx.stream("GET", f"{cp}/app/{dep_id}/stream", timeout=20) as resp:
            assert resp.status_code == 200, f"stream status {resp.status_code}"
            ctype = resp.headers.get("content-type", "")
            for chunk in resp.iter_bytes():
                if chunk:
                    arrivals.append(time.perf_counter() - start)
                    collected.extend(chunk)
        text = collected.decode("utf-8", "replace")
        # every chunk present and in order
        for i in range(N_CHUNKS):
            assert f"chunk-{i} {TOKEN}" in text, f"missing chunk-{i} in stream: {text!r}"
        assert text.index("chunk-0") < text.index(f"chunk-{N_CHUNKS - 1}"), "chunks out of order"
        assert "text/event-stream" in ctype, f"content-type not relayed: {ctype!r}"
        # the KEY proof: first bytes arrived well before the last (spread over time),
        # which is impossible if the proxy buffered the whole response first.
        span = arrivals[-1] - arrivals[0]
        expected = (N_CHUNKS - 1) * GAP
        print(f"  first byte @ {arrivals[0]:.2f}s, last @ {arrivals[-1]:.2f}s, "
              f"span={span:.2f}s (server spread ~{expected:.2f}s), reads={len(arrivals)}")
        assert span >= expected * 0.5, \
            f"response was buffered, not streamed: span={span:.2f}s < {expected * 0.5:.2f}s"
        assert arrivals[0] < expected * 0.5, \
            f"first byte too late ({arrivals[0]:.2f}s) -- looks buffered"
        print("  PASS: SSE relayed incrementally through the proxy")

        print("\n== test 4: kill a node -> streaming app still served (failover) ==")
        victim = [x for x in reps if x["healthy"]][0]["node_name"]
        victim_idx = int(victim.split("-")[-1]) - 1
        print(f"  killing {victim} ...")
        h.stop_agent(victim_idx)
        served = 0
        for _ in range(12):
            try:
                rr = httpx.get(f"{cp}/app/{dep_id}/hello", timeout=10)
                if rr.status_code == 200 and TOKEN in rr.text:
                    served += 1
            except Exception:  # noqa: BLE001
                pass
        assert served == 12, f"streaming app dropped requests during failover: {served}/12"
        print(f"  PASS: {served}/12 served after node kill (failover intact)")

        print("\nALL STREAMING / SSE PASSTHROUGH E2E CHECKS PASSED")
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
