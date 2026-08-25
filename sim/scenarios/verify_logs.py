"""End-to-end scenario: a hosted app's LOGS are captured and retrievable.

When you host a real backend, it *will* eventually misbehave -- and a silent process
(stdout/stderr thrown away) is impossible to debug. MeshPool captures each replica's
recent stdout/stderr on its node and lets the deployer read it back from the control
plane. This proves the whole path end to end:
  * a deployed proc app's startup output is captured on every replica,
  * live request activity shows up in the logs (proving it is real, streaming capture,
    not a one-shot snapshot),
  * logs are gathered from ALL replicas across DIFFERENT nodes through the control plane,
  * an unknown deployment id returns 404,
  * after a node is killed, logs are still retrievable for the survivor.
This is the "why is my hosted app unhealthy?" debugging path.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sim.harness import Harness  # noqa: E402
from meshpool.hosting import site as sitelib  # noqa: E402

TOKEN = "MESHPOOL-LOGS-7Q2Z"

# A genuine backend that LOGS: a startup banner (before it serves) plus one line per
# request. A static file server could never produce these, so seeing them proves real
# stdout/stderr capture from the deployer's own process.
SERVER_PY = '''\
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PORT = int(os.environ.get("PORT", "8000"))
NODE = os.environ.get("MESHPOOL_NODE_NAME", "")
TOKEN = "MESHPOOL-LOGS-7Q2Z"
print(f"{TOKEN} BOOT pid={os.getpid()} port={PORT} node={NODE}", flush=True)


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        path = self.path.split("?", 1)[0].rstrip("/") or "/"
        print(f"{TOKEN} REQ path={path} pid={os.getpid()}", flush=True)
        body = f"{TOKEN} ok {path} pid={os.getpid()}".encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
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
    (root / "Procfile").write_text("web: python -u server.py\n", encoding="utf-8")


def get(cp: str, dep_id: str, path: str):
    return httpx.get(f"{cp}/app/{dep_id}/{path}", timeout=10)


def fetch_logs(cp: str, dep_id: str, tail: int = 200):
    return httpx.get(f"{cp}/api/v1/deployments/{dep_id}/logs?tail={tail}", timeout=10)


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


def main() -> int:
    base = Path(__file__).resolve().parent / "_data_logs"
    h = Harness(n_agents=3, cp_port=8121, data_dir=str(base / "cluster"))
    ok = True
    try:
        print("== building a logging server app on disk ==")
        app_root = base / "myapp"
        build_app(app_root)
        bundle = sitelib.pack_dir(str(app_root), require_index=False)
        print(f"  packed {len(bundle)} files (server.py + Procfile)")

        print("\n== starting cluster ==")
        h.start()
        cp = h.cp_url()

        print("\n== test 1: deploy a proc app (2 replicas on distinct nodes) ==")
        r = httpx.post(f"{cp}/api/v1/deployments",
                       json={"name": "myapp", "app": "proc", "replicas": 2,
                             "site": bundle, "start_cmd": "python -u server.py"},
                       timeout=40)
        r.raise_for_status()
        dep_id = r.json()["id"]
        reps, nodes = _wait_healthy(cp, dep_id, 2)
        assert len(nodes) >= 2, f"expected 2 healthy replicas on distinct nodes: {reps}"
        print(f"  PASS: {len(reps)} replicas healthy on {sorted(nodes)}")

        print("\n== test 2: every replica's STARTUP output was captured ==")
        data = fetch_logs(cp, dep_id).json()
        log_reps = data["replicas"]
        assert len(log_reps) == 2, f"expected logs for 2 replicas: {len(log_reps)}"
        for rep in log_reps:
            joined = "\n".join(rep["lines"])
            assert TOKEN in joined and "BOOT" in joined, \
                f"no startup log for {rep['replica_id']} on {rep['node_name']}: {rep['lines'][-5:]}"
        print(f"  PASS: boot banner captured on both replicas "
              f"({sum(len(x['lines']) for x in log_reps)} lines total)")

        print("\n== test 3: LIVE request activity shows up in the logs ==")
        marker = "sentinel-42"
        for _ in range(6):
            assert get(cp, dep_id, marker).status_code == 200
        time.sleep(0.6)  # let the reader threads drain the pipes
        data = fetch_logs(cp, dep_id).json()
        all_lines = "\n".join(ln for rep in data["replicas"] for ln in rep["lines"])
        assert f"path=/{marker}" in all_lines, \
            f"our request was not logged (capture is not live): ...{all_lines[-400:]}"
        print(f"  PASS: request to /{marker} appears in the captured logs")

        print("\n== test 4: logs for an unknown deployment -> 404 ==")
        missing = httpx.get(f"{cp}/api/v1/deployments/nope-xyz/logs", timeout=10)
        assert missing.status_code == 404, f"expected 404, got {missing.status_code}"
        print("  PASS: unknown deployment rejected with 404")

        print("\n== test 5: kill a node -> logs still retrievable for the survivor ==")
        victim = [x for x in reps if x["healthy"]][0]["node_name"]
        victim_idx = int(victim.split("-")[-1]) - 1
        print(f"  killing {victim} ...")
        h.stop_agent(victim_idx)
        # keep serving so the survivor logs more, then confirm logs still come back
        for _ in range(6):
            get(cp, dep_id, "after-kill")
        got_boot = False
        for _ in range(15):
            resp = fetch_logs(cp, dep_id)
            if resp.status_code == 200:
                lines = "\n".join(ln for rep in resp.json()["replicas"] for ln in rep["lines"])
                if TOKEN in lines and "BOOT" in lines:
                    got_boot = True
                    break
            time.sleep(1)
        assert got_boot, "logs not retrievable after node kill"
        print("  PASS: logs still retrievable after node kill")

        print("\nALL HOSTED-APP LOG-CAPTURE E2E CHECKS PASSED")
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
