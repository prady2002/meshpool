"""Acceptance gate: Mesh Apps blue/green rollout keeps serving traffic.

The gate deploys a real process app, drives continuous HTTP traffic through the
control-plane proxy, rolls out a new version, and requires a non-trivial request
count with zero failures and a clean old->new response transition.

Non-vacuity is embedded: mutate the rollout path to stop old replicas before the
new version is healthy, confirm this gate fails with dropped requests, then
restore the source byte-identically and verify SHA-256.
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


def _app_bundle(version: str, startup_delay: float) -> dict[str, str]:
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
            dep_id: str | None = None) -> str:
    body = {
        "name": name,
        "app": "proc",
        "replicas": 1,
        "site": _app_bundle(version, startup_delay),
        "start_cmd": "python -u server.py",
    }
    if dep_id:
        r = httpx.put(f"{cp}/api/v1/deployments/{dep_id}", json=body, timeout=90)
    else:
        r = httpx.post(f"{cp}/api/v1/deployments", json=body, timeout=90)
    r.raise_for_status()
    got = r.json()
    if got.get("status") != "running" or got.get("replicas") != 1:
        _fail(f"deployment did not become running: {got}")
    return got["id"]


def verify_rollout(cp: str) -> None:
    print("\n== deploy v1 and roll out v2 under traffic ==")
    dep_id = _deploy(cp, "zero-downtime-app", "v1", startup_delay=0.0)
    url = f"{cp}/app/{dep_id}/"
    stop = threading.Event()
    seen: list[str] = []
    failures: list[str] = []

    def traffic() -> None:
        with httpx.Client(timeout=3.0) as client:
            while not stop.is_set():
                try:
                    r = client.get(url)
                    if r.status_code != 200:
                        failures.append(f"HTTP {r.status_code}: {r.text[:80]}")
                    else:
                        seen.append(r.text.strip())
                except Exception as exc:  # noqa: BLE001
                    failures.append(f"{type(exc).__name__}: {exc}")
                time.sleep(0.02)

    t = threading.Thread(target=traffic, daemon=True)
    t.start()
    deadline = time.time() + 15
    while len(seen) < 10 and time.time() < deadline:
        time.sleep(0.05)
    _deploy(cp, "zero-downtime-app", "v2", startup_delay=2.0, dep_id=dep_id)
    deadline = time.time() + 15
    while "VERSION=v2" not in seen and time.time() < deadline:
        time.sleep(0.05)
    time.sleep(0.5)
    stop.set()
    t.join(timeout=5)

    if len(seen) < 50:
        _fail(f"non-vacuity request floor not met: only {len(seen)} successful requests")
    if failures:
        _fail(f"traffic saw {len(failures)} dropped request(s), first={failures[0]}")
    if "VERSION=v1" not in seen or "VERSION=v2" not in seen:
        _fail(f"traffic must observe both versions; got {sorted(set(seen))}")
    first_new = seen.index("VERSION=v2")
    if "VERSION=v1" in seen[first_new:]:
        _fail("responses regressed from v2 back to v1 after the cutover")
    _ok(f"{len(seen)} requests flowed with zero failures and clean v1->v2 cutover")


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
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True, errors="replace",
            timeout=420,
            env=env,
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
