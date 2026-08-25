"""End-to-end scenario: live batch progress events over SSE."""
from __future__ import annotations

import json
import queue
import shutil
import sys
import threading
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sim.harness import Harness  # noqa: E402


def _submit(cp: str) -> str:
    payloads = [{"x": i} for i in range(12)]
    r = httpx.post(f"{cp}/api/v1/jobs", json={
        "name": "square-progress",
        "func": "square",
        "payloads": payloads,
        "cpu_pct": 15,
        "ram_mb": 128,
    }, timeout=10)
    r.raise_for_status()
    return r.json()["job_id"]


def _read_events(cp: str, out: "queue.Queue[dict]", ready: threading.Event,
                 stop: threading.Event) -> None:
    try:
        with httpx.stream("GET", f"{cp}/api/v1/events", timeout=None) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if stop.is_set():
                    break
                if not line.startswith("data: "):
                    continue
                evt = json.loads(line[6:])
                out.put(evt)
                if evt.get("kind") == "hello":
                    ready.set()
    except Exception as exc:  # noqa: BLE001
        out.put({"kind": "reader_error", "data": {"error": str(exc)}})
        ready.set()


def main() -> int:
    data_dir = Path(__file__).resolve().parent / f"_data_progress_{int(time.time() * 1000)}"
    h = Harness(n_agents=2, cp_port=8300, data_dir=str(data_dir), scale="home")
    events: "queue.Queue[dict]" = queue.Queue()
    ready = threading.Event()
    stop = threading.Event()
    ok = True
    try:
        print("== starting cluster (control plane + 2 agents on port 8300) ==")
        h.start()
        cp = h.cp_url()
        reader = threading.Thread(target=_read_events, args=(cp, events, ready, stop), daemon=True)
        reader.start()
        if not ready.wait(timeout=10):
            raise TimeoutError("SSE stream did not send hello")

        print("== submitting 12-unit square job and watching live progress ==")
        job_id = _submit(cp)
        print(f"  job {job_id} submitted")

        deadline = time.time() + 45
        progress_done: list[int] = []
        finished = None
        while time.time() < deadline and finished is None:
            try:
                evt = events.get(timeout=1)
            except queue.Empty:
                continue
            kind = evt.get("kind")
            data = evt.get("data") or {}
            if kind == "reader_error":
                raise RuntimeError(data.get("error"))
            if data.get("id") != job_id:
                continue
            if kind == "job_progress":
                done = int(data.get("done", 0))
                progress_done.append(done)
                print(f"  progress: done={done} running={data.get('running')} total={data.get('total')}")
            elif kind == "job_finished":
                finished = data
                print(f"  finished: {data}")

        if finished is None:
            raise TimeoutError(f"job_finished not observed; progress={progress_done}")
        if finished.get("status") != "completed":
            raise AssertionError(f"expected completed, got {finished}")
        if not progress_done:
            raise AssertionError("no job_progress events observed")
        increases = sum(1 for a, b in zip(progress_done, progress_done[1:]) if b > a)
        if max(progress_done) <= min(progress_done) or increases < 1:
            raise AssertionError(f"done count did not increase over progress stream: {progress_done}")

        print(f"PASS: observed {len(progress_done)} job_progress events; done counts {progress_done}")
    except Exception as exc:  # noqa: BLE001
        ok = False
        print(f"\nFAILED: {type(exc).__name__}: {exc}")
        import traceback
        traceback.print_exc()
    finally:
        stop.set()
        h.stop()
        shutil.rmtree(data_dir, ignore_errors=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
