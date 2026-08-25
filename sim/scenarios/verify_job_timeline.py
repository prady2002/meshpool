"""Acceptance gate for Phase 0.3 job timelines.

Runs a real control plane and two real agents, submits a traced batch job, then
asserts the timeline endpoint exposes ordered node-attributed attempt spans.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from meshpool.common.tracing import TraceContext  # noqa: E402
from sim.harness import Harness  # noqa: E402


def _wait_job(cp: str, job_id: str, timeout: float = 45.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        r = httpx.get(f"{cp}/api/v1/jobs/{job_id}", timeout=10)
        r.raise_for_status()
        job = r.json()
        if job["status"].startswith("completed"):
            return job
        time.sleep(0.25)
    raise TimeoutError(f"job {job_id} did not finish")


def main() -> int:
    data_dir = Path(__file__).resolve().parent / "_data_job_timeline"
    h = Harness(n_agents=2, cp_port=8300, data_dir=str(data_dir))
    ok = True
    try:
        print("== starting timeline cluster on reserved port 8300 ==")
        h.start()
        cp = h.cp_url()
        trace = TraceContext(trace_id="c" * 32, span_id="d" * 16)
        payloads = [{"x": i} for i in range(8)]
        submit = httpx.post(
            f"{cp}/api/v1/jobs",
            headers={"traceparent": trace.to_traceparent()},
            json={"name": "timeline-gate", "func": "square",
                  "payloads": payloads, "cpu_pct": 15, "ram_mb": 64},
            timeout=10,
        )
        submit.raise_for_status()
        job_id = submit.json()["job_id"]
        assert submit.json()["trace_id"] == trace.trace_id
        job = _wait_job(cp, job_id)
        assert job["status"].startswith("completed")

        timeline = httpx.get(f"{cp}/api/v1/jobs/{job_id}/timeline", timeout=10)
        timeline.raise_for_status()
        data = timeline.json()
        events = data["events"]
        attempt_spans = [e for e in events if e["name"] == "unit.attempt"]
        node_ids = {e["node_id"] for e in attempt_spans}
        print(f"  timeline events={len(events)} attempt_spans={len(attempt_spans)} "
              f"lanes={len(data['lanes'])} nodes={len(node_ids)} trace={data['trace_id']}")

        assert data["trace_id"] == trace.trace_id
        assert len(events) >= len(payloads) + 2, "timeline is missing job/event/attempt detail"
        assert len(attempt_spans) == len(payloads), "each unit attempt must be represented"
        assert all(e["duration_ms"] is not None for e in attempt_spans)
        assert all(e["node_id"] for e in attempt_spans), "attempts must be node-attributed"
        assert all(e["attempt"] >= 1 for e in attempt_spans)
        assert any(lane["node_id"] == "control-plane" for lane in data["lanes"])
        assert len([lane for lane in data["lanes"] if lane["node_id"] != "control-plane"]) >= 1
        starts = [e["start_ms"] for e in events]
        assert starts == sorted(starts), "timeline events must be chronologically ordered"
        print("PASS: job timeline is ordered, traced, and node-attributed")
    except Exception as exc:  # noqa: BLE001
        ok = False
        print(f"FAILED: {type(exc).__name__}: {exc}")
        import traceback
        traceback.print_exc()
    finally:
        h.stop()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
