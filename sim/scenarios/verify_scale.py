"""Concurrency/scale acceptance gate for many nodes and concurrent batch jobs.

Boots a five-node LAN simulation, submits multiple batch jobs at once, and proves
that every unit completes exactly once with no failures while work fans out across
the whole fleet.
"""
from __future__ import annotations

import os
import shutil
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sim.harness import Harness  # noqa: E402


CP_PORT = 8320
AGENTS = 5
JOBS = 4
UNITS_PER_JOB = 20
TIMEOUT_S = 120.0


def _submit(cp: str, job_idx: int) -> str:
    payloads = [{"x": (job_idx * 1000) + i} for i in range(UNITS_PER_JOB)]
    r = httpx.post(
        f"{cp}/api/v1/jobs",
        json={
            "name": f"scale-square-{job_idx}",
            "func": "square",
            "payloads": payloads,
            "cpu_pct": 10,
            "ram_mb": 64,
        },
        timeout=10,
    )
    r.raise_for_status()
    return r.json()["job_id"]


def _job(cp: str, job_id: str) -> dict:
    r = httpx.get(f"{cp}/api/v1/jobs/{job_id}", timeout=10)
    r.raise_for_status()
    return r.json()


def _wait_all_jobs(cp: str, job_ids: list[str], timeout: float) -> dict[str, dict]:
    deadline = time.time() + timeout
    latest: dict[str, dict] = {}
    terminal = {"completed", "completed_with_failures"}
    while time.time() < deadline:
        latest = {jid: _job(cp, jid) for jid in job_ids}
        if all(j["status"] in terminal for j in latest.values()):
            return latest
        time.sleep(0.5)
    statuses = {jid: j.get("status") for jid, j in latest.items()}
    raise TimeoutError(f"jobs did not all finish within {timeout:.0f}s: {statuses}")


def _check_square_results(jobs: dict[str, dict]) -> None:
    for job in jobs.values():
        for unit in job["units"]:
            x = unit["payload"]["x"]
            assert unit.get("result") == x * x, (
                f"bad result for unit {unit['id']}: {unit.get('result')} != {x * x}"
            )


def _print_node_spread(nodes: list[dict], spread: Counter) -> None:
    names = {n["id"]: n["name"] for n in nodes}
    print("  observed unit spread:")
    for node in nodes:
        print(f"    - {node['name']:12} ran {spread.get(node['id'], 0)} units")
    unknown = {nid: count for nid, count in spread.items() if nid not in names}
    for nid, count in unknown.items():
        print(f"    - {nid:12} ran {count} units")


def main() -> int:
    data_dir = Path(__file__).resolve().parent / f"_data_scale_{os.getpid()}_{int(time.time() * 1000)}"
    h = Harness(n_agents=AGENTS, cp_port=CP_PORT, scale="home", data_dir=str(data_dir))
    ok = True
    try:
        print(f"== starting scale cluster: {AGENTS} agents, LAN/home mode, port {CP_PORT} ==")
        h.start()
        cp = h.cp_url()
        nodes = h.wait_nodes_online(AGENTS, timeout=30)
        print(f"  [PASS] {len(nodes)} nodes online")

        print(f"== submitting {JOBS} concurrent square jobs x {UNITS_PER_JOB} units ==")
        job_ids: list[str] = []
        with ThreadPoolExecutor(max_workers=JOBS) as pool:
            futures = [pool.submit(_submit, cp, i) for i in range(JOBS)]
            for f in as_completed(futures):
                job_ids.append(f.result())
        print(f"  [PASS] submitted {len(job_ids)} jobs: {', '.join(job_ids)}")

        jobs = _wait_all_jobs(cp, job_ids, TIMEOUT_S)
        print("  [PASS] all jobs reached terminal state")

        total_units = JOBS * UNITS_PER_JOB
        done = sum(1 for j in jobs.values() for u in j["units"] if u["status"] == "done")
        failed = sum(1 for j in jobs.values() for u in j["units"] if u["status"] == "failed")
        observed_total = sum(len(j["units"]) for j in jobs.values())
        statuses = {jid: j["status"] for jid, j in jobs.items()}
        assert observed_total == total_units, f"unit count mismatch: {observed_total} != {total_units}"
        assert all(s == "completed" for s in statuses.values()), f"job statuses: {statuses}"
        assert done == total_units, f"done mismatch: {done} != {total_units}"
        assert failed == 0, f"expected 0 failed units, got {failed}"
        _check_square_results(jobs)
        print(f"  [PASS] correct accounting: done={done}, failed={failed}, total={observed_total}")

        spread: Counter = Counter(
            u.get("assigned_node")
            for j in jobs.values()
            for u in j["units"]
            if u.get("assigned_node")
        )
        _print_node_spread(nodes, spread)
        missing = [n["name"] for n in nodes if spread.get(n["id"], 0) <= 0]
        assert not missing, f"expected every node to execute >0 units; missing {missing}"
        assert sum(spread.values()) == total_units, (
            f"assigned-node accounting mismatch: {sum(spread.values())} != {total_units}"
        )
        print(f"  [PASS] all {AGENTS} nodes executed work; no unit lost")
        print("\nALL SCALE ACCEPTANCE CHECKS PASSED")
    except Exception as exc:  # noqa: BLE001
        ok = False
        print(f"\nFAILED: {type(exc).__name__}: {exc}")
        import traceback

        traceback.print_exc()
    finally:
        h.stop()
        shutil.rmtree(data_dir, ignore_errors=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
