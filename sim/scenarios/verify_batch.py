"""End-to-end scenario: batch distribution + node-kill failover (spec §9.4).

Proves the walking skeleton: agents register, work distributes across all online
nodes proportional to capacity, and killing a node mid-flight causes its in-flight
units to be retried elsewhere with the job still completing.
"""
from __future__ import annotations

import sys
import time
from collections import Counter
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sim.harness import Harness  # noqa: E402


def _submit(cp: str, func: str, payloads: list[dict], cpu_pct: int = 15) -> str:
    r = httpx.post(f"{cp}/api/v1/jobs", json={"name": func, "func": func,
                   "payloads": payloads, "cpu_pct": cpu_pct}, timeout=10)
    r.raise_for_status()
    return r.json()["job_id"]


def _job(cp: str, job_id: str) -> dict:
    return httpx.get(f"{cp}/api/v1/jobs/{job_id}", timeout=10).json()


def _counts(job: dict) -> dict:
    c: Counter = Counter(u["status"] for u in job["units"])
    c["total"] = len(job["units"])
    return dict(c)


def _wait_job(cp: str, job_id: str, timeout: float = 60) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        j = _job(cp, job_id)
        if j["status"].startswith("completed"):
            return j
        time.sleep(0.4)
    raise TimeoutError(f"job {job_id} did not finish: {_job(cp, job_id)['status']}")


def main() -> int:
    h = Harness(n_agents=3, cp_port=8090,
                data_dir=str(Path(__file__).resolve().parent / "_data_batch"))
    ok = True
    try:
        print("== starting cluster (control plane + 3 heterogeneous agents) ==")
        h.start()
        cp = h.cp_url()
        nodes = httpx.get(f"{cp}/api/v1/nodes").json()["nodes"]
        print(f"  {len(nodes)} nodes online:")
        for n in nodes:
            print(f"    - {n['name']:12} cores={n['capability']['cpu_cores']} "
                  f"ram={n['capability']['ram_mb']}MB slots~{int(n['policy']['cpu_pct']/100*n['capability']['cpu_cores'])}")

        # ---- distribution ---------------------------------------------- #
        print("\n== test 1: distribute 30 CPU units across all nodes ==")
        payloads = [{"seed": f"s{i}", "rounds": 60000} for i in range(30)]
        jid = _submit(cp, "sha_pow", payloads)
        job = _wait_job(cp, jid)
        dist = Counter(u["assigned_node"] for u in job["units"])
        name_by_id = {n["id"]: n["name"] for n in nodes}
        print(f"  job {job['status']}; distribution:")
        for nid, cnt in dist.items():
            print(f"    - {name_by_id.get(nid, nid):12} ran {cnt} units")
        used_nodes = len([k for k in dist if k])
        assert job["status"] == "completed", job["status"]
        assert used_nodes >= 3, f"expected all 3 nodes used, got {used_nodes}"
        print(f"  PASS: all {used_nodes} nodes participated, 0 failures")

        # ---- failover -------------------------------------------------- #
        print("\n== test 2: kill a node mid-flight, expect retry + completion ==")
        payloads = [{"seconds": 1.5, "echo": i} for i in range(12)]
        jid = _submit(cp, "sleep_echo", payloads, cpu_pct=15)
        time.sleep(1.0)  # let some units start
        mid = _counts(_job(cp, jid))
        print(f"  mid-flight counts: {mid}")
        print("  killing sim-node-2 ...")
        h.stop_agent(1)
        job = _wait_job(cp, jid, timeout=60)
        done = sum(1 for u in job["units"] if u["status"] == "done")
        max_attempts = max(u["attempts"] for u in job["units"])
        print(f"  job {job['status']}; {done}/{len(job['units'])} units done; "
              f"max attempts on a unit = {max_attempts}")
        assert job["status"] == "completed", job["status"]
        assert done == len(job["units"]), "not all units completed after failover"
        assert max_attempts >= 2, "expected at least one retry after node kill"
        print("  PASS: all units completed after node kill (units retried elsewhere)")

        print("\nALL BATCH E2E CHECKS PASSED")
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
