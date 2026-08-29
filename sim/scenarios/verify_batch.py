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


_IN_FLIGHT = {"running", "assigned"}


def _await_inflight_victim(cp: str, job_id: str, name_by_id: dict,
                           timeout: float = 45.0) -> tuple[int, str, dict]:
    """Block until a unit is observably in flight; return the node hosting it.

    Replaces ``time.sleep(1.0)  # let some units start`` followed by killing a
    hardcoded node. That was racy in two independent ways, and both produce the same
    misleading outcome -- a failure blamed on the product:

    * On a loaded machine no unit may have started within the fixed second, so the
      kill lands on an idle node.
    * Even when units *had* started, nothing guaranteed one was on the node the test
      chose to kill.

    In either case no work is lost, so nothing is ever retried, and the
    ``max_attempts >= 2`` assertion fails against a system that is behaving
    perfectly. Choosing the victim from observed server state removes both races:
    the node killed is, by construction, one that is running a unit right now.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = _job(cp, job_id)
        for u in job["units"]:
            name = name_by_id.get(u.get("assigned_node") or "", "")
            if str(u["status"]).lower() in _IN_FLIGHT and name.startswith("sim-node-"):
                return int(name.rsplit("-", 1)[1]) - 1, name, _counts(job)
        if str(job["status"]).startswith("completed"):
            raise AssertionError(
                "the job finished before any unit was seen in flight, so there was "
                "no live work to interrupt; this test cannot observe failover")
        time.sleep(0.05)
    raise TimeoutError(
        f"no unit of job {job_id} entered a running state within {timeout:.0f}s")


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
        victim_idx, victim_name, mid = _await_inflight_victim(cp, jid, name_by_id)
        print(f"  mid-flight counts: {mid}")
        print(f"  killing {victim_name} (observed running a unit right now) ...")
        h.stop_agent(victim_idx)
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
