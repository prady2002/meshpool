"""Acceptance gate: DAG work stealing and spot/preemption semantics.

Runs real control-plane and agent processes.  The gate has non-vacuity switches:

* ``MESHPOOL_BATCH_SPECULATION=0`` is used internally for the baseline run; the
  measured speculative run must dispatch a duplicate to a different node and finish
  faster without changing the accepted ledger total.
* Setting ``MESHPOOL_DISABLE_RECLAIM_REQUEUE=1`` before invoking this script makes
  the preemption assertion fail because reclaimed-node units are not immediately
  moved off the node.
"""
from __future__ import annotations

import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sim.harness import Harness  # noqa: E402


BASE = Path(__file__).resolve().parent


@contextmanager
def _env(values: dict[str, str | None]):
    old = {k: os.environ.get(k) for k in values}
    try:
        for k, v in values.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _submit_dag(cp: str, name: str, stages: list[dict]) -> str:
    r = httpx.post(f"{cp}/api/v1/dags", json={"name": name, "stages": stages}, timeout=10)
    r.raise_for_status()
    return r.json()["dag_id"]


def _dag(cp: str, dag_id: str) -> dict:
    r = httpx.get(f"{cp}/api/v1/dags/{dag_id}", timeout=10)
    r.raise_for_status()
    return r.json()


def _job(cp: str, job_id: str) -> dict:
    r = httpx.get(f"{cp}/api/v1/jobs/{job_id}", timeout=10)
    r.raise_for_status()
    return r.json()


def _cluster(cp: str) -> dict:
    r = httpx.get(f"{cp}/api/v1/cluster", timeout=10)
    r.raise_for_status()
    return r.json()


def _ledger_total(cp: str) -> float:
    r = httpx.get(f"{cp}/api/v1/ledger", timeout=10)
    r.raise_for_status()
    return float(r.json()["summary"]["donated"])


def _stage(dag: dict, stage_id: str) -> dict:
    return next(s for s in dag["stages"] if s["id"] == stage_id)


def _wait_done(cp: str, dag_id: str, timeout: float = 90.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        d = _dag(cp, dag_id)
        if str(d["status"]).startswith("completed"):
            return d
        time.sleep(0.25)
    raise TimeoutError(f"DAG {dag_id} did not finish: {_dag(cp, dag_id)}")


def _run_speculation_case(enabled: bool, port: int) -> dict:
    data_dir = str(BASE / f"_data_dag_worksteal_{'on' if enabled else 'off'}")
    env = {
        "MESHPOOL_BATCH_SPECULATION": "1" if enabled else "0",
        "MESHPOOL_BATCH_SPECULATION_MIN_COMPLETED": "1",
        "MESHPOOL_BATCH_SPECULATION_FLOOR_MS": "300",
        "MESHPOOL_BATCH_SPECULATION_MULTIPLIER": "1.0",
        "MESHPOOL_DISABLE_RECLAIM_REQUEUE": None,
    }
    with _env(env):
        h = Harness(n_agents=3, cp_port=port, data_dir=data_dir)
        try:
            h.start()
            cp = h.cp_url()
            payloads = [
                {
                    "x": i,
                    "_meshpool_slow_node_name": "sim-node-1",
                    "_meshpool_slow_extra_s": 8.0,
                    "_meshpool_billing_ms": 10,
                }
                for i in range(8)
            ]
            dag_id = _submit_dag(cp, f"worksteal-{'on' if enabled else 'off'}", [
                {"id": "map", "func": "square", "payloads": payloads,
                 "cpu_pct": 10, "ram_mb": 64},
            ])
            t0 = time.perf_counter()
            final = _wait_done(cp, dag_id)
            elapsed = time.perf_counter() - t0
            map_stage = _stage(final, "map")
            job = _job(cp, map_stage["job_id"])
            summary = _cluster(cp)["summary"]
            pairs = summary.get("speculative_pairs", [])
            return {
                "elapsed": elapsed,
                "dag": final,
                "job": job,
                "ledger": _ledger_total(cp),
                "attempts": summary.get("speculative_attempts", 0),
                "wins": summary.get("speculative_wins", 0),
                "pairs": pairs,
                "used_slow": sum(1 for u in job["units"]
                                 if u.get("assigned_node")
                                 and any(n["id"] == u["assigned_node"] and n["name"] == "sim-node-1"
                                         for n in _cluster(cp)["nodes"])),
            }
        finally:
            h.stop()


def _run_preemption_case(port: int) -> dict:
    data_dir = str(BASE / "_data_dag_preemption")
    env = {
        "MESHPOOL_BATCH_SPECULATION": "0",
        "MESHPOOL_BATCH_SPECULATION_MIN_COMPLETED": "1",
        "MESHPOOL_BATCH_SPECULATION_FLOOR_MS": "300",
        "MESHPOOL_BATCH_SPECULATION_MULTIPLIER": "1.0",
    }
    if os.environ.get("MESHPOOL_DAG_DISABLE_REQUEUE_PROOF") == "1":
        env["MESHPOOL_DISABLE_RECLAIM_REQUEUE"] = "1"
    with _env(env):
        h = Harness(n_agents=3, cp_port=port, data_dir=data_dir)
        try:
            h.start()
            cp = h.cp_url()
            dag_id = _submit_dag(cp, "preempt-dag", [
                {"id": "warmup", "func": "square",
                 "payloads": [{"x": i} for i in range(4)], "cpu_pct": 10, "ram_mb": 64},
                {"id": "work", "func": "sleep_echo", "depends_on": ["warmup"],
                 "payloads": [{"seconds": 5.0, "echo": i} for i in range(10)],
                 "cpu_pct": 10, "ram_mb": 64},
            ])

            deadline = time.time() + 45
            chosen_node = None
            before = 0
            work_job_id = None
            while time.time() < deadline:
                d = _dag(cp, dag_id)
                if _stage(d, "warmup")["status"] != "succeeded":
                    time.sleep(0.2)
                    continue
                work = _stage(d, "work")
                if work["job_id"]:
                    work_job_id = work["job_id"]
                    j = _job(cp, work_job_id)
                    counts: dict[str, int] = {}
                    for u in j["units"]:
                        if u["status"] in ("assigned", "running") and u.get("assigned_node"):
                            counts[u["assigned_node"]] = counts.get(u["assigned_node"], 0) + 1
                    if counts:
                        chosen_node, before = max(counts.items(), key=lambda kv: kv[1])
                        break
                time.sleep(0.2)
            assert chosen_node and before > 0, "non-vacuity: no active DAG units to reclaim"

            requeued_before = int(_cluster(cp)["summary"].get("requeued_units", 0))
            r = httpx.post(f"{cp}/api/v1/nodes/{chosen_node}/reclaim", json={}, timeout=10)
            r.raise_for_status()
            time.sleep(0.3)
            requeued_after = int(_cluster(cp)["summary"].get("requeued_units", 0))
            after_job = _job(cp, work_job_id)
            after = sum(1 for u in after_job["units"]
                        if u["status"] in ("assigned", "running")
                        and u.get("assigned_node") == chosen_node)
            moved = before - after
            requeued_delta = requeued_after - requeued_before
            assert requeued_delta >= before, (
                f"reclaim did not synchronously requeue the active units: "
                f"before={before}, requeued_delta={requeued_delta}"
            )
            assert moved > 0, (
                f"reclaim did not immediately move active units off node {chosen_node}: "
                f"before={before}, after={after}"
            )

            final = _wait_done(cp, dag_id, timeout=90)
            warmup = _stage(final, "warmup")
            work = _stage(final, "work")
            work_job = _job(cp, work["job_id"])
            echoes = sorted(u["result"]["echo"] for u in work_job["units"])
            return {
                "reclaimed_node": chosen_node,
                "before": before,
                "after": after,
                "moved": moved,
                "requeued_delta": requeued_delta,
                "final": final,
                "warmup_exec": warmup["execution_count"],
                "work_exec": work["execution_count"],
                "echoes": echoes,
            }
        finally:
            h.stop()


def main() -> int:
    ok = True
    try:
        print("== work stealing: baseline with speculation OFF ==")
        off = _run_speculation_case(False, 8324)
        print(f"  elapsed={off['elapsed']:.3f}s ledger={off['ledger']:.3f} "
              f"spec_attempts={off['attempts']} wins={off['wins']}")
        assert off["attempts"] == 0, "baseline was not actually speculation-off"
        assert off["elapsed"] > 5.0, (
            f"non-vacuity: slow-node delay did not activate (elapsed={off['elapsed']:.3f}s)")

        print("\n== work stealing: speculation ON ==")
        on = _run_speculation_case(True, 8328)
        print(f"  elapsed={on['elapsed']:.3f}s ledger={on['ledger']:.3f} "
              f"spec_attempts={on['attempts']} wins={on['wins']}")
        print(f"  speculative pairs={on['pairs']}")
        assert on["attempts"] > 0 and on["wins"] > 0, (
            "non-vacuity: no speculative duplicate both ran and won")
        assert any(p.get("original") != p.get("backup") for p in on["pairs"]), (
            "backup task was not dispatched to a different node")
        assert on["elapsed"] < off["elapsed"], (
            f"speculation did not improve wall-clock: off={off['elapsed']:.3f}s "
            f"on={on['elapsed']:.3f}s")
        assert abs(on["ledger"] - off["ledger"]) <= 0.001, (
            f"speculation changed accepted ledger total: off={off['ledger']} on={on['ledger']}")
        print("  PASS: backup ran elsewhere, finished faster, and ledger total matched")

        print("\n== preemption: reclaim active DAG node ==")
        pre = _run_preemption_case(8332)
        print(f"  reclaimed_node={pre['reclaimed_node']} before={pre['before']} "
              f"after={pre['after']} moved={pre['moved']} "
              f"requeued_delta={pre['requeued_delta']}")
        print(f"  warmup_execution_count={pre['warmup_exec']} "
              f"work_execution_count={pre['work_exec']}")
        assert pre["final"]["status"] == "completed", pre["final"]["status"]
        assert pre["warmup_exec"] == 4, (
            "completed warmup stage was recomputed during preemption")
        assert pre["echoes"] == list(range(10)), pre["echoes"]
        print("  PASS: reclaimed-node units moved, DAG completed, completed stage not recomputed")

        print("\nALL DAG WORK STEALING / PREEMPTION CHECKS PASSED")
    except Exception as exc:  # noqa: BLE001
        ok = False
        print(f"\nFAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
