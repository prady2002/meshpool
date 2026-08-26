"""Acceptance gate: fraud detection slashes a node that returns wrong results.

Scope: in-process control-plane machinery, no fake green path.  A deterministic
batch unit is run through the real redundant-verification code with one lying
node.  The gate proves detection, slashing as a balanced ledger transfer, and
two non-vacuity controls: an honest node is not slashed, and disabling the
detector lets the lie through.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from meshpool.batch.engine import BatchManager  # noqa: E402
from meshpool.batch.functions import run_function  # noqa: E402
from meshpool.common.db import Store  # noqa: E402
from meshpool.common.models import BatchJob, UnitStatus, WorkUnit, now_ms  # noqa: E402
from meshpool.common.trust import ContributionLedger, TrustEngine  # noqa: E402
from meshpool.controlplane.verification import VerificationPolicy, Verifier  # noqa: E402


class _Events:
    def publish(self, *_, **__):
        pass


class _Node:
    def __init__(self, node_id: str):
        self.id = node_id
        self.name = node_id
        self.owner = f"dev:{node_id}"
        self.status = "online"
        self.capability = type("C", (), {"cpu_cores": 4, "ram_mb": 4096})()
        self.policy = type("P", (), {"cpu_pct": 100, "ram_mb": 4096})()

    def donatable_cpu_slots(self):
        return 4

    def provides(self, workload_class):
        return workload_class == "batch"


class _Registry:
    def __init__(self, nodes):
        self.nodes = nodes

    def schedulable_nodes(self):
        return list(self.nodes)

    def online(self):
        return list(self.nodes)


class _BatchStore:
    def list_jobs(self):
        return []

    def put_job(self, _):
        pass

    def credits_by_owner(self):
        return {}

    def get_job(self, _):
        return None


def _corrupt(v):
    return v + 1 if isinstance(v, (int, float)) else v


class _Transport:
    def __init__(self, liars):
        self.liars = set(liars)
        self.calls = 0

    async def call(self, node_id, op, payload, timeout=None):
        self.calls += 1
        res = run_function(payload["func"], payload["payload"])
        if node_id in self.liars:
            res = _corrupt(res)
        return {"result": res, "compute_ms": 1}


def _make_job():
    unit = WorkUnit(id="u1", payload={"x": 9}, status=UnitStatus.PENDING)
    job = BatchJob(id="job1", name="fraud", func="square", units=[unit],
                   cpu_pct_per_unit=10, ram_mb_per_unit=64, owner="cons:alice",
                   status="running", created_ms=now_ms())
    return job, unit


def _run(liars: set[str], primary_id: str, redundant_rate: float) -> dict:
    nodes = [_Node("liar"), _Node("honest1"), _Node("honest2")]
    store = Store(":memory:")
    ledger = ContributionLedger(store)
    trust = TrustEngine(store, tier=2)
    verifier = Verifier(ledger=ledger, policy=VerificationPolicy(redundant_rate),
                        trust=trust)
    mgr = BatchManager(_Registry(nodes), _BatchStore(), ledger, trust, _Events(),
                       _Transport(liars))
    mgr._verify_sample_rate = 0.0
    mgr.verifier = verifier
    primary = next(n for n in nodes if n.id == primary_id)
    job, unit = _make_job()
    expected_value = 5.0
    verifier.ensure_stake(primary.id, primary.owner, expected_value)
    result = run_function("square", unit.payload)
    if primary.id in liars:
        result = _corrupt(result)
    verdict = asyncio.run(mgr._redundant_check(
        job, unit, result, primary, expected_job_value=expected_value))
    totals = store.ledger_totals_by_owner()
    store.close()
    return {
        "verdict": verdict,
        "calls": mgr.transport.calls,
        "penalized": verifier.stats.nodes_penalized,
        "slashed": verifier.stats.stake_slashed,
        "totals": totals,
    }


def main() -> int:
    ok = True
    try:
        print("== test 1: dishonest primary is detected and slashed ==")
        bad = _run({"liar"}, "liar", redundant_rate=1.0)
        assert bad["calls"] >= 2, f"redundant path did not execute: {bad}"
        assert bad["verdict"] is False, bad
        assert bad["penalized"] == 1, bad
        assert bad["slashed"] > 0.0, bad
        assert bad["totals"]["meshpool:slashing"]["donated"] == bad["slashed"], bad
        print(f"  PASS: liar detected, penalized={bad['penalized']}, "
              f"slashed={bad['slashed']}")

        print("\n== test 2: NON-VACUITY honest primary is not slashed ==")
        honest = _run(set(), "honest1", redundant_rate=1.0)
        assert honest["calls"] >= 1, f"redundant path did not execute: {honest}"
        assert honest["verdict"] is True, honest
        assert honest["penalized"] == 0, honest
        assert honest["slashed"] == 0.0, honest
        print("  PASS: honest primary verified without penalty")

        print("\n== test 3: NON-VACUITY detector can fail when checks are disabled ==")
        disabled = _run({"liar"}, "liar", redundant_rate=0.0)
        assert disabled["calls"] == 0, disabled
        assert disabled["verdict"] is None, disabled
        assert disabled["penalized"] == 0 and disabled["slashed"] == 0.0, disabled
        print("  PASS: disabled detector misses the liar, so test 1 is meaningful")

        print("\nALL FRAUD DETECTION CHECKS PASSED")
    except Exception as exc:  # noqa: BLE001
        ok = False
        print(f"\nFAILED: {type(exc).__name__}: {exc}")
        import traceback
        traceback.print_exc()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
