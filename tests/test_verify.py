"""Tier-2 result verification + credit conservation in the batch engine.

On the public mesh the coordinator recomputes a stranger's *deterministic* unit
against a local oracle and rejects a mismatch, so a lying node can neither get its
bad result accepted nor get paid — while the ledger stays conserved (every credit
spent by a consumer was earned by a donor).
"""
from __future__ import annotations

import asyncio

from meshpool.batch.engine import (
    VERIFIABLE_FUNCS, BatchManager, _results_match,
)
from meshpool.batch.functions import run_function
from meshpool.common.models import BatchJob, UnitStatus, WorkUnit, now_ms
from meshpool.common.trust import ContributionLedger, TrustEngine


class _Events:
    def publish(self, *a, **k):  # noqa: D401 - test stub
        pass


class _Node:
    def __init__(self, node_id: str, owner: str):
        self.id = node_id
        self.owner = owner


def _corrupt(v):
    if isinstance(v, bool):
        return not v
    if isinstance(v, (int, float)):
        return v + 1
    if isinstance(v, dict):
        return {k: _corrupt(x) for k, x in v.items()}
    if isinstance(v, list):
        return v[1:] if v else [1]
    if isinstance(v, str):
        return v + "X"
    return v


class _Transport:
    """Runs the real function, optionally corrupting the answer (a dishonest node)."""

    def __init__(self, honest: bool = True):
        self.honest = honest
        self.calls = 0

    async def call(self, node_id, op, payload, timeout=None):
        self.calls += 1
        res = run_function(payload["func"], payload["payload"])
        return {"result": res if self.honest else _corrupt(res)}


def _mgr(store, tier, transport):
    return BatchManager(
        registry=None, store=store, ledger=ContributionLedger(store),
        trust=TrustEngine(store, tier=tier), events=_Events(), transport=transport)


def _job(func="square", payload=None, owner="cons:alice"):
    unit = WorkUnit(id="u1", payload=payload or {"x": 6}, status=UnitStatus.PENDING)
    job = BatchJob(id="job1", name="j", func=func, units=[unit],
                   cpu_pct_per_unit=10, ram_mb_per_unit=64, owner=owner,
                   status="running", created_ms=now_ms())
    return job, unit


# --- structural comparison ------------------------------------------------- #
def test_results_match_tolerates_json_coercions():
    assert _results_match(1.0000001, 1.0) is True          # float rounding
    assert _results_match([1, 2, 3], (1, 2, 3)) is True     # tuple -> list
    assert _results_match({"a": 1, "b": [1.0]}, {"a": 1, "b": [1.0]}) is True
    assert _results_match(2, 3) is False
    assert _results_match({"n": 7, "prime": True}, {"n": 7, "prime": False}) is False
    assert _results_match({"a": 1}, {"a": 1, "b": 2}) is False


# --- when verification applies --------------------------------------------- #
def test_should_verify_only_for_tier2_verifiable_unproven(tmp_store):
    mgr2 = _mgr(tmp_store, 2, _Transport())
    job, _ = _job("square")
    assert mgr2._should_verify(job, "fresh") is True        # new stranger, deterministic
    sleepy, _ = _job("sleep_echo", payload={"seconds": 0})
    assert mgr2._should_verify(sleepy, "fresh") is False    # non-deterministic func
    assert "sleep_echo" not in VERIFIABLE_FUNCS

    mgr0 = _mgr(tmp_store, 0, _Transport())
    assert mgr0._should_verify(job, "fresh") is False        # LAN: full trust, no checks


# --- honest node: earns, consumer spends, ledger conserves ----------------- #
def test_honest_unit_credits_donor_and_consumer(tmp_store):
    mgr = _mgr(tmp_store, 2, _Transport(honest=True))
    job, unit = _job("square", owner="cons:alice")
    node = _Node("nodeA", owner="dev:donorA")

    asyncio.run(mgr._run_unit(job, unit, node))

    assert unit.status is UnitStatus.DONE and unit.result == 36
    totals = tmp_store.ledger_totals_by_owner()
    donated = totals["dev:donorA"]["donated"]
    consumed = totals["cons:alice"]["consumed"]
    assert donated > 0 and abs(donated - consumed) < 1e-9   # conserved
    # Passed the spot-check. Trust is beta reputation with decay (plan 6.2), not an
    # EMA, so a single honest observation moves the score above the 0.5 prior but
    # cannot reach certainty -- that ceiling is exactly what stops a Sybil buying
    # trust with a farm of cheap jobs.
    scored = mgr.trust.score("nodeA")
    assert 0.5 < scored < 1.0, scored


# --- lying node: rejected, penalised, paid nothing ------------------------- #
def test_lying_unit_is_rejected_and_uncredited(tmp_store):
    mgr = _mgr(tmp_store, 2, _Transport(honest=False))
    job, unit = _job("square", owner="cons:bob")
    node = _Node("liar", owner="dev:liar")

    asyncio.run(mgr._run_unit(job, unit, node))

    assert unit.status is UnitStatus.PENDING                # requeued for a retry
    assert "verification" in (unit.error or "")
    assert tmp_store.ledger_totals_by_owner() == {}         # nobody was paid
    assert mgr.trust.score("liar") < 1.0                    # reputation penalised


# --- Tier 0 is unchanged: no verification, no gating ----------------------- #
def test_tier0_accepts_without_verification(tmp_store):
    # Even a "wrong" result is accepted at Tier 0 (single owner / full trust) — this
    # is the byte-for-byte-unchanged LAN behaviour the global tiers are layered on.
    mgr = _mgr(tmp_store, 0, _Transport(honest=False))
    job, unit = _job("square", owner="default")
    node = _Node("lan1", owner="default")

    asyncio.run(mgr._run_unit(job, unit, node))

    assert unit.status is UnitStatus.DONE                   # not spot-checked
    assert mgr.transport.calls == 1


def test_credit_conservation_across_many_units(tmp_store):
    mgr = _mgr(tmp_store, 2, _Transport(honest=True))
    node = _Node("nodeC", owner="dev:donorC")
    for i in range(6):
        job, unit = _job("is_prime", payload={"n": 7 + i}, owner="cons:carol")
        asyncio.run(mgr._run_unit(job, unit, node))
    totals = tmp_store.ledger_totals_by_owner()
    total_donated = sum(t["donated"] for t in totals.values())
    total_consumed = sum(t["consumed"] for t in totals.values())
    assert total_donated > 0
    assert abs(total_donated - total_consumed) < 1e-9
