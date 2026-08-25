"""Trust scoring + contribution ledger (Tier 0 active, Tier 1 fields inert)."""
from __future__ import annotations

from meshpool.common.models import LedgerEntry, new_id
from meshpool.common.trust import (ContributionLedger, TrustEngine, fair_share_priority,
                                    update_ema)


def test_update_ema_moves_toward_sample():
    assert update_ema(1.0, 0.0, alpha=0.5) == 0.5
    # repeated failures decay a perfect score downward
    s = 1.0
    for _ in range(5):
        s = update_ema(s, 0.0, alpha=0.2)
    assert 0.0 < s < 1.0


def test_fair_share_tier0_constant_tier1_monotonic():
    assert fair_share_priority(0.0, tier=0) == 1.0
    assert fair_share_priority(999.0, tier=0) == 1.0  # single owner: no contention
    lo = fair_share_priority(1.0, tier=1)
    hi = fair_share_priority(100.0, tier=1)
    assert hi > lo  # more contribution => higher priority under contention


def test_tier0_score_is_pass_through(tmp_store):
    eng = TrustEngine(tmp_store, tier=0)
    eng.record_task_result("n1", success=False)  # recorded for UI...
    assert eng.score("n1") == 1.0                # ...but Tier 0 trusts fully
    assert eng.needs_redundant_execution("n1") is False


def test_tier1_score_reflects_failures(tmp_store):
    eng = TrustEngine(tmp_store, tier=1)
    for _ in range(20):
        eng.record_task_result("bad", success=False)
        eng.record_benchmark("bad", passed=False)
    assert eng.score("bad") < 1.0


def test_tier2_new_stranger_is_on_probation(tmp_store):
    """A never-seen device at Tier 2 must be verified until it proves itself, even
    though its EMA fields default to a perfect 1.0."""
    eng = TrustEngine(tmp_store, tier=2)
    assert eng.needs_verification("newcomer") is True         # 0 samples < warmup
    # a good actor graduates after enough passing samples...
    for _ in range(10):
        eng.record_benchmark("good", passed=True)
    assert eng.needs_verification("good") is False
    # ...a liar's reputation collapses, so it stays permanently verified.
    for _ in range(12):
        eng.record_benchmark("liar", passed=False)
        eng.record_task_result("liar", success=False)
    assert eng.needs_verification("liar") is True
    assert eng.score("liar") < 0.6


def test_tier0_and_tier1_verification_gate(tmp_store):
    assert TrustEngine(tmp_store, tier=0).needs_verification("x") is False  # full trust
    # Tier 1 only verifies clearly-degraded nodes, not fresh ones.
    eng = TrustEngine(tmp_store, tier=1)
    assert eng.needs_verification("fresh") is False


def test_ledger_tracks_credits(tmp_store):
    ledger = ContributionLedger(tmp_store)
    tmp_store.add_ledger(LedgerEntry(
        id=new_id("led"), node_id="n1", owner="alice",
        compute_units=10.0, kind="donated"))
    tmp_store.add_ledger(LedgerEntry(
        id=new_id("led"), node_id="n1", owner="alice",
        compute_units=4.0, kind="consumed"))
    # net credits = donated - consumed (sign applied by the store query)
    assert abs(ledger.credits("alice") - 6.0) < 1e-9
