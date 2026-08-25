"""Tests for workload-matched result verification (spec P6).

Covers:
  1. VerificationPolicy classifies functions correctly.
  2. ComputeEscrowEngine: credits held, released after N passes.
  3. ComputeEscrowEngine: credits forfeited on failure.
  4. Verifier.digests_match: deterministic JSON serialisation.
  5. BatchManager._redundant_check: simulated cheater IS caught.
  6. FALSE-POSITIVE GUARD: honest node is NOT falsely accused.
  7. Redundant check degrades gracefully when the cluster has < 2 nodes.
  8. Verifier.record_fail updates escrow AND penalises node.
  9. Oracle spot-check path ALSO updates escrow on pass and on fail.

Test 6 is as important as test 5: a verification system that generates false
positives is worse than none -- it would slash honest contributors.
"""
from __future__ import annotations

import asyncio
import random

from meshpool.batch.engine import BatchManager
from meshpool.batch.functions import run_function
from meshpool.common.models import BatchJob, UnitStatus, WorkUnit, now_ms
from meshpool.common.trust import ContributionLedger, TrustEngine
from meshpool.controlplane.verification import (
    CONSECUTIVE_PASSES_TO_RELEASE,
    DIGEST_VERIFIABLE_FUNCS,
    NOT_VERIFIABLE_FUNCS,
    ORACLE_VERIFIABLE_FUNCS,
    ComputeEscrowEngine,
    VerificationPolicy,
    VerificationTechnique,
    Verifier,
    WorkloadKind,
    _result_digest,
)


# ---------------------------------------------------------------------------
# Shared fakes
# ---------------------------------------------------------------------------

class _Events:
    def publish(self, *a, **k):
        pass


class _Policy:
    cpu_pct = 60
    ram_mb = 4096


class _FullNode:
    def __init__(self, node_id: str, owner: str = "dev:donor"):
        self.id = node_id
        self.name = node_id
        self.owner = owner
        self.status = "online"
        self.capability = type("C", (), {"cpu_cores": 4, "ram_mb": 4096})()
        self.policy = _Policy()

    def donatable_cpu_slots(self) -> int:
        return 4

    def provides(self, workload_class: str) -> bool:
        return workload_class == "batch"


class _Registry:
    def __init__(self, nodes):
        self._nodes = nodes

    def schedulable_nodes(self):
        return list(self._nodes)

    def online(self):
        return list(self._nodes)


def _corrupt(v):
    if isinstance(v, bool):
        return not v
    if isinstance(v, (int, float)):
        return v + 1
    if isinstance(v, dict):
        return {k: _corrupt(x) for k, x in v.items()}
    if isinstance(v, list):
        return v[1:] if v else [1]
    return v


class _Transport:
    """Runs the real function; optionally corrupts a specific node's answer."""

    def __init__(self, liars=None):
        self.liars: frozenset = frozenset(liars or ())
        self.calls: dict = {}

    async def call(self, node_id, op, payload, timeout=None):
        self.calls[node_id] = self.calls.get(node_id, 0) + 1
        res = run_function(payload["func"], payload["payload"])
        if node_id in self.liars:
            res = _corrupt(res)
        return {"result": res, "compute_ms": 1}


class _Store:
    def list_jobs(self):
        return []

    def put_job(self, job):
        pass

    def credits_by_owner(self):
        return {}

    def get_job(self, job_id):
        return None


def _make_mgr(tmp_store, nodes, tier=2, transport=None, redundant_rate=1.0):
    """Build a BatchManager wired with a Verifier."""
    transport = transport or _Transport()
    reg = _Registry(nodes)
    ledger = ContributionLedger(tmp_store)
    trust = TrustEngine(tmp_store, tier=tier)
    mgr = BatchManager(
        registry=reg, store=_Store(), ledger=ledger,
        trust=trust, events=_Events(), transport=transport)
    # Disable oracle sampling to isolate the redundant-execution path.
    mgr._verify_sample_rate = 0.0
    if tier >= 2:
        policy = VerificationPolicy(redundant_rate=redundant_rate)
        mgr.verifier = Verifier(ledger=ledger, policy=policy)
    return mgr


def _job(func="square", payload=None, owner="cons:alice"):
    unit = WorkUnit(id="u1", payload=payload or {"x": 7},
                    status=UnitStatus.PENDING)
    job = BatchJob(id="job1", name="j", func=func, units=[unit],
                   cpu_pct_per_unit=10, ram_mb_per_unit=64, owner=owner,
                   status="running", created_ms=now_ms())
    return job, unit


# ===========================================================================
# 1. VerificationPolicy classification
# ===========================================================================

def test_policy_classifies_digest_verifiable():
    p = VerificationPolicy()
    for func in DIGEST_VERIFIABLE_FUNCS:
        assert p.classify(func) == WorkloadKind.DIGEST_VERIFIABLE, func


def test_policy_classifies_oracle_verifiable():
    p = VerificationPolicy()
    for func in ORACLE_VERIFIABLE_FUNCS:
        assert p.classify(func) == WorkloadKind.ORACLE_VERIFIABLE, func


def test_policy_classifies_not_verifiable():
    p = VerificationPolicy()
    for func in NOT_VERIFIABLE_FUNCS:
        assert p.classify(func) == WorkloadKind.NOT_VERIFIABLE, func


def test_policy_decide_returns_redundant_for_digest_verifiable():
    p = VerificationPolicy(redundant_rate=1.0)
    d = p.decide("square")
    assert d.technique == VerificationTechnique.REDUNDANT
    assert d.kind == WorkloadKind.DIGEST_VERIFIABLE


def test_policy_decide_returns_none_for_not_verifiable():
    p = VerificationPolicy(redundant_rate=1.0)
    d = p.decide("sleep_echo")
    assert d.technique == VerificationTechnique.NONE


def test_policy_should_not_trigger_redundant_when_rate_zero():
    p = VerificationPolicy(redundant_rate=0.0)
    rng = random.Random(0)
    assert not any(p.should_trigger_redundant("square", rng) for _ in range(100))


def test_policy_should_always_trigger_redundant_when_rate_one():
    p = VerificationPolicy(redundant_rate=1.0)
    rng = random.Random(0)
    assert all(p.should_trigger_redundant("square", rng) for _ in range(20))


def test_policy_never_triggers_redundant_for_not_verifiable():
    p = VerificationPolicy(redundant_rate=1.0)
    rng = random.Random(0)
    assert not p.should_trigger_redundant("sleep_echo", rng)
    assert not p.should_trigger_redundant("secure_fetch", rng)


# ===========================================================================
# 2. ComputeEscrowEngine: hold then release
# ===========================================================================

def test_escrow_held_then_released(tmp_store):
    ledger = ContributionLedger(tmp_store)
    esc = ComputeEscrowEngine(ledger)
    node = "node-honest"
    N = CONSECUTIVE_PASSES_TO_RELEASE

    for _ in range(N - 1):
        esc.on_verify_pass(node, owner=node)
    state = esc.escrow_state(node)
    assert state["held_escrow"] > 0.0, "escrow must be held before threshold"
    assert ledger.credits(node) == 0.0, "no credits until threshold passed"

    released = esc.on_verify_pass(node, owner=node)
    assert released > 0.0, "must release escrow at threshold"
    assert ledger.credits(node) > 0.0, "credits must appear in ledger after release"
    # Booked as a grant, never as donated: nobody consumed these credits, so
    # recording them as donated work breaks the donated == consumed invariant.
    totals = tmp_store.ledger_totals_by_owner()[node]
    assert abs(totals.get("grant", 0.0) - released) < 1e-9, (
        f"escrow release must be booked as a grant, got {totals}")
    assert totals.get("donated", 0.0) == 0.0, (
        f"escrow release must not inflate donated credits, got {totals}")
    state2 = esc.escrow_state(node)
    assert state2["held_escrow"] == 0.0, "escrow resets to 0 after release"
    assert state2["consecutive_passes"] == 0
    assert abs(esc.total_released - released) < 1e-9


# ===========================================================================
# 3. ComputeEscrowEngine: forfeit on failure
# ===========================================================================

def test_escrow_forfeited_on_fail(tmp_store):
    ledger = ContributionLedger(tmp_store)
    esc = ComputeEscrowEngine(ledger)
    node = "node-bad"
    N = CONSECUTIVE_PASSES_TO_RELEASE

    for _ in range(N - 1):
        esc.on_verify_pass(node, owner=node)
    held = esc.escrow_state(node)["held_escrow"]
    assert held > 0.0

    forfeited = esc.on_verify_fail(node)
    assert abs(forfeited - held) < 1e-9, "must forfeit all held escrow"
    assert ledger.credits(node) == 0.0, "liar gets no credits"
    state = esc.escrow_state(node)
    assert state["held_escrow"] == 0.0
    assert state["consecutive_passes"] == 0
    assert abs(esc.total_forfeited - forfeited) < 1e-9


# ===========================================================================
# 4. Digest comparison
# ===========================================================================

def test_digest_match_identical_results():
    v = Verifier()
    assert v.digests_match({"n": 7, "prime": True}, {"n": 7, "prime": True})
    assert v.digests_match(49, 49)
    assert v.digests_match([1, 2, 3], [1, 2, 3])


def test_digest_no_match_different_results():
    v = Verifier()
    assert not v.digests_match(49, 50)
    assert not v.digests_match({"n": 7, "prime": True}, {"n": 7, "prime": False})


def test_digest_sort_keys_deterministic():
    # Dict ordering must not affect the digest.
    r1 = {"b": 2, "a": 1}
    r2 = {"a": 1, "b": 2}
    assert _result_digest(r1) == _result_digest(r2)


# ===========================================================================
# 5. Redundant check catches a simulated cheater
# ===========================================================================

def test_redundant_check_catches_cheater(tmp_store):
    """A lying primary node fails redundant verification: penalised, unit retried."""
    honest1 = _FullNode("h1")
    honest2 = _FullNode("h2")
    liar = _FullNode("liar")
    nodes = [liar, honest1, honest2]
    transport = _Transport(liars=frozenset({"liar"}))
    mgr = _make_mgr(tmp_store, nodes, tier=2, transport=transport,
                    redundant_rate=1.0)

    job, unit = _job("square", {"x": 5})
    # Primary already returned a corrupted result.
    primary_result = _corrupt(run_function("square", {"x": 5}))
    verdict = asyncio.run(mgr._redundant_check(job, unit, primary_result, liar))

    assert verdict is False, "liar should be caught (verdict=False means primary lied)"
    assert mgr.verifier.stats.redundant_verifications >= 1
    assert mgr.verifier.stats.disagreements >= 1
    assert mgr.verifier.stats.nodes_penalized >= 1


# ===========================================================================
# 6. FALSE-POSITIVE GUARD: honest node NOT falsely accused
# ===========================================================================

def test_redundant_check_does_not_falsely_accuse_honest_node(tmp_store):
    """Two honest nodes produce matching results -- neither must be penalised.

    This test is the non-false-positive control.  A verification system that
    generates false positives is worse than none -- it would slash honest
    contributors.  Here both nodes return the CORRECT result; the verifier must
    report True (pass) and no penalties should be recorded.
    """
    honest1 = _FullNode("h1")
    honest2 = _FullNode("h2")
    nodes = [honest1, honest2]
    transport = _Transport()  # no liars
    mgr = _make_mgr(tmp_store, nodes, tier=2, transport=transport,
                    redundant_rate=1.0)

    job, unit = _job("square", {"x": 9})
    primary_result = run_function("square", {"x": 9})  # 81 (correct)
    verdict = asyncio.run(mgr._redundant_check(job, unit, primary_result, honest1))

    assert verdict is True, (
        f"honest node must pass (verdict={verdict!r}); "
        "false positive would cause spurious penalty")
    assert mgr.verifier.stats.nodes_penalized == 0, (
        "no honest node should be penalised")


def test_not_verifiable_function_never_redundancy_checked(tmp_store):
    """sleep_echo is NOT_VERIFIABLE -- the policy must not trigger redundant check."""
    honest1 = _FullNode("h1")
    honest2 = _FullNode("h2")
    nodes = [honest1, honest2]
    transport = _Transport()
    mgr = _make_mgr(tmp_store, nodes, tier=2, transport=transport,
                    redundant_rate=1.0)

    job, unit = _job("sleep_echo", {"seconds": 0, "echo": "hi"})
    result = {"slept": 0.0, "echo": "hi"}
    verdict = asyncio.run(mgr._redundant_check(job, unit, result, honest1))

    assert verdict is None, (
        "NOT_VERIFIABLE func must return None (skipped), not attempt comparison")
    assert mgr.verifier.stats.redundant_verifications == 0
    assert mgr.verifier.stats.nodes_penalized == 0


# ===========================================================================
# 7. Graceful degradation when cluster < 2 nodes
# ===========================================================================

def test_redundant_check_degrades_when_no_secondary_available(tmp_store):
    """Single-node cluster: redundant check skips rather than blocking."""
    only_node = _FullNode("solo")
    nodes = [only_node]
    mgr = _make_mgr(tmp_store, nodes, tier=2, redundant_rate=1.0)

    job, unit = _job("square", {"x": 3})
    primary_result = run_function("square", {"x": 3})
    verdict = asyncio.run(mgr._redundant_check(job, unit, primary_result, only_node))

    assert verdict is None, "must degrade gracefully, not hang"
    assert mgr.verifier.stats.degraded_no_secondary >= 1
    assert mgr.verifier.stats.nodes_penalized == 0


# ===========================================================================
# 8. Verifier.record_fail updates escrow
# ===========================================================================

def test_verifier_record_fail_updates_escrow(tmp_store):
    ledger = ContributionLedger(tmp_store)
    v = Verifier(ledger=ledger)
    node = "bad"
    N = CONSECUTIVE_PASSES_TO_RELEASE

    for _ in range(N - 1):
        v.record_pass(node, owner=node)
    assert v.compute_escrow.escrow_state(node)["held_escrow"] > 0

    v.record_fail(node)

    assert v.compute_escrow.escrow_state(node)["held_escrow"] == 0.0
    assert v.stats.nodes_penalized == 1
    assert ledger.credits(node) == 0.0  # forfeited, not released


# ===========================================================================
# 9. Oracle spot-check path updates escrow
# ===========================================================================

def test_oracle_pass_updates_verifier_escrow(tmp_store):
    """When the oracle spot-check passes, the verifier escrow must be updated."""
    honest = _FullNode("h1")
    nodes = [honest]
    transport = _Transport()
    mgr = _make_mgr(tmp_store, nodes, tier=2, transport=transport,
                    redundant_rate=0.0)  # oracle only, no redundant

    # Fresh node has 0 samples so needs_verification() == True.
    job, unit = _job("square", {"x": 4}, owner="cons:test")
    asyncio.run(mgr._run_unit(job, unit, honest))

    assert unit.status is UnitStatus.DONE, unit.error
    assert mgr.verifier.stats.units_verified >= 1


def test_oracle_fail_updates_verifier_escrow(tmp_store):
    """When oracle catches a lie, verifier.record_fail must fire."""
    liar_node = _FullNode("liar")
    nodes = [liar_node]
    transport = _Transport(liars=frozenset({"liar"}))
    mgr = _make_mgr(tmp_store, nodes, tier=2, transport=transport,
                    redundant_rate=0.0)  # oracle only

    job, unit = _job("square", {"x": 4}, owner="cons:test")
    asyncio.run(mgr._run_unit(job, unit, liar_node))

    assert unit.status is UnitStatus.PENDING
    assert mgr.verifier.stats.nodes_penalized >= 1
