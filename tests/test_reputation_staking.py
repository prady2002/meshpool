from __future__ import annotations

from meshpool.common.trust import ContributionLedger, TrustEngine
from meshpool.controlplane.verification import (
    HeldBackEscrowEngine,
    InteractiveBisectionDispute,
    StakingEngine,
)


def test_beta_reputation_is_value_weighted_and_decays(tmp_store):
    trust = TrustEngine(tmp_store, tier=2)
    now = 1_000_000
    trust.record_verification("node", True, work_value=0.01, at_ms=now)
    cheap_score = trust.score("node")
    cheap_stake = trust.required_stake("node", expected_job_value=100.0)

    trust.record_verification("node", True, work_value=2_000.0, at_ms=now + 1)
    proven_stake = trust.required_stake("node", expected_job_value=100.0)

    assert cheap_score > 0.5
    assert proven_stake < cheap_stake, "valuable verified work must reduce required stake"

    old_score = trust.score("node")
    trust.record_verification("node", False, work_value=100.0,
                              at_ms=now + 365 * 24 * 60 * 60 * 1000)
    assert trust.score("node") < old_score, "recent fraud must outweigh ancient success"


def test_slashing_moves_stake_without_breaking_conservation(tmp_store):
    ledger = ContributionLedger(tmp_store)
    trust = TrustEngine(tmp_store, tier=2)
    staking = StakingEngine(ledger, trust=trust)

    required = staking.ensure_stake("bad-node", "dev:bad", expected_job_value=3.0)
    slashed = staking.slash("bad-node", expected_job_value=3.0)

    assert slashed == required
    totals = tmp_store.ledger_totals_by_owner()
    assert totals["dev:bad"]["consumed"] == slashed
    assert totals["meshpool:slashing"]["donated"] == slashed
    assert abs(
        sum(v["donated"] for v in totals.values())
        - sum(v["consumed"] for v in totals.values())
    ) < 1e-9


def test_heldback_escrow_releases_on_fast_forwarded_clock(tmp_store):
    ledger = ContributionLedger(tmp_store)
    t = {"now": 0}
    escrow = HeldBackEscrowEngine(ledger, clock_ms=lambda: t["now"])

    immediate, held = escrow.record_earning("node", "dev:node", 100.0, detail="job")
    ledger.record_consumption("node", "cons:alice", 100.0, detail="job")
    assert immediate == 50.0
    assert held == 50.0
    assert ledger.credits("dev:node") == 50.0
    assert escrow.release_due("node") == 0.0

    t["now"] = 270 * 24 * 60 * 60 * 1000 + 1
    released = escrow.release_due("node")
    assert released == 50.0
    assert ledger.credits("dev:node") == 100.0
    assert ledger.credits("meshpool:heldback-escrow") == 0.0
    totals = tmp_store.ledger_totals_by_owner()
    assert abs(
        sum(v["donated"] for v in totals.values())
        - sum(v["consumed"] for v in totals.values())
    ) < 1e-9


def test_truebit_bisection_localizes_first_bad_step_logarithmically():
    dispute = InteractiveBisectionDispute()
    honest = [i * i for i in range(64)]
    dishonest = list(honest)
    dishonest[37:] = [x + 1 for x in dishonest[37:]]

    verdict = dispute.localize(dishonest, honest)

    assert verdict.localized is True
    assert verdict.first_disagreement == 37
    assert verdict.rounds <= 8
