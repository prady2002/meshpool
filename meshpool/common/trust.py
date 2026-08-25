"""Trust scoring + fair-share accounting — "Proof of Contribution" (spec §7).

Architecturally complete for all three tiers, but only Tier 0 behaviour is active
in the MVP:

* Tier 0 (home): every node fully trusted; scoring is pass-through, no redundant
  execution, fair-share is a no-op (single owner).
* Tier 1 (office): EMA reputation + known-answer benchmarks + redundant execution
  below a threshold + fair-share scheduling by contribution credits.
* Tier 2 (public): out of MVP scope.

Keeping the ledger schema and these interfaces stable from day one means enabling
Tier 1 is a config flip, not a rewrite (spec §13.5).
"""
from __future__ import annotations

from typing import Optional

from .db import Store
from .models import LedgerEntry, TrustScore, new_id, now_ms

# Nodes scoring below this get redundant (double-checked) execution at Tier 1.
REDUNDANT_EXECUTION_THRESHOLD = 0.6

# At Tier 2 (public/adversarial) a freshly-seen device is on probation until it has
# accumulated at least this many reputation samples — its results are verified
# against a local oracle so a brand-new identity can't be trusted just because its
# EMA fields default to 1.0. Once it clears warmup with good behaviour it graduates
# to unverified (distributed) execution, which is the whole point of the mesh.
WARMUP_SAMPLES = 8

# Even after a node graduates, at Tier 2 the coordinator keeps auditing a small
# random fraction of its results against the oracle. This catches a "traitor" that
# behaves through warmup and only then starts cheating: the first audited lie trips
# verify_failures, after which it is verified on every unit forever.
VERIFY_SAMPLE_RATE = 0.15


def update_ema(prev: float, sample: float, alpha: float = 0.2) -> float:
    """Exponential moving average — recent behaviour weighted over history."""
    return (1.0 - alpha) * prev + alpha * sample


class TrustEngine:
    def __init__(self, store: Store, tier: int = 0) -> None:
        self.store = store
        self.tier = tier

    def _score(self, node_id: str) -> TrustScore:
        return self.store.get_trust(node_id) or TrustScore(node_id=node_id)

    # --- signal recording (no-ops worth persisting even at Tier 0 for the UI) #
    def record_task_result(self, node_id: str, success: bool) -> None:
        s = self._score(node_id)
        s.task_success_rate_ema = update_ema(s.task_success_rate_ema, 1.0 if success else 0.0)
        s.samples += 1
        self.store.put_trust(s)

    def record_benchmark(self, node_id: str, passed: bool) -> None:
        s = self._score(node_id)
        s.benchmark_pass_rate_ema = update_ema(s.benchmark_pass_rate_ema, 1.0 if passed else 0.0)
        s.samples += 1
        self.store.put_trust(s)

    def record_verification(self, node_id: str, passed: bool) -> None:
        """Record the outcome of a Tier-2 oracle spot-check.

        A pass advances the node toward graduating out of probation; a failure both
        drags its benchmark reputation down (like any failed benchmark) *and* trips
        ``verify_failures``, which pins the node to per-unit verification for good —
        a node proven to have lied once is never again trusted blindly.
        """
        s = self._score(node_id)
        if passed:
            s.verify_passes += 1
        else:
            s.verify_failures += 1
        s.benchmark_pass_rate_ema = update_ema(s.benchmark_pass_rate_ema, 1.0 if passed else 0.0)
        s.samples += 1
        self.store.put_trust(s)

    def record_uptime(self, node_id: str, up: bool) -> None:
        s = self._score(node_id)
        s.uptime_ratio_ema = update_ema(s.uptime_ratio_ema, 1.0 if up else 0.0, alpha=0.05)
        self.store.put_trust(s)

    def record_latency_sample(self, node_id: str, consistent: bool) -> None:
        s = self._score(node_id)
        s.latency_consistency_ema = update_ema(
            s.latency_consistency_ema, 1.0 if consistent else 0.0, alpha=0.1)
        self.store.put_trust(s)

    def score(self, node_id: str) -> float:
        if self.tier == 0:
            return 1.0  # full mutual trust — no verification overhead
        return self._score(node_id).score()

    def samples(self, node_id: str) -> int:
        return self._score(node_id).samples

    def needs_redundant_execution(self, node_id: str) -> bool:
        if self.tier == 0:
            return False
        return self.score(node_id) < REDUNDANT_EXECUTION_THRESHOLD

    def needs_verification(self, node_id: str) -> bool:
        """Should this node's results be spot-checked against a trusted oracle?

        Tier 0: never (single owner). Tier 1: only clearly-degraded nodes. Tier 2:
        also every not-yet-proven stranger during its warmup window — that is what
        makes it safe to accept work results from someone you've never met.
        """
        if self.tier == 0:
            return False
        s = self._score(node_id)
        if self.tier >= 2:
            # A never-proven stranger is verified through its whole warmup window.
            if s.samples < WARMUP_SAMPLES:
                return True
            # A node ever caught returning a wrong answer is verified on every unit
            # from then on — graduating a proven cheat back to blind trust would let
            # it resume cheating the moment it slips below detection.
            if s.verify_failures > 0:
                return True
        return s.score() < REDUNDANT_EXECUTION_THRESHOLD


class ContributionLedger:
    """Tracks compute-time donated vs. consumed per node/owner for fair-share."""

    def __init__(self, store: Store) -> None:
        self.store = store

    def record_donation(self, node_id: str, owner: str, compute_units: float,
                        detail: str = "") -> None:
        self.store.add_ledger(LedgerEntry(
            id=new_id("led"), node_id=node_id, owner=owner, kind="donated",
            compute_units=max(0.0, compute_units), ts_ms=now_ms(), detail=detail))

    def record_consumption(self, node_id: str, owner: str, compute_units: float,
                           detail: str = "") -> None:
        self.store.add_ledger(LedgerEntry(
            id=new_id("led"), node_id=node_id, owner=owner, kind="consumed",
            compute_units=max(0.0, compute_units), ts_ms=now_ms(), detail=detail))

    def record_grant(self, owner: str, compute_units: float, detail: str = "") -> None:
        """One-time (or promotional) credit grant — e.g. the starter balance a newly
        verified device/account receives so it can consume before it has donated."""
        self.store.add_ledger(LedgerEntry(
            id=new_id("led"), node_id="system", owner=owner, kind="grant",
            compute_units=max(0.0, compute_units), ts_ms=now_ms(), detail=detail))

    def credits(self, owner: str) -> float:
        return self.store.credits_by_owner().get(owner, 0.0)


def fair_share_priority(credits: float, tier: int = 0) -> float:
    """scheduling_priority(user_or_team) = f(contribution_credits).

    At Tier 0 (single owner) contention doesn't exist, so priority is constant.
    At Tier 1+ higher net contribution yields higher priority when the cluster is
    contested. A gentle log curve avoids letting one heavy contributor starve
    everyone else."""
    if tier == 0:
        return 1.0
    import math
    return 1.0 + math.log1p(max(0.0, credits)) - math.log1p(max(0.0, -credits))
