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

# Beta reputation decays with a 90-day half-life: old good behaviour fades, and
# old bad behaviour can be overcome by sustained correct work.  Samples are
# weighted by verified job value so reputation is costly to farm.
BETA_DECAY_HALF_LIFE_MS = 90 * 24 * 60 * 60 * 1000
MAX_REPUTATION_WEIGHT_PER_SAMPLE = 100.0
BASE_STAKE_MULTIPLIER = 20.0
MIN_STAKE_MULTIPLIER = 2.0
SLASH_DESTINATION_OWNER = "meshpool:slashing"


def update_ema(prev: float, sample: float, alpha: float = 0.2) -> float:
    """Exponential moving average — recent behaviour weighted over history."""
    return (1.0 - alpha) * prev + alpha * sample


class TrustEngine:
    def __init__(self, store: Store, tier: int = 0) -> None:
        self.store = store
        self.tier = tier

    def _score(self, node_id: str) -> TrustScore:
        return self.store.get_trust(node_id) or TrustScore(node_id=node_id)

    def snapshot(self, node_id: str) -> TrustScore:
        """Public read of a node's current reputation signals.

        Returns a default-valued score for a node that has never been observed.
        Callers must therefore read ``samples`` before trusting any EMA on it:
        the defaults are all optimistic, so an unseen node looks perfect. See
        ``controlplane/nodeclass.py``, which is why this accessor exists.
        """
        return self._score(node_id)

    def _decay_beta(self, s: TrustScore, at_ms: Optional[int] = None) -> TrustScore:
        at = now_ms() if at_ms is None else int(at_ms)
        if s.beta_last_ms <= 0:
            s.beta_last_ms = at
            return s
        elapsed = max(0, at - s.beta_last_ms)
        if elapsed <= 0:
            return s
        factor = 0.5 ** (elapsed / BETA_DECAY_HALF_LIFE_MS)
        # Decay evidence toward the neutral Beta(1,1) prior, not toward zero.
        s.beta_alpha = 1.0 + (s.beta_alpha - 1.0) * factor
        s.beta_beta = 1.0 + (s.beta_beta - 1.0) * factor
        s.proven_value *= factor
        s.beta_last_ms = at
        return s

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

    def record_verification(self, node_id: str, passed: bool,
                            work_value: float = 1.0,
                            at_ms: Optional[int] = None) -> None:
        """Record the outcome of a Tier-2 oracle spot-check.

        A pass advances the node toward graduating out of probation; a failure both
        drags its benchmark reputation down (like any failed benchmark) *and* trips
        ``verify_failures``, which pins the node to per-unit verification for good —
        a node proven to have lied once is never again trusted blindly.
        """
        s = self._decay_beta(self._score(node_id), at_ms=at_ms)
        if passed:
            s.verify_passes += 1
            s.beta_alpha += min(MAX_REPUTATION_WEIGHT_PER_SAMPLE, max(0.0, work_value))
            s.proven_value += max(0.0, work_value)
        else:
            s.verify_failures += 1
            s.beta_beta += min(MAX_REPUTATION_WEIGHT_PER_SAMPLE,
                               max(1.0, work_value * 4.0))
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
        s = self._decay_beta(self._score(node_id))
        self.store.put_trust(s)
        return s.score()

    def proven_value(self, node_id: str) -> float:
        s = self._decay_beta(self._score(node_id))
        self.store.put_trust(s)
        return s.proven_value

    def required_stake(self, node_id: str, expected_job_value: float) -> float:
        """Graduated stake: new nodes stake ~20x value; proven nodes stake less.

        Reputation is value-scoped: confidence is capped by how much verified value
        the node has already delivered relative to this job, preventing farms of
        worthless micro-jobs from buying trust for expensive work.
        """
        value = max(0.0, expected_job_value)
        if value <= 0:
            return 0.0
        rep = self.score(node_id) if self.tier > 0 else 1.0
        confidence = min(1.0, self.proven_value(node_id) / max(value * 20.0, 1.0))
        effective_rep = rep * confidence
        multiplier = MIN_STAKE_MULTIPLIER + (
            BASE_STAKE_MULTIPLIER - MIN_STAKE_MULTIPLIER) * (1.0 - effective_rep)
        return value * multiplier

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

    def record_grant(self, owner: str, compute_units: float, detail: str = "",
                     node_id: str = "system") -> None:
        """One-time (or promotional) credit grant — e.g. the starter balance a newly
        verified device/account receives so it can consume before it has donated.

        This is the ledger's **only** issuance channel. ``donated`` and ``consumed``
        are two halves of one transfer and must stay equal in aggregate; anything
        that creates credits without a matching consumer belongs here instead, or
        the conservation invariant silently stops meaning anything. ``node_id``
        attributes a grant that was earned by a specific node (e.g. an escrow
        release) rather than issued to the account as a whole.
        """
        self.store.add_ledger(LedgerEntry(
            id=new_id("led"), node_id=node_id, owner=owner, kind="grant",
            compute_units=max(0.0, compute_units), ts_ms=now_ms(), detail=detail))

    def record_transfer(self, from_owner: str, to_owner: str, compute_units: float,
                        detail: str = "", node_id: str = "system") -> None:
        """Move existing credits between real owners without minting or burning.

        The ledger has no separate debit kind: a transfer is represented as a
        matched consumed/donated pair.  This preserves the global donated ==
        consumed invariant while making slashing and escrow release visible in the
        same accounting surface as ordinary work.
        """
        amount = max(0.0, compute_units)
        if amount <= 0.0:
            return
        transfer_id = new_id("xfer")
        self.store.add_ledger_pair(
            LedgerEntry(id=f"{transfer_id}:debit", node_id=node_id,
                        owner=from_owner, kind="consumed",
                        compute_units=amount, ts_ms=now_ms(), detail=detail),
            LedgerEntry(id=f"{transfer_id}:credit", node_id=node_id,
                        owner=to_owner, kind="donated",
                        compute_units=amount, ts_ms=now_ms(), detail=detail),
        )

    def record_slash(self, offender_owner: str, amount: float,
                     destination_owner: str = SLASH_DESTINATION_OWNER,
                     node_id: str = "system", detail: str = "fraud slash") -> None:
        self.record_transfer(offender_owner, destination_owner, amount,
                             detail=detail, node_id=node_id)

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
