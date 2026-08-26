"""Workload-matched result verification (spec P6 trust layer).

WHY THIS EXISTS
---------------
MeshPool runs YOUR work on STRANGERS computers.  A malicious donor can return
garbage instantly and collect credits, and today nothing detects it.  Without
verification the entire economic model collapses: lying is strictly more
profitable than computing.

The key insight: verification must be MATCHED to the workload.  There is no
single scheme that works for everything.  This module provides the strategy layer.

WORKLOAD -> TECHNIQUE MAP
-------------------------
  Deterministic (int/hash)   -> Redundant execution    (N* compute cost)
  Deterministic, float       -> Spot-check / oracle    (p* compute cost)
  Non-deterministic          -> Not verifiable         (0 cost)
  Storage                    -> Chunk challenge (P3)   (tiny cost)

DETERMINISM CLASSIFICATION
--------------------------
The critical decision: which functions produce BIT-IDENTICAL results across
heterogeneous CPUs?  A false accusation (honest node penalised for a difference
that IS NOT cheating) is worse than no verification at all.

Safe for digest comparison (DIGEST_VERIFIABLE):
  - square:    integer multiplication only; exact across all IEEE-754 impls.
  - is_prime:  trial division, pure integer arithmetic.
  - sha_pow:   hashlib.sha256 is a NIST standard; hex output always identical.
  - wasm:*:    WASM is deterministic by specification (integer machine, no FP).

Oracle-verifiable only (ORACLE_VERIFIABLE) -- correct but not bit-identical:
  - embed: uses round(v/norm, 5) on float arithmetic.  Pure-Python sequential
    loops are deterministic in IEEE-754, but cross-CPU SIMD/FMA or future NumPy
    vectorisation could differ by 1 ULP.  Conservative treatment avoids false
    accusations on heterogeneous hardware.

Not verifiable (NOT_VERIFIABLE):
  - sleep_echo:   timing-dependent side-channels; primarily a test function.
  - secure_fetch: result depends on a brokered API_TOKEN whose value the
    coordinator does NOT hold in cleartext (defeating the brokered-secret point).

ESCROW MODEL
------------
Bonus credits for a compute unit are held in escrow until N consecutive
verifications pass.  Forfeited on failure.  Mirrors StorageAuditEngine (P3)
exactly: same escrow hold/release/forfeit logic, same ContributionLedger.

The escrow is a BONUS on top of regular credit recording.  Regular credits
(record_donation / record_consumption) continue to flow immediately so the base
economic model is unchanged.  Escrow provides an additional incentive for good
behaviour and an additional penalty for cheating above and beyond the existing
spot-check path.

SCALING COST
------------
Redundant execution costs N* compute.  The default rate (MESHPOOL_VERIFY_REDUNDANT_RATE)
is 0.25: 25% of digest-verifiable units are re-run on a second node.  At 25% the
fleet throughput penalty is at most 8%.  Set to 0 to disable entirely (oracle-only
mode) or 1.0 for maximum security on a small test cluster.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

from ..common.logging import get_logger
from ..common.models import now_ms
from ..common.trust import (
    BASE_STAKE_MULTIPLIER, SLASH_DESTINATION_OWNER, ContributionLedger,
)

log = get_logger("verification")

# ---------------------------------------------------------------------------
# Determinism classification constants
# ---------------------------------------------------------------------------

# Pure integer / hash output -- bit-identical across all IEEE-754 implementations.
DIGEST_VERIFIABLE_FUNCS: frozenset[str] = frozenset({"square", "is_prime", "sha_pow"})

# WASM modules: deterministic by WASM specification (integer sequential machine).
WASM_DIGEST_VERIFIABLE: bool = True

# Oracle-verifiable: correct but may not be bit-identical across heterogeneous CPUs.
ORACLE_VERIFIABLE_FUNCS: frozenset[str] = frozenset({"embed"})

# Not verifiable: depends on secrets, timing, or external state we cannot replicate.
NOT_VERIFIABLE_FUNCS: frozenset[str] = frozenset({"sleep_echo", "secure_fetch"})

# Escrow parameters -- mirror StorageAuditEngine for consistency.
ESCROW_PER_VERIFIED_UNIT: float = 0.5
CONSECUTIVE_PASSES_TO_RELEASE: int = 3

# Default fraction of DIGEST_VERIFIABLE units to put through redundant execution.
# Override via MESHPOOL_VERIFY_REDUNDANT_RATE env var (e.g. "1.0" for 100%).
DEFAULT_REDUNDANT_RATE: float = float(
    os.environ.get("MESHPOOL_VERIFY_REDUNDANT_RATE", "0.25"))

ESCROW_OWNER = "meshpool:heldback-escrow"
THREE_MONTHS_MS = 90 * 24 * 60 * 60 * 1000
NINE_MONTHS_MS = 270 * 24 * 60 * 60 * 1000
MIN_HOLDBACK_EARNING = 1.0


# ---------------------------------------------------------------------------
# Enums and decision type
# ---------------------------------------------------------------------------

class WorkloadKind(str, Enum):
    DIGEST_VERIFIABLE = "digest_verifiable"
    ORACLE_VERIFIABLE = "oracle_verifiable"
    NOT_VERIFIABLE = "not_verifiable"


class VerificationTechnique(str, Enum):
    REDUNDANT = "redundant"      # run on N independent nodes, compare digests
    SPOT_CHECK = "spot_check"    # re-execute on trusted local oracle
    CANARY = "canary"            # inject known-answer trap units (inference)
    NONE = "none"                # cannot verify this workload type


class VerificationDecision:
    """Policy decision for a single batch function."""
    __slots__ = ("kind", "technique", "reason")

    def __init__(self, kind: WorkloadKind, technique: VerificationTechnique,
                 reason: str = "") -> None:
        self.kind = kind
        self.technique = technique
        self.reason = reason

    def __repr__(self) -> str:
        return (f"VerificationDecision(kind={self.kind.value}, "
                f"technique={self.technique.value})")


# ---------------------------------------------------------------------------
# VerificationPolicy
# ---------------------------------------------------------------------------

class VerificationPolicy:
    """Classifies a batch function and selects an appropriate verification technique.

    Design for extension: register new workload types by overriding ``classify``
    and extending the ``decide`` logic.  The policy is injected into BatchManager
    via ``verifier.policy``; changing it requires no changes to the engine.
    """

    def __init__(
        self,
        redundant_rate: float = DEFAULT_REDUNDANT_RATE,
        redundant_n: int = 2,
        redundant_n_tiebreak: int = 3,
    ) -> None:
        # Fraction of DIGEST_VERIFIABLE units to verify by redundant execution.
        # 0.0 disables redundant execution (oracle-only mode).
        self.redundant_rate = redundant_rate
        # How many nodes to use in a redundant check.
        self.redundant_n = redundant_n
        # How many nodes in a tiebreak round.
        self.redundant_n_tiebreak = redundant_n_tiebreak

    def classify(self, func: str) -> WorkloadKind:
        """Classify a function by its determinism properties."""
        try:
            from ..batch.functions import is_wasm_function
            if is_wasm_function(func) and WASM_DIGEST_VERIFIABLE:
                return WorkloadKind.DIGEST_VERIFIABLE
        except Exception:  # noqa: BLE001
            pass
        if func in DIGEST_VERIFIABLE_FUNCS:
            return WorkloadKind.DIGEST_VERIFIABLE
        if func in ORACLE_VERIFIABLE_FUNCS:
            return WorkloadKind.ORACLE_VERIFIABLE
        if func in NOT_VERIFIABLE_FUNCS:
            return WorkloadKind.NOT_VERIFIABLE
        # Unknown function (user-supplied): conservatively not verifiable.
        return WorkloadKind.NOT_VERIFIABLE

    def decide(self, func: str) -> VerificationDecision:
        """Choose a verification technique for a given function."""
        kind = self.classify(func)
        if kind == WorkloadKind.DIGEST_VERIFIABLE and self.redundant_rate > 0:
            return VerificationDecision(
                kind, VerificationTechnique.REDUNDANT,
                "digest-safe function; redundant execution enabled")
        if kind in (WorkloadKind.DIGEST_VERIFIABLE, WorkloadKind.ORACLE_VERIFIABLE):
            return VerificationDecision(
                kind, VerificationTechnique.SPOT_CHECK,
                "oracle-verifiable function")
        return VerificationDecision(
            kind, VerificationTechnique.NONE,
            "function is not verifiable without false-positive risk")

    def should_trigger_redundant(self, func: str, rng: random.Random) -> bool:
        """Whether a specific unit should undergo redundant execution.

        Only applies to DIGEST_VERIFIABLE functions and only when redundant_rate > 0.
        Uses the supplied RNG so tests are fully deterministic.
        """
        if self.redundant_rate <= 0:
            return False
        if self.classify(func) != WorkloadKind.DIGEST_VERIFIABLE:
            return False
        return rng.random() < self.redundant_rate


# ---------------------------------------------------------------------------
# ComputeEscrowEngine
# ---------------------------------------------------------------------------

class ComputeEscrowEngine:
    """Holds bonus compute credits in escrow until N consecutive verifications pass.

    Mirrors StorageAuditEngine's escrow model (spec P3):
      - on_verify_pass: credit accrues; released as real donation after N passes.
      - on_verify_fail: all held escrow forfeited (no credits for the cheater).

    The escrow is a BONUS on top of regular credit recording.  Regular
    record_donation / record_consumption calls continue immediately, keeping the
    base economy unchanged.  The escrow adds extra incentive for good behaviour.
    """

    def __init__(
        self,
        ledger: ContributionLedger,
        consecutive_passes_to_release: int = CONSECUTIVE_PASSES_TO_RELEASE,
        escrow_per_unit: float = ESCROW_PER_VERIFIED_UNIT,
    ) -> None:
        self.ledger = ledger
        self._consecutive_passes_to_release = consecutive_passes_to_release
        self._escrow_per_unit = escrow_per_unit
        # per-node state: node_id -> {consecutive_passes, held_escrow}
        self._node_state: dict[str, dict] = {}
        self.total_released: float = 0.0
        self.total_forfeited: float = 0.0

    def on_verify_pass(self, node_id: str, owner: str) -> float:
        """Record a passing verification.  Returns escrow released (may be 0)."""
        state = self._node_state.setdefault(
            node_id, {"consecutive_passes": 0, "held_escrow": 0.0})
        state["consecutive_passes"] += 1
        state["held_escrow"] += self._escrow_per_unit
        released = 0.0
        if state["consecutive_passes"] >= self._consecutive_passes_to_release:
            released = state["held_escrow"]
            # A grant, not a donation -- see the identical note in storage/audit.py.
            # Nobody consumed these credits, so booking them as donated work breaks
            # the donated == consumed conservation invariant (this is exactly what
            # verify_chaos.py caught). Spendable balance is unaffected.
            self.ledger.record_grant(
                owner, released,
                detail=(
                    f"compute escrow released: "
                    f"{state['consecutive_passes']} consecutive verified units"),
                node_id=node_id)
            log.info("compute escrow released", node=node_id, amount=released,
                     passes=state["consecutive_passes"])
            state["consecutive_passes"] = 0
            state["held_escrow"] = 0.0
            self.total_released += released
        return released

    def on_verify_fail(self, node_id: str) -> float:
        """Record a failing verification.  Forfeits all held escrow."""
        state = self._node_state.setdefault(
            node_id, {"consecutive_passes": 0, "held_escrow": 0.0})
        forfeited = state["held_escrow"]
        state["consecutive_passes"] = 0
        state["held_escrow"] = 0.0
        self.total_forfeited += forfeited
        if forfeited > 0:
            log.warning("compute escrow FORFEITED", node=node_id, amount=forfeited)
        return forfeited

    def escrow_state(self, node_id: str) -> dict:
        """Return the current escrow state for a node (copy)."""
        return dict(self._node_state.get(
            node_id, {"consecutive_passes": 0, "held_escrow": 0.0}))

    def total_held(self) -> float:
        """Sum of all held escrow across all nodes."""
        return sum(s["held_escrow"] for s in self._node_state.values())


# ---------------------------------------------------------------------------
# VerificationStats
# ---------------------------------------------------------------------------

class VerificationStats:
    """Observable counters for the verification system.

    These are the only externally visible proof that the fraud-detection path
    actually ran.  Without counters a gate asserting the liar was caught cannot
    tell apart caught-nobody-because-nobody-lied from caught-nobody-because-check-never-ran.
    """

    def __init__(self) -> None:
        self.units_verified: int = 0
        self.redundant_verifications: int = 0
        self.disagreements: int = 0
        self.nodes_penalized: int = 0
        self.degraded_no_secondary: int = 0
        self.degraded_secondary_error: int = 0
        self.degraded_no_tiebreak: int = 0
        self.stake_slashed: float = 0.0
        self.heldback_recorded: float = 0.0
        self.heldback_released: float = 0.0

    def as_dict(self) -> dict:
        return {
            "units_verified": self.units_verified,
            "redundant_verifications": self.redundant_verifications,
            "disagreements": self.disagreements,
            "nodes_penalized": self.nodes_penalized,
            "degraded": {
                "no_secondary": self.degraded_no_secondary,
                "secondary_error": self.degraded_secondary_error,
                "no_tiebreak": self.degraded_no_tiebreak,
            },
            "stake_slashed": self.stake_slashed,
            "heldback_recorded": self.heldback_recorded,
            "heldback_released": self.heldback_released,
        }


class StakingEngine:
    """Graduated staking and slashing for public-mesh fraud detection.

    A brand-new node must risk roughly 20x the expected value of a job.  As the
    TrustEngine records value-weighted Beta evidence, ``required_stake`` decays.
    On detected fraud the stake is transferred to a real destination account via
    a balanced ledger transfer; no credits are minted or destroyed.
    """

    def __init__(self, ledger: ContributionLedger, trust=None,
                 destination_owner: str = SLASH_DESTINATION_OWNER) -> None:
        self.ledger = ledger
        self.trust = trust
        self.destination_owner = destination_owner
        self.locked_by_node: dict[str, dict[str, float | str]] = {}

    def required_stake(self, node_id: str, expected_job_value: float) -> float:
        if self.trust is not None:
            return self.trust.required_stake(node_id, expected_job_value)
        return max(0.0, expected_job_value) * BASE_STAKE_MULTIPLIER

    def ensure_stake(self, node_id: str, owner: str, expected_job_value: float) -> float:
        required = self.required_stake(node_id, expected_job_value)
        state = self.locked_by_node.setdefault(node_id, {"owner": owner, "locked": 0.0})
        state["owner"] = owner
        state["locked"] = max(float(state.get("locked", 0.0)), required)
        return required

    def slash(self, node_id: str, expected_job_value: float = 0.0,
              reason: str = "detected fraud") -> float:
        state = self.locked_by_node.get(node_id)
        owner = str(state.get("owner")) if state else f"dev:{node_id}"
        locked = float(state.get("locked", 0.0)) if state else 0.0
        amount = locked if locked > 0.0 else self.required_stake(node_id, expected_job_value)
        if amount <= 0.0:
            return 0.0
        self.ledger.record_slash(owner, amount, self.destination_owner,
                                 node_id=node_id, detail=reason)
        if state is not None:
            state["locked"] = 0.0
        return amount


class HeldBackEscrowEngine:
    """Storj-style held-back escrow: 50% -> 25% -> 0% over nine months."""

    def __init__(self, ledger: ContributionLedger, clock_ms=now_ms,
                 escrow_owner: str = ESCROW_OWNER) -> None:
        self.ledger = ledger
        self.clock_ms = clock_ms
        self.escrow_owner = escrow_owner
        self.first_seen_ms: dict[str, int] = {}
        self.held: dict[str, dict[str, float | str | int]] = {}

    def holdback_rate(self, node_id: str, at_ms: Optional[int] = None) -> float:
        now = self.clock_ms() if at_ms is None else int(at_ms)
        first = self.first_seen_ms.setdefault(node_id, now)
        age = max(0, now - first)
        if age < THREE_MONTHS_MS:
            return 0.50
        if age < NINE_MONTHS_MS:
            return 0.25
        return 0.0

    def record_earning(self, node_id: str, owner: str, amount: float,
                       detail: str = "") -> tuple[float, float]:
        amount = max(0.0, amount)
        if amount < MIN_HOLDBACK_EARNING:
            self.ledger.record_donation(node_id, owner, amount, detail=detail)
            return amount, 0.0
        rate = self.holdback_rate(node_id)
        held_amount = amount * rate
        immediate = amount - held_amount
        if immediate > 0.0:
            self.ledger.record_donation(node_id, owner, immediate, detail=detail)
        if held_amount > 0.0:
            self.ledger.record_donation(node_id, self.escrow_owner, held_amount,
                                        detail=f"heldback:{owner}:{detail}")
            state = self.held.setdefault(
                node_id, {"owner": owner, "held": 0.0, "release_after_ms": 0})
            state["owner"] = owner
            state["held"] = float(state.get("held", 0.0)) + held_amount
            state["release_after_ms"] = self.first_seen_ms[node_id] + NINE_MONTHS_MS
        return immediate, held_amount

    def release_due(self, node_id: str, at_ms: Optional[int] = None) -> float:
        now = self.clock_ms() if at_ms is None else int(at_ms)
        state = self.held.get(node_id)
        if not state:
            return 0.0
        if now < int(state.get("release_after_ms", 0)):
            return 0.0
        amount = float(state.get("held", 0.0))
        if amount <= 0.0:
            return 0.0
        owner = str(state.get("owner"))
        self.ledger.record_transfer(self.escrow_owner, owner, amount,
                                    detail="heldback escrow release",
                                    node_id=node_id)
        state["held"] = 0.0
        return amount

    def total_held(self) -> float:
        return sum(float(s.get("held", 0.0)) for s in self.held.values())


@dataclass(frozen=True)
class BisectionVerdict:
    first_disagreement: Optional[int]
    rounds: int
    localized: bool


class InteractiveBisectionDispute:
    """Truebit-style O(log N) dispute localization over committed step hashes."""

    @staticmethod
    def hash_step(value: Any) -> str:
        return _result_digest(value)

    def localize(self, claimant_trace: list[Any], challenger_trace: list[Any]) -> BisectionVerdict:
        if len(claimant_trace) != len(challenger_trace):
            raise ValueError("trace lengths differ")
        if not claimant_trace:
            return BisectionVerdict(None, 0, False)
        c = [self.hash_step(v) for v in claimant_trace]
        h = [self.hash_step(v) for v in challenger_trace]
        if c[-1] == h[-1]:
            return BisectionVerdict(None, 1, False)
        lo, hi = 0, len(c) - 1
        rounds = 1
        while lo < hi:
            mid = (lo + hi) // 2
            rounds += 1
            if c[mid] == h[mid]:
                lo = mid + 1
            else:
                hi = mid
        return BisectionVerdict(lo, rounds, True)


# ---------------------------------------------------------------------------
# Digest comparison helper
# ---------------------------------------------------------------------------

def _result_digest(result: Any) -> str:
    """Compute a deterministic SHA-256 digest of a result for comparison.

    Uses JSON serialisation with sorted keys to ensure consistent output
    regardless of dict insertion order.  Falls back to repr() if the result
    is not JSON-serialisable (should not happen for curated functions).
    """
    try:
        serialised = json.dumps(result, sort_keys=True, separators=(",", ":"),
                                default=str)
    except (TypeError, ValueError):
        serialised = repr(result)
    return hashlib.sha256(serialised.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Verifier (thin orchestration layer injected into BatchManager)
# ---------------------------------------------------------------------------

class Verifier:
    """Orchestrates workload-matched result verification.

    Wired into BatchManager post-construction by build_app() -- same injection
    pattern as spend_caps and storage.  The batch engine calls the verifier's
    helpers from _run_unit() and _redundant_check(); the verifier never
    calls back into the engine.

    Provides:
      policy         -- VerificationPolicy (classify, decide, should_trigger_redundant)
      compute_escrow -- ComputeEscrowEngine (hold/release/forfeit bonus credits)
      stats          -- VerificationStats (observable counters for the dashboard)
      digests_match  -- compare two results by their canonical SHA-256 digest
    """

    def __init__(
        self,
        ledger: Optional[ContributionLedger] = None,
        policy: Optional[VerificationPolicy] = None,
        trust=None,
        clock_ms=now_ms,
    ) -> None:
        self.policy = policy or VerificationPolicy()
        self.compute_escrow: Optional[ComputeEscrowEngine] = (
            ComputeEscrowEngine(ledger) if ledger is not None else None)
        self.staking: Optional[StakingEngine] = (
            StakingEngine(ledger, trust=trust) if ledger is not None else None)
        self.heldback: Optional[HeldBackEscrowEngine] = (
            HeldBackEscrowEngine(ledger, clock_ms=clock_ms) if ledger is not None else None)
        self.stats = VerificationStats()

    def digests_match(self, result_a: Any, result_b: Any) -> bool:
        """True iff both results produce the same canonical SHA-256 digest."""
        return _result_digest(result_a) == _result_digest(result_b)

    def should_trigger_redundant(self, func: str, rng: random.Random) -> bool:
        """Whether this unit should undergo redundant execution."""
        return self.policy.should_trigger_redundant(func, rng)

    def ensure_stake(self, node_id: str, owner: str, expected_job_value: float) -> float:
        if self.staking is None:
            return 0.0
        return self.staking.ensure_stake(node_id, owner, expected_job_value)

    def record_pass(self, node_id: str, owner: str) -> float:
        """Update escrow and stats for a verified-pass result."""
        self.stats.units_verified += 1
        released = 0.0
        if self.compute_escrow is not None:
            released = self.compute_escrow.on_verify_pass(node_id, owner)
        return released

    def record_fail(self, node_id: str, expected_job_value: float = 0.0) -> float:
        """Update escrow and stats for a verified-fail result."""
        self.stats.nodes_penalized += 1
        forfeited = 0.0
        if self.compute_escrow is not None:
            forfeited = self.compute_escrow.on_verify_fail(node_id)
        if self.staking is not None:
            self.stats.stake_slashed += self.staking.slash(
                node_id, expected_job_value=expected_job_value)
        return forfeited

    def record_earning(self, node_id: str, owner: str, amount: float,
                       detail: str = "") -> tuple[float, float]:
        if self.heldback is None:
            return amount, 0.0
        immediate, held = self.heldback.record_earning(node_id, owner, amount, detail)
        self.stats.heldback_recorded += held
        released = self.heldback.release_due(node_id)
        self.stats.heldback_released += released
        return immediate, held

    def as_dict(self) -> dict:
        """Serialisable snapshot for the /api/v1/verification endpoint."""
        esc = self.compute_escrow
        return {
            "policy": {
                "redundant_rate": self.policy.redundant_rate,
                "redundant_n": self.policy.redundant_n,
                "active": True,
            },
            "stats": self.stats.as_dict(),
            "escrow": {
                "total_held": esc.total_held() if esc else 0.0,
                "total_released": esc.total_released if esc else 0.0,
                "total_forfeited": esc.total_forfeited if esc else 0.0,
            },
            "staking": {
                "active": self.staking is not None,
                "destination_owner": (
                    self.staking.destination_owner if self.staking else SLASH_DESTINATION_OWNER),
                "total_slashed": self.stats.stake_slashed,
            },
            "heldback": {
                "active": self.heldback is not None,
                "escrow_owner": self.heldback.escrow_owner if self.heldback else ESCROW_OWNER,
                "total_held": self.heldback.total_held() if self.heldback else 0.0,
                "total_recorded": self.stats.heldback_recorded,
                "total_released": self.stats.heldback_released,
            },
            # The endpoint's Tier 0/1 branch returns this block, so omitting it
            # here would make the response *shape* depend on the deployment tier:
            # a client that reads `classification` works against a tier-0 mesh
            # and KeyErrors against a tier-2 one. The classification itself is
            # static policy -- which functions are digest-verifiable, oracle-
            # verifiable, or not verifiable at all -- so there is no reason for
            # it to disappear once verification is actually switched on. If
            # anything it matters *more* then.
            "classification": {
                "digest_verifiable": sorted(DIGEST_VERIFIABLE_FUNCS),
                "oracle_verifiable": sorted(ORACLE_VERIFIABLE_FUNCS),
                "not_verifiable": sorted(NOT_VERIFIABLE_FUNCS),
            },
        }
