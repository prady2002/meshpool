"""Vivaldi network coordinate system (Dabek et al., SIGCOMM 2004).

Each node holds a 2-D Euclidean coordinate plus a *height* (last-mile latency
that violates the triangle inequality) and an *error* estimate.  After enough
RTT observations the Euclidean distance between two coordinates approximates
the RTT between the two machines without an O(n^2) measurement matrix.

Algorithm
---------
On each observation of RTT (ms) to a peer::

    pred   = ||x_l - x_r|| + h_l + h_r       # predicted RTT (Euclidean + heights)
    e_s    = |pred - rtt| / rtt               # sample relative error
    w      = e_local / (e_local + e_remote)   # trust weight (smaller error = more trusted)
    e_l    = e_s * Ce * w + e_l * (1 - Ce*w) # EWMA error update
    delta  = Cc * w
    x_l    = x_l + delta * (rtt - pred) * unit(x_l - x_r)
    h_l    = max(0.001, h_l + delta * (rtt - pred) * h_l / rtt)

where ``unit(v)`` returns a *random* direction when ``||v||`` is near zero ---
this is critical: without it, nodes that start at the same coordinate (the cold-
start case) would never separate because every update would add a zero vector.

This module is pure, stateless, and carries no I/O; it is fully unit-testable.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Optional

# --------------------------------------------------------------------------- #
# Algorithm constants (Section 4 of Dabek et al. 2004)
# --------------------------------------------------------------------------- #
Ce: float = 0.25   # error-estimate EWMA update rate
Cc: float = 0.25   # coordinate step-size scaling

NDIM: int = 2      # dimensionality of the Euclidean space

# Nodes with error >= this threshold are "not yet trusted" for placement.
HIGH_ERROR_THRESHOLD: float = 0.8

_EPS: float = 1e-10   # guard against zero-length vectors / division-by-zero


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #
@dataclass
class Coordinate:
    """A Vivaldi network coordinate for one node.

    Attributes:
        vec:       Position in the 2-D synthetic coordinate space (ms units).
        height:    Last-mile latency that violates the triangle inequality (ms).
                   Always positive; seeded to 0.01 so height-only RTTs are
                   non-zero from the start.
        error:     Relative error estimate in [0, 1].  1.0 = completely
                   untrustworthy (fresh node); 0.0 = perfect predictor (never
                   reached in practice).
        n_samples: Number of RTT observations incorporated so far.
    """

    vec: list[float] = field(default_factory=lambda: [0.0, 0.0])
    height: float = 0.01
    error: float = 1.0
    n_samples: int = 0

    def copy(self) -> "Coordinate":
        """Return a shallow copy (vec is a new list)."""
        return Coordinate(list(self.vec), self.height, self.error, self.n_samples)

    @classmethod
    def origin(cls) -> "Coordinate":
        """The default 'I know nothing' coordinate: origin + high error."""
        return cls(vec=[0.0] * NDIM, height=0.01, error=1.0, n_samples=0)

    def is_trusted(self) -> bool:
        """True when the coordinate has converged enough to be used for placement."""
        return self.n_samples >= 3 and self.error < HIGH_ERROR_THRESHOLD


# --------------------------------------------------------------------------- #
# Core geometry helpers
# --------------------------------------------------------------------------- #
def _unit_vec(v: list[float], rng: Optional[random.Random] = None) -> list[float]:
    """Return the unit vector of *v*; a random direction when *v* is near-zero.

    The random-direction case is **critical** for cold-start correctness.
    Two nodes that start at exactly the same coordinate (both at the origin)
    would have ``v = [0, 0]``.  Without the random fallback every update adds
    ``delta * (rtt - pred) * [0, 0] = [0, 0]``, so the nodes can never
    separate regardless of the observed RTT.
    """
    n = math.sqrt(sum(x * x for x in v))
    if n < _EPS:
        r = rng if rng is not None else random
        theta = r.uniform(0.0, 2.0 * math.pi)
        return [math.cos(theta), math.sin(theta)]
    return [x / n for x in v]


def euclidean(a: list[float], b: list[float]) -> float:
    """Euclidean distance between two coordinate vectors."""
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def distance_to(a: Coordinate, b: Coordinate) -> float:
    """Predicted RTT (ms) between two nodes: Euclidean distance + both heights."""
    return euclidean(a.vec, b.vec) + a.height + b.height


# --------------------------------------------------------------------------- #
# Core update rule
# --------------------------------------------------------------------------- #
def update(
    local: Coordinate,
    peer: Coordinate,
    rtt_ms: float,
    rng: Optional[random.Random] = None,
) -> Coordinate:
    """Return an updated copy of *local* after observing *rtt_ms* to *peer*.

    This is a **pure function**: neither *local* nor *peer* is mutated.

    Args:
        local:   The node whose coordinate is being updated.
        peer:    The remote node whose RTT was just measured.
        rtt_ms:  Observed round-trip time in milliseconds.  Must be > 0.
        rng:     Optional :class:`random.Random` for reproducible random
                 directions in the degenerate (same-position) case.

    Returns:
        A new :class:`Coordinate` reflecting the update.

    Raises:
        ValueError: if *rtt_ms* is not positive.
    """
    if rtt_ms <= 0.0:
        raise ValueError(f"RTT must be positive, got {rtt_ms!r}")

    # ------------------------------------------------------------------ #
    # Step 1: predicted RTT from current coordinates
    # ------------------------------------------------------------------ #
    pred_ms = distance_to(local, peer)
    pred_ms = max(pred_ms, _EPS)   # guard against near-zero in division below

    # ------------------------------------------------------------------ #
    # Step 2: relative-error weight (lower error = higher confidence)
    # ------------------------------------------------------------------ #
    w = local.error / (local.error + peer.error + _EPS)

    # ------------------------------------------------------------------ #
    # Step 3: sample relative error and EWMA update
    # ------------------------------------------------------------------ #
    e_s = abs(pred_ms - rtt_ms) / rtt_ms
    e_s = min(e_s, 1.0)   # clamp sample before EWMA so extreme outliers don't explode

    new_error = e_s * Ce * w + local.error * (1.0 - Ce * w)
    new_error = max(0.0, min(1.0, new_error))

    # ------------------------------------------------------------------ #
    # Step 4: coordinate update
    # ------------------------------------------------------------------ #
    delta = Cc * w

    diff = [local.vec[i] - peer.vec[i] for i in range(NDIM)]
    u = _unit_vec(diff, rng)

    force = delta * (rtt_ms - pred_ms)
    new_vec = [local.vec[i] + force * u[i] for i in range(NDIM)]

    # ------------------------------------------------------------------ #
    # Step 5: height update (must stay positive)
    # ------------------------------------------------------------------ #
    height_delta = delta * (rtt_ms - pred_ms) * local.height / rtt_ms
    new_height = max(0.001, local.height + height_delta)

    return Coordinate(
        vec=new_vec,
        height=new_height,
        error=new_error,
        n_samples=local.n_samples + 1,
    )


# --------------------------------------------------------------------------- #
# Convenience: simulate convergence for a set of nodes
# --------------------------------------------------------------------------- #
def converge(
    coords: dict[str, Coordinate],
    rtt_fn,
    n_iter: int = 2000,
    rng: Optional[random.Random] = None,
) -> dict[str, Coordinate]:
    """Run *n_iter* Vivaldi update steps on a set of in-memory coordinates.

    ``rtt_fn(a_id, b_id) -> float`` must return the true RTT (ms) between any
    pair.  At each step one random pair is chosen and both nodes are updated
    (symmetrically -- RTT A->B == RTT B->A in this model).

    This is used by the unit-test convergence check and by the gate's mutation
    guard.  It is NOT used by the production path (the production path updates
    coordinates one measurement at a time as heartbeats arrive).
    """
    r = rng if rng is not None else random.Random(42)
    ids = list(coords.keys())
    coords = {k: v.copy() for k, v in coords.items()}  # don't mutate caller's dict

    for _ in range(n_iter):
        a_id, b_id = r.sample(ids, 2)
        rtt = rtt_fn(a_id, b_id)
        # Update the node with the higher error first (it needs more correction).
        if coords[a_id].error >= coords[b_id].error:
            coords[a_id] = update(coords[a_id], coords[b_id], rtt, rng=r)
        else:
            coords[b_id] = update(coords[b_id], coords[a_id], rtt, rng=r)

    return coords


def median_relative_error(
    coords: dict[str, Coordinate],
    rtt_fn,
) -> float:
    """Compute the median relative prediction error over all node pairs.

    ``|predicted - true| / true`` for every distinct ordered pair (a, b) where
    a != b.  Returns 0.0 for a single-node dict.
    """
    ids = list(coords.keys())
    errors = []
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            a, b = ids[i], ids[j]
            true_rtt = rtt_fn(a, b)
            pred_rtt = distance_to(coords[a], coords[b])
            errors.append(abs(pred_rtt - true_rtt) / max(true_rtt, _EPS))
    if not errors:
        return 0.0
    errors.sort()
    mid = len(errors) // 2
    return errors[mid] if len(errors) % 2 == 1 else (errors[mid - 1] + errors[mid]) / 2.0
