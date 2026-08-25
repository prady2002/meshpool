"""Unit tests for the Vivaldi network coordinate system.

These tests verify:
  1. Basic API contracts (pure functions, immutability, correct types).
  2. Correctness of the geometric helpers (euclidean, distance_to).
  3. Properties of the update rule (error decreases over time, coordinate moves).
  4. Cold-start: all nodes at the origin converge to a sensible embedding.
  5. Convergence quality on a synthetic 3-cluster topology.
     MEASURED result is printed and asserted (median relative error < 0.15).
  6. Edge cases: new node (high error, not trusted for placement), negative RTT.
"""
from __future__ import annotations

import math
import random

import pytest

from meshpool.common.vivaldi import (
    Ce, Cc, Coordinate, NDIM,
    HIGH_ERROR_THRESHOLD,
    _unit_vec,
    converge,
    distance_to,
    euclidean,
    median_relative_error,
    update,
)


# ---------------------------------------------------------------------------
# 1. API contracts
# ---------------------------------------------------------------------------

class TestCoordinate:
    def test_origin_high_error(self):
        c = Coordinate.origin()
        assert c.error == 1.0
        assert c.n_samples == 0
        assert len(c.vec) == NDIM
        assert all(v == 0.0 for v in c.vec)

    def test_copy_is_independent(self):
        c = Coordinate(vec=[1.0, 2.0], height=0.5, error=0.3, n_samples=5)
        d = c.copy()
        d.vec[0] = 99.0
        assert c.vec[0] == 1.0, "copy must be independent"

    def test_not_trusted_fresh(self):
        c = Coordinate.origin()
        assert not c.is_trusted()

    def test_trusted_after_enough_samples(self):
        c = Coordinate(vec=[1.0, 0.0], height=0.01, error=0.2, n_samples=10)
        assert c.is_trusted()

    def test_not_trusted_high_error(self):
        c = Coordinate(vec=[1.0, 0.0], height=0.01,
                       error=HIGH_ERROR_THRESHOLD, n_samples=100)
        assert not c.is_trusted()


class TestGeometry:
    def test_euclidean_zero(self):
        assert euclidean([0.0, 0.0], [0.0, 0.0]) == 0.0

    def test_euclidean_pythagorean(self):
        # 3-4-5 right triangle
        assert abs(euclidean([0.0, 0.0], [3.0, 4.0]) - 5.0) < 1e-9

    def test_distance_to_includes_heights(self):
        a = Coordinate(vec=[0.0, 0.0], height=1.0, error=0.1)
        b = Coordinate(vec=[3.0, 4.0], height=2.0, error=0.1)
        # euclidean = 5, heights = 1+2 = 3 → total = 8
        assert abs(distance_to(a, b) - 8.0) < 1e-9

    def test_distance_symmetric(self):
        a = Coordinate(vec=[1.0, 2.0], height=0.5, error=0.1)
        b = Coordinate(vec=[4.0, 6.0], height=1.5, error=0.2)
        assert abs(distance_to(a, b) - distance_to(b, a)) < 1e-9


# ---------------------------------------------------------------------------
# 2. Unit-vector helper (including random-direction case)
# ---------------------------------------------------------------------------

class TestUnitVec:
    def test_normal_case(self):
        u = _unit_vec([3.0, 4.0])
        assert abs(math.sqrt(u[0] ** 2 + u[1] ** 2) - 1.0) < 1e-9

    def test_zero_vector_returns_unit(self):
        rng = random.Random(1)
        u = _unit_vec([0.0, 0.0], rng=rng)
        assert abs(math.sqrt(u[0] ** 2 + u[1] ** 2) - 1.0) < 1e-9

    def test_zero_vector_different_seeds_give_different_directions(self):
        dirs = set()
        for seed in range(20):
            rng = random.Random(seed)
            u = _unit_vec([0.0, 0.0], rng=rng)
            dirs.add(round(u[0], 3))
        # At least 5 distinct x-components → random directions are diverse
        assert len(dirs) >= 5


# ---------------------------------------------------------------------------
# 3. Update rule properties
# ---------------------------------------------------------------------------

class TestUpdate:
    def test_returns_new_coordinate(self):
        local = Coordinate.origin()
        peer = Coordinate(vec=[100.0, 0.0], height=0.01, error=0.5)
        updated = update(local, peer, 100.0, rng=random.Random(0))
        assert updated is not local   # must be a new object
        assert local.vec == [0.0, 0.0]  # input not mutated

    def test_negative_rtt_raises(self):
        local = Coordinate.origin()
        peer = Coordinate(vec=[1.0, 0.0], height=0.01, error=0.5)
        with pytest.raises(ValueError):
            update(local, peer, -1.0)

    def test_zero_rtt_raises(self):
        local = Coordinate.origin()
        peer = Coordinate(vec=[1.0, 0.0], height=0.01, error=0.5)
        with pytest.raises(ValueError):
            update(local, peer, 0.0)

    def test_n_samples_incremented(self):
        local = Coordinate(vec=[0.0, 0.0], height=0.01, error=1.0, n_samples=7)
        peer = Coordinate(vec=[50.0, 0.0], height=0.01, error=0.5)
        updated = update(local, peer, 50.0, rng=random.Random(0))
        assert updated.n_samples == 8

    def test_error_stays_in_bounds(self):
        rng = random.Random(42)
        local = Coordinate.origin()
        peer = Coordinate(vec=[50.0, 50.0], height=0.01, error=0.5)
        for _ in range(100):
            local = update(local, peer, 70.0, rng=rng)
        assert 0.0 <= local.error <= 1.0

    def test_height_stays_positive(self):
        rng = random.Random(42)
        local = Coordinate.origin()
        peer = Coordinate(vec=[0.0, 0.0], height=0.01, error=0.5)
        # Very small RTT will try to shrink height below zero
        for _ in range(200):
            local = update(local, peer, 0.001, rng=rng)
        assert local.height > 0.0, "height must never go negative"

    def test_coordinate_moves_toward_correct_position(self):
        # If peer is at (100, 0) and true RTT is 100 ms,
        # local starts at (0, 0).  After many updates, predicted RTT should
        # converge toward 100 ms (ignoring heights).
        rng = random.Random(0)
        peer = Coordinate(vec=[100.0, 0.0], height=0.0, error=0.01)
        # Force peer height to ~0 to isolate the Euclidean part
        peer.height = 0.001
        local = Coordinate(vec=[0.0, 0.0], height=0.001, error=1.0)
        for _ in range(500):
            local = update(local, peer, 100.0, rng=rng)
        pred = distance_to(local, peer)
        assert abs(pred - 100.0) / 100.0 < 0.10, (
            f"predicted RTT {pred:.1f} ms should be close to 100 ms after 500 updates"
        )

    def test_error_decreases_on_consistent_observations(self):
        """Error should trend downward when observations are consistent."""
        rng = random.Random(0)
        peer = Coordinate(vec=[50.0, 0.0], height=0.01, error=0.3)
        local = Coordinate(vec=[0.0, 0.0], height=0.01, error=1.0)
        for _ in range(300):
            local = update(local, peer, 50.0, rng=rng)
        assert local.error < 0.5, (
            f"error {local.error:.3f} should have decreased from 1.0 after "
            "300 consistent observations"
        )


# ---------------------------------------------------------------------------
# 4. Cold-start: all nodes at origin
# ---------------------------------------------------------------------------

def test_cold_start_all_at_origin():
    """Nodes starting at the exact same coordinate must still separate.

    This is the classic degenerate case: all coordinates are [0, 0], so the
    difference vector between any pair is [0, 0], and a naive implementation
    would add a zero force to every coordinate forever.  The random-direction
    fallback in _unit_vec is the fix.  We verify it works by checking that
    after 500 updates on a 3-node ring, the nodes are at distinct positions.
    """
    rng = random.Random(99)
    # 3 nodes, all at origin.  True RTTs: A-B=50ms, B-C=60ms, A-C=100ms
    nodes = {
        "A": Coordinate.origin(),
        "B": Coordinate.origin(),
        "C": Coordinate.origin(),
    }
    true_rtt = {"A": {"B": 50, "C": 100}, "B": {"A": 50, "C": 60},
                "C": {"A": 100, "B": 60}}

    def rtt_fn(a, b):
        return true_rtt[a][b]

    converged = converge(nodes, rtt_fn, n_iter=500, rng=rng)

    # All three positions must be distinct
    positions = [tuple(converged[k].vec) for k in ["A", "B", "C"]]
    assert positions[0] != positions[1], "A and B must separate from origin"
    assert positions[1] != positions[2], "B and C must separate from origin"
    assert positions[0] != positions[2], "A and C must separate from origin"


# ---------------------------------------------------------------------------
# 5. Convergence quality on a 3-cluster synthetic topology
# ---------------------------------------------------------------------------

def _build_3cluster_rtt():
    """9 nodes in 3 clusters.  Returns a (nodes, rtt_fn) pair.

    Cluster membership:
      C0: nodes 0,1,2 — intra-cluster 5 ms
      C1: nodes 3,4,5 — intra-cluster 5 ms
      C2: nodes 6,7,8 — intra-cluster 5 ms
    Inter-cluster RTTs (symmetric):
      C0 <-> C1 : 80 ms
      C1 <-> C2 : 80 ms
      C0 <-> C2 : 150 ms
    """
    n_nodes = 9
    ids = [f"n{i}" for i in range(n_nodes)]
    cluster = {ids[i]: i // 3 for i in range(n_nodes)}
    inter = {frozenset({0, 2}): 150.0}  # C0-C2

    def rtt_fn(a, b):
        ca, cb = cluster[a], cluster[b]
        if ca == cb:
            return 5.0
        return inter.get(frozenset({ca, cb}), 80.0)

    return ids, rtt_fn


def test_convergence_quality_measured():
    """Assert median relative prediction error < 0.15 on a 3-cluster topology.

    The actual median is printed so the gate report can quote it.  This is a
    MEASURED number, not a target — it reflects the real quality of the Vivaldi
    algorithm on this topology.
    """
    ids, rtt_fn = _build_3cluster_rtt()
    coords = {nid: Coordinate.origin() for nid in ids}
    rng = random.Random(42)

    converged = converge(coords, rtt_fn, n_iter=2000, rng=rng)
    err = median_relative_error(converged, rtt_fn)
    print(f"\n  [MEASURED] 3-cluster median relative error: {err:.4f} ({err*100:.1f}%)")

    assert err < 0.15, (
        f"Vivaldi failed to converge to <15% median relative error; got {err:.4f}. "
        "Check the algorithm constants (Ce, Cc) and the update rule."
    )


def test_intra_cluster_closer_than_inter():
    """Nodes in the same cluster must end up Euclidean-closer than remote nodes."""
    ids, rtt_fn = _build_3cluster_rtt()
    coords = {nid: Coordinate.origin() for nid in ids}
    rng = random.Random(7)

    converged = converge(coords, rtt_fn, n_iter=2000, rng=rng)

    # Pick a representative from each cluster
    c0, c1, c2 = converged["n0"], converged["n3"], converged["n6"]

    # n0 and n1 are in the same cluster
    intra = distance_to(converged["n0"], converged["n1"])
    inter_01 = distance_to(c0, c1)
    inter_02 = distance_to(c0, c2)

    print(f"\n  [MEASURED] intra-cluster predicted RTT: {intra:.1f} ms")
    print(f"  [MEASURED] C0<->C1 predicted RTT:        {inter_01:.1f} ms")
    print(f"  [MEASURED] C0<->C2 predicted RTT:        {inter_02:.1f} ms")

    assert intra < inter_01, (
        f"intra-cluster pair ({intra:.1f} ms) must be closer than "
        f"C0-C1 ({inter_01:.1f} ms)"
    )
    assert intra < inter_02, (
        f"intra-cluster pair ({intra:.1f} ms) must be closer than "
        f"C0-C2 ({inter_02:.1f} ms)"
    )
    # The wider pair (C0-C2 = 150 ms) should be predicted further than C0-C1 (80 ms)
    assert inter_02 > inter_01, (
        f"C0-C2 ({inter_02:.1f} ms) should be predicted further than "
        f"C0-C1 ({inter_01:.1f} ms)"
    )


# ---------------------------------------------------------------------------
# 6. Edge cases
# ---------------------------------------------------------------------------

def test_fresh_node_not_trusted_for_placement():
    """A brand-new node with error=1.0 must not be trusted for placement."""
    fresh = Coordinate.origin()
    assert not fresh.is_trusted()
    # Even with some samples, if error is high, not trusted
    high_err = Coordinate(vec=[10.0, 10.0], height=0.01,
                          error=HIGH_ERROR_THRESHOLD, n_samples=50)
    assert not high_err.is_trusted()


def test_high_latency_link_node():
    """A node behind a high-latency last-mile link still converges to the correct RTT.

    In Vivaldi, the Euclidean coordinate absorbs the RTT distance — if the peer
    is "close" in network topology, the local coordinate moves far away in
    embedding space to match the observed latency.  Heights model systematic
    triangle-inequality violations (multiple peers seeing the same unexplained
    offset), not raw latency magnitude.  So we assert convergence of the
    *predicted RTT*, not the height value.
    """
    local = Coordinate.origin()
    peer = Coordinate(vec=[50.0, 0.0], height=0.01, error=0.3)
    rng = random.Random(0)
    for _ in range(500):
        local = update(local, peer, 250.0, rng=rng)
    pred = distance_to(local, peer)
    print(f"\n  [MEASURED] after 500 updates with rtt=250ms: "
          f"predicted={pred:.1f}ms height={local.height:.3f}ms")
    # Coordinate must converge: predicted RTT close to true RTT
    assert abs(pred - 250.0) / 250.0 < 0.15, (
        f"predicted RTT {pred:.1f} ms should be near 250 ms after 500 updates"
    )
    assert local.height > 0.0, "height must stay positive"


def test_coordinate_drift_recovery():
    """After the network changes, coordinates should reconverge."""
    rng = random.Random(0)
    peer = Coordinate(vec=[50.0, 0.0], height=0.01, error=0.3)
    local = Coordinate(vec=[0.0, 0.0], height=0.01, error=1.0)

    # Phase 1: converge to RTT=50 ms
    for _ in range(400):
        local = update(local, peer, 50.0, rng=rng)
    pred_phase1 = distance_to(local, peer)

    # Phase 2: network changes, new RTT=150 ms — coordinate should reconverge
    for _ in range(400):
        local = update(local, peer, 150.0, rng=rng)
    pred_phase2 = distance_to(local, peer)

    print(f"\n  [MEASURED] before network change: {pred_phase1:.1f} ms → "
          f"after change: {pred_phase2:.1f} ms (target: ~150 ms)")

    assert pred_phase2 > pred_phase1, (
        "coordinate must drift further from peer when RTT increases"
    )
    # Should converge to something near 150 ms
    assert abs(pred_phase2 - 150.0) / 150.0 < 0.30, (
        f"after reconvergence, predicted RTT {pred_phase2:.1f} ms should be near 150 ms"
    )
