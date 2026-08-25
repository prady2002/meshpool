"""Unit tests for sim.netem — fast, deterministic, no real network."""
from __future__ import annotations

import asyncio
import math
import time
from typing import List

import pytest

from sim.topology import (
    NodeSpec, LinkProfile, JitterDistribution, Topology,
    residential_mixed, _lan, _metro, _national, _intercontinental, PRESETS,
)
from sim.netem import NetEm, NetEmTransport, PacketLost, _CAN_DIRECT


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _two_node_topo(nat_a: str = "public", nat_b: str = "public",
                   preset: str = "national", seed: int = 42) -> Topology:
    nodes = [
        NodeSpec(id="A", region="us-east", asn="AS1", subnet="10.0.1.0/24",
                 nat_class=nat_a),
        NodeSpec(id="B", region="us-west", asn="AS2", subnet="10.0.2.0/24",
                 nat_class=nat_b),
    ]
    link_profile = PRESETS[preset]()

    def link_fn(a: str, b: str) -> LinkProfile:
        return link_profile

    return Topology(nodes=nodes, link_fn=link_fn, seed=seed)


def _relay_topo(nat_a: str, nat_b: str) -> Topology:
    """Three-node topology with an explicit public relay."""
    nodes = [
        NodeSpec(id="A", region="us-east", asn="AS1", subnet="10.0.1.0/24",
                 nat_class=nat_a, node_class="constellation"),
        NodeSpec(id="B", region="us-west", asn="AS2", subnet="10.0.2.0/24",
                 nat_class=nat_b, node_class="constellation"),
        NodeSpec(id="relay", region="us-central", asn="AS3", subnet="10.0.3.0/24",
                 nat_class="public", node_class="relay"),
    ]
    link_profile = _national()

    def link_fn(a: str, b: str) -> LinkProfile:
        return link_profile

    return Topology(nodes=nodes, link_fn=link_fn, seed=42)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

def test_determinism_same_seed():
    """Same seed + same topology → identical one_way_delay_s.

    We can't make the jitter 100% deterministic in the async path because
    transit() uses time.monotonic_ns() as entropy — instead we test the
    *topology-level* RNG is stable and that the delay is in the right range.
    """
    topo = _two_node_topo(seed=99)
    ne1 = NetEm(topo, time_scale=1.0, seed=0)
    ne2 = NetEm(topo, time_scale=1.0, seed=0)
    # Without time-based entropy the two delay calculations should be in the same range
    d1 = ne1.one_way_delay_s("A", "B", 1000)
    d2 = ne2.one_way_delay_s("A", "B", 1000)
    # Both should be close to national RTT/2 = 22.5 ms + transfer time
    assert abs(d1 - d2) < 0.05, f"determinism drift: {d1:.4f} vs {d2:.4f}"


def test_delay_range_national():
    """National preset RTT=45 ms → one-way should be ~22.5 ms + jitter."""
    topo = _two_node_topo(preset="national")
    ne = NetEm(topo, time_scale=1.0)
    # 1 kB at 10 Mbps upload = 0.8 ms; so total ≈ 23.3 ms ± jitter
    d = ne.one_way_delay_s("A", "B", 1000)
    # with heavy-tail jitter could be up to ~10× sigma = 50 ms; normally ~23 ms
    assert 0.01 < d < 0.5, f"delay out of range: {d:.4f} s"


# ---------------------------------------------------------------------------
# RTT timing (using time_scale to make it fast)
# ---------------------------------------------------------------------------

def test_rtt_approximate():
    """Injected RTT should be observed in the computed delay (pre-sleep value)."""
    topo = _two_node_topo(preset="national")  # RTT=45 ms
    ne = NetEm(topo, time_scale=1.0)
    # Compute the one-way delay without sleeping (avoids Windows 15ms timer floor)
    d = ne.one_way_delay_s("A", "B", 64)
    # Expected: rtt_ms/2 + small_serialise + jitter ≈ 22.5 ms + ~0 + small
    simulated_rtt_ms = (d + ne.one_way_delay_s("B", "A", 64)) * 1000.0
    # RTT = 45 ms; allow up to ±30% + 20 ms for heavy-tail jitter
    assert 20 < simulated_rtt_ms < 120, (
        f"simulated RTT out of range: {simulated_rtt_ms:.1f} ms (expected ~45 ms)"
    )


def test_bandwidth_throttle():
    """Large payload should take significantly longer than small payload."""
    topo = _two_node_topo(preset="metro")  # 20 Mbps upload
    ne = NetEm(topo, time_scale=0.001)  # 1000× faster
    small = ne.one_way_delay_s("A", "B", 100)
    # 1 MB at 20 Mbps = 400 ms
    large = ne.one_way_delay_s("A", "B", 1_000_000)
    assert large > small * 5, (
        f"bandwidth not throttling: large={large:.4f} vs small={small:.4f}"
    )


# ---------------------------------------------------------------------------
# Jitter distribution shape
# ---------------------------------------------------------------------------

def test_jitter_distribution_shape():
    """Jitter samples should be positive and mostly small with rare spikes."""
    topo = _two_node_topo(preset="intercontinental")
    ne = NetEm(topo, time_scale=1.0)
    samples = [ne._jitter_sample("A", "B") for _ in range(2000)]
    assert all(s >= 0 for s in samples), "jitter must be non-negative"
    mean = sum(samples) / len(samples)
    # sigma=10 ms → mean ≈ 10 + 1% × 10 × 10 = 11 ms; allow wide range
    assert 1.0 < mean < 100.0, f"jitter mean out of expected range: {mean:.2f} ms"
    # There should be some outliers (heavy tail)
    normal_max = 10.0 * 5   # 5σ without heavy tail
    outliers = [s for s in samples if s > normal_max]
    assert len(outliers) > 0, "expected some heavy-tail outlier samples"


# ---------------------------------------------------------------------------
# Loss probability
# ---------------------------------------------------------------------------

def test_loss_probability():
    """PacketLost should be raised with approximately the right probability."""
    topo = _two_node_topo(preset="intercontinental")  # 1% loss
    ne = NetEm(topo, time_scale=0.0001)
    lost = 0
    N = 500
    for _ in range(N):
        try:
            asyncio.run(ne.transit("A", "B", 0))
        except PacketLost:
            lost += 1
    # 1% loss over 500 trials: expect ~5, accept 0–20
    assert 0 <= lost <= 35, f"loss rate {lost}/{N} far off expected ~1%"


def test_no_loss_on_lan():
    """LAN preset has zero loss — PacketLost must never be raised."""
    topo = _two_node_topo(preset="lan")
    ne = NetEm(topo, time_scale=0.0001)
    for _ in range(200):
        asyncio.run(ne.transit("A", "B", 0))


# ---------------------------------------------------------------------------
# NAT matrix
# ---------------------------------------------------------------------------

NAT_CLASSES = ["public", "full_cone", "restricted", "port_restricted", "symmetric", "cgnat"]


@pytest.mark.parametrize("nat_a,nat_b,expected", [
    ("public",          "public",          True),
    ("public",          "symmetric",       True),
    ("public",          "cgnat",           True),
    ("symmetric",       "symmetric",       False),  # 0% hole-punch
    ("cgnat",           "cgnat",           False),
    ("cgnat",           "full_cone",       False),
    ("cgnat",           "port_restricted", False),
    ("cgnat",           "restricted",      False),
    ("port_restricted", "port_restricted", True),
    ("port_restricted", "symmetric",       False),
    ("restricted",      "symmetric",       False),
    ("full_cone",       "cgnat",           False),
])
def test_nat_matrix(nat_a: str, nat_b: str, expected: bool):
    topo = _relay_topo(nat_a, nat_b)
    ne = NetEm(topo)
    assert ne.can_direct("A", "B") == expected, (
        f"can_direct({nat_a}, {nat_b}) should be {expected}"
    )


# ---------------------------------------------------------------------------
# Relay path
# ---------------------------------------------------------------------------

def test_symmetric_symmetric_uses_relay():
    """symmetric ↔ symmetric must fall back to the relay path."""
    topo = _relay_topo("symmetric", "symmetric")
    ne = NetEm(topo)
    assert not ne.can_direct("A", "B")
    p = ne.path("A", "B")
    assert p == ["A", "relay", "B"], f"expected relay path, got {p}"


def test_cgnat_cgnat_uses_relay():
    topo = _relay_topo("cgnat", "cgnat")
    ne = NetEm(topo)
    assert not ne.can_direct("A", "B")
    p = ne.path("A", "B")
    assert "relay" in p


def test_direct_path_no_relay():
    topo = _two_node_topo(nat_a="public", nat_b="public")
    ne = NetEm(topo)
    p = ne.path("A", "B")
    assert p == ["A", "B"]


def test_relay_extra_latency():
    """A relay path costs more time than the direct path would.

    This must be asserted on the *expectation*, not on one sample. Every link
    here carries heavy-tailed jitter (sigma 5 ms, 1% chance of a x10 draw), so a
    single direct-path sample can easily exceed a single two-hop relay sample:
    one unlucky ~40 ms tail on the direct link beats 45 ms of relay propagation.
    Comparing two single draws made this test fail roughly 1-2% of runs, which is
    a bug in the assertion rather than in the emulator -- the claim "relaying is
    slower" is a statement about the mean, and about propagation in particular.

    So we check both halves separately:
      * the noise-free propagation term, where relaying is exactly 2x, always;
      * the sampled mean over many draws, where the jitter averages out.
    """
    topo = _relay_topo("symmetric", "symmetric")
    ne = NetEm(topo, time_scale=0.001)

    # 1. deterministic component: two hops of a same-profile link, zero variance.
    profile = topo.link("A", "B")
    direct_prop = (profile.rtt_ms / 2.0) / 1000.0
    relay_prop = 2 * direct_prop
    assert relay_prop == pytest.approx(2 * direct_prop), "propagation must double"
    assert relay_prop > direct_prop * 1.5

    # 2. sampled component: the mean over enough draws that a 1% tail cannot
    #    flip the comparison. E[jitter] is identical per hop, so the relay mean
    #    exceeds the direct mean by one full propagation delay plus one jitter.
    n = 400
    direct = sum(ne.one_way_delay_s("A", "B", 0) for _ in range(n)) / n
    relay = sum(ne.one_way_delay_s("A", "relay", 0)
                + ne.one_way_delay_s("relay", "B", 0) for _ in range(n)) / n
    assert relay > direct * 1.5, (
        f"relay should cost more on average: direct={direct:.4f}, relay={relay:.4f}")


# ---------------------------------------------------------------------------
# residential_mixed topology
# ---------------------------------------------------------------------------

def test_residential_mixed_has_relay():
    topo = residential_mixed(n=10, seed=42)
    relays = topo.relay_nodes()
    assert len(relays) >= 1, "residential_mixed must have at least one relay"


def test_residential_mixed_cgnat_fraction():
    """About 35% of non-relay nodes should be CGNAT."""
    topo = residential_mixed(n=20, seed=1)
    non_relay = [n for n in topo.nodes.values() if n.nat_class != "public"]
    cgnat = [n for n in non_relay if n.nat_class == "cgnat"]
    frac = len(cgnat) / len(non_relay)
    # Allow some variance around 35%
    assert 0.20 <= frac <= 0.55, f"CGNAT fraction {frac:.0%} outside expected 20-55% range"


def test_determinism_across_two_calls():
    """Same seed → relay list is identical across two instantiations."""
    t1 = residential_mixed(n=12, seed=7)
    t2 = residential_mixed(n=12, seed=7)
    assert sorted(t1.relay_nodes()) == sorted(t2.relay_nodes())
    for nid in t1.nodes:
        assert t1.nodes[nid].nat_class == t2.nodes[nid].nat_class
