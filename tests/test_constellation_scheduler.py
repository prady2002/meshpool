from __future__ import annotations

from meshpool.common.models import Capability, DonationPolicy, Node, NodeStatus
from meshpool.controlplane.scheduler import (
    build_constellation_index,
    constellation_nodes_for_model,
    topology_partition,
)


def _node(nid: str, ram_mb: int = 64_000, flops: float = 10.0) -> Node:
    return Node(
        id=nid,
        name=nid,
        url=f"http://{nid}",
        capability=Capability(cpu_cores=8, ram_mb=ram_mb, flops_estimate=flops),
        policy=DonationPolicy(cpu_pct=100, ram_mb=ram_mb),
    )


def test_constellation_clusters_by_rtt_not_name():
    nodes = [_node(n) for n in ["z-a", "m-a", "a-a", "z-b", "m-b", "a-b"]]
    metro_a = {"z-a", "m-a", "a-a"}
    metro_b = {"z-b", "m-b", "a-b"}

    def rtt(a: str, b: str) -> float:
        return 15.0 if ({a, b} <= metro_a or {a, b} <= metro_b) else 150.0

    index = build_constellation_index(nodes, rtt)
    assert {c.node_ids for c in index.constellations} == {
        tuple(sorted(metro_a)),
        tuple(sorted(metro_b)),
    }


def test_capacity_aware_lookup_skips_too_small_low_latency_group():
    small = [_node(f"s{i}", ram_mb=64_000) for i in range(3)]
    big = [_node(f"b{i}", ram_mb=256_000) for i in range(3)]
    small_ids = {n.id for n in small}
    big_ids = {n.id for n in big}

    def rtt(a: str, b: str) -> float:
        return 12.0 if ({a, b} <= small_ids or {a, b} <= big_ids) else 150.0

    chosen = constellation_nodes_for_model(
        small + big,
        rtt,
        required_memory_mb=550 * 1024,
        pipeline_depth=3,
    )
    assert {n.id for n in chosen} == big_ids


def test_latency_group_beats_faster_scattered_nodes_for_pipeline():
    slow = [_node(f"metro-{i}", ram_mb=128_000, flops=10.0) for i in range(3)]
    fast = [_node(f"fast-{i}", ram_mb=128_000, flops=1000.0) for i in range(3)]
    slow_ids = {n.id for n in slow}

    def rtt(a: str, b: str) -> float:
        return 15.0 if {a, b} <= slow_ids else 150.0

    picked = topology_partition(
        fast + slow,
        n_layers=6,
        per_layer_mb=50_000,
        rtt_predictor=rtt,
    )
    assert {a.node_id for a in picked} <= slow_ids
    assert sum(n.capability.flops_estimate for n in fast) > sum(n.capability.flops_estimate for n in slow)


def test_scattered_mesh_degrades_to_no_constellation():
    nodes = [_node("a"), _node("b")]
    index = build_constellation_index(nodes, lambda _a, _b: 150.0)
    assert index.constellations == []


def test_constellation_reforms_after_churn():
    nodes = [_node(f"m{i}") for i in range(4)]
    nodes[0].status = NodeStatus.OFFLINE
    index = build_constellation_index(nodes, lambda _a, _b: 12.0)
    assert len(index.constellations) == 1
    assert set(index.constellations[0].node_ids) == {"m1", "m2", "m3"}
