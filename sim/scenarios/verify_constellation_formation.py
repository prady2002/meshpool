"""Acceptance gate: constellation formation for low-latency pipeline placement.

Honest scope
------------
Measured under deterministic one-machine RTT emulation. This gate proves that
MeshPool discovers soft constellations from measured RTT/Vivaldi predictions,
advertises pooled RAM/VRAM capacity, and lets the scheduler prefer a low-RTT
metro over faster machines scattered across high-latency links. It is not a
public-internet measurement and reports no real-world throughput claim.

Exit 0 = PASS. Exit 1 = FAIL.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from meshpool.common.models import Capability, DonationPolicy, Node, NodeStatus  # noqa: E402
from meshpool.controlplane.scheduler import (  # noqa: E402
    CONSTELLATION_RTT_THRESHOLD_MS,
    build_constellation_index,
    constellation_nodes_for_model,
    topology_partition,
)


def node(nid: str, *, ram_mb: int = 128_000, vram_mb: int = 0, flops: float = 10.0) -> Node:
    return Node(
        id=nid,
        name=nid,
        url=f"http://{nid}",
        capability=Capability(
            cpu_cores=8,
            ram_mb=ram_mb,
            gpu=vram_mb > 0,
            vram_mb=vram_mb,
            flops_estimate=flops,
        ),
        policy=DonationPolicy(cpu_pct=100, ram_mb=ram_mb),
    )


def main() -> int:
    failures: list[str] = []

    def check(cond: bool, name: str, detail: str = "") -> None:
        if cond:
            print(f"  PASS: {name}" + (f" ({detail})" if detail else ""))
        else:
            print(f"  FAIL: {name}" + (f" ({detail})" if detail else ""))
            failures.append(name)

    print("== test 1: known metro topology clusters by RTT ==")
    metros = {
        "east": {f"east-{i}" for i in range(3)},
        "west": {f"west-{i}" for i in range(3)},
        "eu": {f"eu-{i}" for i in range(3)},
    }
    all_nodes = [node(nid) for group in metros.values() for nid in sorted(group)]

    def metro_rtt(a: str, b: str) -> float:
        for group in metros.values():
            if {a, b} <= group:
                return 15.0
        return 150.0

    index = build_constellation_index(all_nodes, metro_rtt)
    found = {frozenset(c.node_ids) for c in index.constellations}
    expected = {frozenset(group) for group in metros.values()}
    check(found == expected, "discovered clusters match injected ground truth",
          f"found={sorted(sorted(s) for s in found)}")

    too_strict = build_constellation_index(
        all_nodes, metro_rtt, rtt_threshold_ms=10.0)
    check(too_strict.constellations == [],
          "paired control: threshold below measured metro RTT forms no clusters")

    id_fake = build_constellation_index(
        all_nodes,
        lambda a, b: 15.0 if a.split("-")[-1] == b.split("-")[-1] else 150.0,
    )
    fake_found = {frozenset(c.node_ids) for c in id_fake.constellations}
    check(fake_found != expected,
          "paired control: node-id-shaped clusters differ from measured RTT truth")

    two_node = build_constellation_index(
        [node("pair-a"), node("pair-b")],
        lambda _a, _b: 12.0,
    )
    check(len(two_node.constellations) == 1
          and set(two_node.constellations[0].node_ids) == {"pair-a", "pair-b"},
          "two-node metro is a valid minimum-size constellation",
          f"found={list(two_node.constellations[0].node_ids) if two_node.constellations else []}")

    chain_nodes = [node("chain-a"), node("chain-b"), node("chain-c")]

    def chain_rtt(a: str, b: str) -> float:
        return 150.0 if {a, b} == {"chain-a", "chain-c"} else 12.0

    chain_index = build_constellation_index(chain_nodes, chain_rtt)
    check(all(
        max(chain_rtt(a, b) for i, a in enumerate(c.node_ids)
            for b in c.node_ids[i + 1:]) <= CONSTELLATION_RTT_THRESHOLD_MS
        for c in chain_index.constellations
    ), "every advertised constellation is pairwise within the RTT threshold",
       f"found={sorted(list(c.node_ids) for c in chain_index.constellations)}")
    check(not any(set(c.node_ids) == {"chain-a", "chain-b", "chain-c"}
                  for c in chain_index.constellations),
          "paired control: a low-latency chain is not treated as one metro")

    print("\n== test 2: capacity-aware pooled virtual machine ==")
    small = [node(f"small-{i}", ram_mb=64_000, vram_mb=8_000) for i in range(3)]
    big = [node(f"big-{i}", ram_mb=256_000, vram_mb=64_000) for i in range(3)]
    small_ids = {n.id for n in small}
    big_ids = {n.id for n in big}

    def capacity_rtt(a: str, b: str) -> float:
        if {a, b} <= small_ids:
            return 10.0
        if {a, b} <= big_ids:
            return 20.0
        return 150.0

    cap_index = build_constellation_index(small + big, capacity_rtt)
    small_const = next(c for c in cap_index.constellations if set(c.node_ids) == small_ids)
    big_const = next(c for c in cap_index.constellations if set(c.node_ids) == big_ids)
    check(small_const.pooled_ram_mb == 192_000 and small_const.pooled_vram_mb == 24_000,
          "constellation advertises pooled RAM and VRAM",
          f"ram={small_const.pooled_ram_mb} vram={small_const.pooled_vram_mb}")
    before_small_lookup = cap_index.lookup_scans
    low_need = cap_index.fitting(100_000)
    check(low_need is not None and set(low_need.node_ids) == small_ids,
          "capacity index returns the lowest-RTT fitting constellation",
          f"chosen={list(low_need.node_ids) if low_need else None}")
    check(cap_index.lookup_scans == before_small_lookup,
          "paired control: low-need capacity lookup used the bucket index")
    kimi_mb = 550 * 1024
    chosen = constellation_nodes_for_model(
        small + big,
        capacity_rtt,
        required_memory_mb=kimi_mb,
        pipeline_depth=3,
    )
    check({n.id for n in chosen} == big_ids,
          "Kimi-sized request skips low-latency group that lacks pooled capacity",
          f"small={small_const.pooled_usable_mb} big={big_const.pooled_usable_mb} need={kimi_mb}")
    check(not small_const.fits_memory(kimi_mb),
          "paired control: small constellation really cannot fit requested memory")

    print("\n== test 3: pipeline placement minimizes depth × RTT over raw FLOPS ==")
    slow = [node(f"metro-{i}", ram_mb=128_000, flops=10.0) for i in range(3)]
    fast = [node(f"fast-{i}", ram_mb=128_000, flops=1000.0) for i in range(3)]
    slow_ids = {n.id for n in slow}
    fast_ids = {n.id for n in fast}

    def pipeline_rtt(a: str, b: str) -> float:
        return 15.0 if {a, b} <= slow_ids else 150.0

    parts = topology_partition(
        fast + slow,
        n_layers=6,
        per_layer_mb=50_000,
        rtt_predictor=pipeline_rtt,
    )
    chosen_ids = {p.node_id for p in parts}
    check(chosen_ids <= slow_ids and bool(chosen_ids),
          "low-RTT slower metro beats fast scattered nodes",
          f"chosen={sorted(chosen_ids)}")
    check(sum(n.capability.flops_estimate for n in fast) > sum(n.capability.flops_estimate for n in slow),
          "paired control: scattered candidates are genuinely faster by FLOPS",
          f"fast={sum(n.capability.flops_estimate for n in fast)} slow={sum(n.capability.flops_estimate for n in slow)}")
    check((len(parts) - 1) * 15.0 < (3 - 1) * 150.0,
          "pipeline_depth × RTT discriminator favors the metro placement")


    dense = [node(f"dense-{i}", ram_mb=128_000, flops=1000.0) for i in range(4)]
    bursty = [node(f"bursty-{i}", ram_mb=128_000, flops=10.0) for i in range(4)]
    dense_ids = {n.id for n in dense}
    bursty_ids = {n.id for n in bursty}

    def avg_rtt(a: str, b: str) -> float:
        if {a, b} <= dense_ids:
            return 20.0
        if {a, b} <= bursty_ids:
            return 24.0 if {a, b} == {"bursty-0", "bursty-1"} else 8.0
        return 150.0

    avg_choice = constellation_nodes_for_model(
        dense + bursty,
        avg_rtt,
        required_memory_mb=300_000,
        pipeline_depth=4,
    )
    bursty_pairs = [
        avg_rtt(a, b)
        for i, a in enumerate(sorted(bursty_ids))
        for b in sorted(bursty_ids)[i + 1:]
    ]
    check({n.id for n in avg_choice} == bursty_ids,
          "scheduler minimizes average pipeline RTT before FLOPS or max RTT",
          f"chosen={sorted(n.id for n in avg_choice)}")
    check(sum(bursty_pairs) / len(bursty_pairs) < 20.0
          and max(bursty_pairs) > 20.0
          and sum(n.capability.flops_estimate for n in bursty) < sum(n.capability.flops_estimate for n in dense),
          "paired control: chosen group has lower average RTT, higher max RTT, and lower FLOPS")

    print("\n== test 4: degenerate meshes degrade sanely ==")
    two_scattered = build_constellation_index(
        [node("solo-a"), node("solo-b")],
        lambda _a, _b: 150.0,
    )
    check(two_scattered.constellations == [],
          "2-node scattered mesh forms no constellation")
    all_scattered = build_constellation_index(
        [node(f"scattered-{i}") for i in range(5)],
        lambda _a, _b: 150.0,
    )
    check(all_scattered.constellations == [],
          "all-scattered mesh forms no constellation")
    single_metro_ids = {f"one-{i}" for i in range(4)}
    single_metro = build_constellation_index(
        [node(nid) for nid in single_metro_ids],
        lambda _a, _b: 14.0,
    )
    check(len(single_metro.constellations) == 1
          and set(single_metro.constellations[0].node_ids) == single_metro_ids,
          "single metro forms one constellation")
    churned_nodes = [node(nid) for nid in single_metro_ids]
    churned_nodes[0].status = NodeStatus.OFFLINE
    churned = build_constellation_index(churned_nodes, lambda _a, _b: 14.0)
    check(len(churned.constellations) == 1
          and churned_nodes[0].id not in churned.constellations[0].node_ids,
          "soft constellation reforms after churn removes an offline node",
          f"members={list(churned.constellations[0].node_ids) if churned.constellations else []}")

    print("\n== test 5: pre-clustered capacity lookup is O(1) ==")
    before = cap_index.lookup_scans
    fit = cap_index.fitting(kimi_mb)
    check(fit is not None and set(fit.node_ids) == big_ids,
          "capacity lookup returns fitting constellation")
    check(cap_index.lookup_scans == before,
          "capacity lookup used bucket index, not a constellation scan",
          f"lookup_scans_delta={cap_index.lookup_scans - before}")
    check(CONSTELLATION_RTT_THRESHOLD_MS == 25.0,
          "formation threshold is the plan's <25 ms low-latency neighbourhood")

    if failures:
        print("\nFAILED CONSTELLATION CHECKS:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("\nALL CONSTELLATION CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
