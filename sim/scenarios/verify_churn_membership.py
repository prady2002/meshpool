"""Acceptance gate: SWIM gossip membership + sharded registry lookup scale.

Honest scope
------------
Measured under deterministic in-process emulation on one machine. This proves the
membership algorithm's direct probe, indirect probe and suspicion mechanics under
30% churn, and proves registry lookups use indexes/shards instead of enumerating
every node. It is not a 10k-node internet deployment; that remains Phase 8.

Exit 0 = PASS. Exit 1 = FAIL.
"""
from __future__ import annotations

import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from meshpool.cloud.swim import NAT_KEEPALIVE_S, SwimCluster, SwimConfig  # noqa: E402
from meshpool.common.models import Capability, RegisterRequest  # noqa: E402
from meshpool.controlplane.registry import EventBus, NodeRegistry, ShardedNodeMap  # noqa: E402

logging.getLogger("meshpool").setLevel(logging.WARNING)


class FakeStore:
    def __init__(self) -> None:
        self.nodes = {}
        self.tokens = {}

    def list_nodes(self):
        return list(self.nodes.values())

    def load_node_tokens(self):
        return dict(self.tokens)

    def put_node(self, node):
        self.nodes[node.id] = node

    def put_node_token(self, node_id, token, ts_ms):
        self.tokens[node_id] = token


def _req(i: int) -> RegisterRequest:
    return RegisterRequest(
        name=f"node-{i}",
        url=f"http://10.0.{i // 250}.{i % 250}:9001",
        capability=Capability(cpu_cores=1, ram_mb=1024),
    )


def main() -> int:
    failures: list[str] = []

    def check(cond: bool, name: str, detail: str = "") -> None:
        if cond:
            print(f"  PASS: {name}" + (f" ({detail})" if detail else ""))
        else:
            print(f"  FAIL: {name}" + (f" ({detail})" if detail else ""))
            failures.append(name)

    print("== test 1: SWIM non-vacuity guards ==")
    check(20.0 <= NAT_KEEPALIVE_S <= 25.0, "probe cadence matches NAT keepalive budget",
          f"{NAT_KEEPALIVE_S:.0f}s")
    guarded = SwimCluster(["a", "b", "c"], config=SwimConfig(indirect_checks=1, probe_count=2))
    guarded.block_one_way("a", "b")
    guarded.tick()
    check(guarded.memberships["a"].members["b"].state == "alive",
          "indirect probe rescues one bad direct link")

    no_indirect = SwimCluster(["a", "b", "c"], config=SwimConfig(indirect_checks=0, probe_count=2))
    no_indirect.block_one_way("a", "b")
    no_indirect.tick()
    check(no_indirect.memberships["a"].members["b"].state == "suspect",
          "without indirect probe the same link becomes suspect")

    suspicion = SwimCluster(["a", "b", "c"], config=SwimConfig(suspicion_ticks=2, probe_count=2))
    suspicion.churn_offline({"c"})
    suspicion.tick()
    check(suspicion.memberships["a"].members["c"].state == "suspect",
          "failed peer is suspected before death")
    check(suspicion.memberships["a"].members["c"].state != "dead",
          "suspicion prevents immediate eviction")

    print("\n== test 2: 30% churn converges within bound ==")
    total = 100
    offline = {f"node-{i}" for i in range(0, total, 3)}  # 34%, at least the requested 30%
    nodes = [f"node-{i}" for i in range(total)]
    cluster = SwimCluster(
        nodes,
        config=SwimConfig(indirect_checks=3, suspicion_ticks=3, probe_count=8),
        seed=42,
    )
    cluster.churn_offline(offline)
    bound = 20
    converged_at = None
    for tick in range(1, bound + 1):
        cluster.tick()
        if cluster.converged():
            converged_at = tick
            break
    check(converged_at is not None, "membership converges after heavy churn",
          f"offline={len(offline)}/{total} bound={bound} ticks actual={converged_at}")

    live = next(n for n in nodes if n not in offline)
    dead_seen = sum(
        1 for nid in offline
        if cluster.memberships[live].members[nid].state == "dead"
    )
    check(dead_seen == len(offline), "a live node sees every churned peer as dead",
          f"dead_seen={dead_seen}")

    print("\n== test 3: sharded registry uses indexed lookups instead of scans ==")
    reg = NodeRegistry(FakeStore(), node_timeout_s=30, events=EventBus())
    check(isinstance(reg._nodes, ShardedNodeMap), "registry storage is sharded")
    count = 10_000
    first_resp = None
    for i in range(count):
        resp = reg.register(_req(i), 3)
        if i == 1234:
            first_resp = resp
    check(len(reg.list()) == count, "registered scale population", f"nodes={count}")
    before = reg._nodes.values_calls
    assert first_resp is not None
    again = reg.register(_req(1234), 3)
    check(again.node_id == first_resp.node_id, "URL re-registration preserves identity")
    check(reg._nodes.values_calls == before,
          "URL re-registration used index, not all-node scan",
          f"values_calls_delta={reg._nodes.values_calls - before}")
    before = reg._nodes.values_calls
    for i in range(0, count, 500):
        node = reg.get(reg.register(_req(i), 3).node_id)
        check(node is not None, f"indexed get for node-{i}")
    check(reg._nodes.values_calls == before,
          "get/register hot path stayed index/shard based",
          f"values_calls_delta={reg._nodes.values_calls - before}")

    if failures:
        print("\nFAILED CHURN MEMBERSHIP CHECKS:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("\nALL CHURN MEMBERSHIP CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
