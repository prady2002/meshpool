from __future__ import annotations

from meshpool.cloud.swim import SwimCluster, SwimConfig
from meshpool.common.models import Capability, RegisterRequest
from meshpool.controlplane.registry import EventBus, NodeRegistry, ShardedNodeMap


def _req(name: str, url: str) -> RegisterRequest:
    return RegisterRequest(name=name, url=url, capability=Capability(cpu_cores=1, ram_mb=1024))


def test_swim_indirect_probe_prevents_false_suspicion_on_one_bad_link():
    cluster = SwimCluster(["a", "b", "c"], config=SwimConfig(indirect_checks=1, probe_count=2))
    cluster.block_one_way("a", "b")
    cluster.tick()
    assert cluster.memberships["a"].members["b"].state == "alive"


def test_swim_suspicion_precedes_death_under_churn():
    cluster = SwimCluster(["a", "b", "c"], config=SwimConfig(suspicion_ticks=2, probe_count=2))
    cluster.churn_offline({"c"})
    cluster.tick()
    assert cluster.memberships["a"].members["c"].state == "suspect"
    cluster.tick()
    assert cluster.memberships["a"].members["c"].state == "suspect"
    cluster.tick()
    assert cluster.memberships["a"].members["c"].state == "dead"


def test_registry_reregistration_uses_indexes_not_node_scan(tmp_store):
    reg = NodeRegistry(tmp_store, node_timeout_s=10, events=EventBus())
    assert isinstance(reg._nodes, ShardedNodeMap)
    first = reg.register(_req("n0", "http://n0:9001"), 3)
    before = reg._nodes.values_calls
    again = reg.register(_req("n0", "http://n0:9001"), 3)
    assert again.node_id == first.node_id
    assert reg._nodes.values_calls == before


def test_registry_online_uses_status_index(tmp_store):
    reg = NodeRegistry(tmp_store, node_timeout_s=10, events=EventBus())
    for i in range(20):
        reg.register(_req(f"n{i}", f"http://n{i}:9001"), 3)
    before = reg._nodes.values_calls
    online = reg.online()
    assert len(online) == 20
    assert reg._nodes.values_calls == before
