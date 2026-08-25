"""Engine adapter seam: default MeshShard backend + documented Petals/exo stubs."""
from __future__ import annotations

import pytest

from meshpool.inference import model as M
from meshpool.inference.adapters import (DEFAULT_BACKEND, MeshShardBackend, get_backend)
from meshpool.controlplane.scheduler import topology_partition
from meshpool.common.models import Capability, DonationPolicy, Node


def _node(nid, ram, flops):
    return Node(id=nid, name=nid, url=f"http://{nid}",
                capability=Capability(cpu_cores=4, ram_mb=ram, flops_estimate=flops),
                policy=DonationPolicy(cpu_pct=100, ram_mb=ram))


def test_default_backend_is_meshshard_and_available():
    b = get_backend(DEFAULT_BACKEND)
    assert isinstance(b, MeshShardBackend)
    assert b.available() is True


def test_resolve_config_known_and_unknown():
    b = get_backend()
    assert b.resolve_config("nano-gpt").n_layer == 8
    assert b.resolve_config("does-not-exist") is None


def test_plan_partition_matches_scheduler():
    b = get_backend()
    nodes = [_node("n1", 8000, 60), _node("n2", 4000, 30), _node("n3", 2000, 15)]
    cfg = b.resolve_config("nano-gpt")
    got = b.plan_partition(nodes, cfg)
    want = topology_partition(nodes, cfg.n_layer, cfg.per_layer_mb)
    assert [(a.node_id, a.start_layer, a.end_layer) for a in got] == \
           [(a.node_id, a.start_layer, a.end_layer) for a in want]


def test_unknown_backend_raises():
    with pytest.raises(ValueError):
        get_backend("nope")


@pytest.mark.parametrize("name", ["petals", "exo"])
def test_documented_stubs_are_unavailable(name):
    b = get_backend(name)
    assert b.available() is False
    with pytest.raises(NotImplementedError):
        b.resolve_config("any")
