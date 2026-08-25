"""Placement algorithms: topology partition (sharding), best-fit, bin-pack."""
from __future__ import annotations

from meshpool.common.models import Capability, DonationPolicy, Node, NodeLoad
from meshpool.controlplane.scheduler import (best_fit_node, bin_pack_replicas,
                                             topology_partition,
                                             usable_layer_memory_mb)


def mknode(nid, cores, ram, flops, cpu_pct=100, url=None):
    return Node(
        id=nid, name=nid, url=url or f"http://{nid}",
        capability=Capability(cpu_cores=cores, ram_mb=ram, flops_estimate=flops),
        policy=DonationPolicy(cpu_pct=cpu_pct, ram_mb=ram),
        load=NodeLoad(),
    )


def test_topology_partition_covers_all_layers_contiguously():
    nodes = [mknode("n1", 8, 8000, 60), mknode("n2", 4, 4000, 30),
             mknode("n3", 2, 2000, 15)]
    # caps @ per_layer 1000, headroom .8 => 6/3/1 = 10 total; no single node holds 8
    asg = topology_partition(nodes, n_layers=8, per_layer_mb=1000)
    assert asg, "expected a valid partition"
    asg = sorted(asg, key=lambda a: a.start_layer)
    assert asg[0].start_layer == 0
    assert asg[-1].end_layer == 8
    for a, b in zip(asg, asg[1:]):
        assert a.end_layer == b.start_layer  # contiguous, no gaps/overlap
    assert asg[0].is_first and asg[-1].is_last
    # no single stage holds the whole model (proves "too big for one machine")
    assert all((a.end_layer - a.start_layer) < 8 for a in asg)


def test_topology_partition_refuses_when_pool_too_small():
    nodes = [mknode("n1", 2, 2000, 15)]  # cap 1 layer only
    assert topology_partition(nodes, n_layers=8, per_layer_mb=1000) == []


def test_best_fit_prefers_tightest_node():
    big = mknode("big", 16, 8000, 100, cpu_pct=100)
    small = mknode("small", 8, 4000, 50, cpu_pct=60)
    chosen = best_fit_node([big, small], cpu_pct=50, ram_mb=1000)
    assert chosen.id == "small"  # tighter fit keeps big node free for big work


def test_bin_pack_spreads_then_packs():
    nodes = [mknode("a", 16, 8000, 100), mknode("b", 16, 8000, 100)]
    placements = bin_pack_replicas(nodes, count=3, cpu_pct=40, ram_mb=1000)
    assert len(placements) == 3
    assert len({p.id for p in placements}) == 2  # anti-affinity spread first

def test_integrated_gpu_does_not_shrink_a_large_ram_node():
    """A small VRAM carve-out must never make a big-RAM machine look tiny.

    Regression for a real defect: the probe on a consumer laptop reports
    ``gpu=True, vram_mb=512`` for an integrated GPU. The planner used VRAM
    *instead of* RAM, so a 32 GB machine was scored at 512 MB => 0 layers, and
    ``POST /api/v1/models`` failed with "insufficient pooled memory: pooled
    capacity 0 layers" on hardware with plenty of memory. Layers can live in
    system RAM, so the bound is the larger of the two.
    """
    igpu = Node(
        id="igpu", name="igpu", url="http://igpu",
        capability=Capability(cpu_cores=12, ram_mb=32029, flops_estimate=300,
                              gpu=True, vram_mb=512),
        policy=DonationPolicy(cpu_pct=100, ram_mb=32029),
        load=NodeLoad(),
    )
    assert usable_layer_memory_mb(igpu) == 32029

    # ...and it must actually be able to hold the model on its own.
    asg = topology_partition([igpu], n_layers=8, per_layer_mb=1000)
    assert asg, "a 32 GB node with an integrated GPU must be able to host 8 layers"
    assert asg[0].start_layer == 0 and asg[-1].end_layer == 8


def test_discrete_gpu_with_more_vram_than_ram_uses_vram():
    """The larger-of-the-two rule must still prefer VRAM when VRAM is larger.

    Note: this one passes under both the old and the new rule by construction —
    it is not a regression guard for the iGPU bug above, it is a guard against
    *over*-correcting it (e.g. "just always use RAM"), which would throw away a
    48 GB accelerator's capacity on a machine with 8 GB of system RAM.
    """
    dgpu = Node(
        id="dgpu", name="dgpu", url="http://dgpu",
        capability=Capability(cpu_cores=8, ram_mb=8000, flops_estimate=900,
                              gpu=True, vram_mb=48000),
        policy=DonationPolicy(cpu_pct=100, ram_mb=0),
        load=NodeLoad(),
    )
    assert usable_layer_memory_mb(dgpu) == 48000


def test_donation_ram_cap_still_bounds_a_gpu_node():
    """The rule raises the ceiling; it must not defeat the operator's cap."""
    capped = Node(
        id="capped", name="capped", url="http://capped",
        capability=Capability(cpu_cores=12, ram_mb=32029, flops_estimate=300,
                              gpu=True, vram_mb=512),
        policy=DonationPolicy(cpu_pct=100, ram_mb=1000),  # operator donates 1 GB
        load=NodeLoad(),
    )
    # 1000 MB * 0.8 // 1000 = 0 layers -> cannot host the model alone
    assert topology_partition([capped], n_layers=8, per_layer_mb=1000) == []


def _narrow(node, services):
    node.capability.services = services
    return node


def test_shard_planner_skips_nodes_that_cannot_serve_shards():
    """A node that 501s on /shard/load must never appear in a layer partition.

    Regression for a real defect found by probing the planner directly. ``meshd``
    (the Go agent) implements work units and blobs but returns 501 for
    ``/shard/load``. The planner filtered on memory alone, so it happily gave a
    meshd node the *last* stage of the pipeline -- the LM head -- and the deploy
    died at load time, after the plan had been committed.
    """
    py = mknode("python-node", 8, 16000, 10)
    go = _narrow(mknode("meshd-node", 8, 32000, 10), ["batch", "storage"])
    asg = topology_partition([py, go], n_layers=8, per_layer_mb=1000)
    assert asg, "the python node alone can hold all 8 layers"
    assert {a.node_id for a in asg} == {"python-node"}


def test_adding_a_shard_incapable_node_cannot_break_a_working_pool():
    """The sharp edge of the same bug: *adding capacity broke a deployment*.

    ``python-node`` alone hosts the model. Before the fix, adding a bigger
    meshd node made the planner hand it half the layers, so a pool that worked
    stopped working the moment it grew. Capacity must be monotonic.
    """
    py = mknode("python-node", 8, 16000, 10)
    alone = topology_partition([py], n_layers=8, per_layer_mb=1000)
    go = _narrow(mknode("meshd-node", 8, 32000, 10), ["batch", "storage"])
    grown = topology_partition([py, go], n_layers=8, per_layer_mb=1000)
    assert grown, "adding a node must never turn a servable pool into an unservable one"
    covered = sum(a.end_layer - a.start_layer for a in grown)
    assert covered == 8
    assert [a.node_id for a in alone] == [a.node_id for a in grown]


def test_shard_incapable_node_is_still_chosen_once_it_advertises_shards():
    """Positive control for the two tests above.

    Without this, they could pass for the wrong reason -- e.g. if the planner
    had started rejecting the node over memory, or rejecting every second node.
    The *only* thing that changes here is the advertisement.
    """
    py = mknode("python-node", 8, 16000, 10)
    wide = _narrow(mknode("meshd-node", 8, 32000, 10),
                   ["batch", "storage", "hosting", "shards"])
    asg = topology_partition([py, wide], n_layers=8, per_layer_mb=1000)
    assert "meshd-node" in {a.node_id for a in asg}


def test_replica_planner_skips_nodes_that_cannot_host():
    """Same defect on the hosting path: /start_replica is 501 on meshd."""
    py = mknode("a", 16, 8000, 100)
    go = _narrow(mknode("b", 16, 8000, 100), ["batch", "storage"])
    placements = bin_pack_replicas([py, go], count=3, cpu_pct=40, ram_mb=1000)
    assert placements, "the capable node must still take replicas"
    assert {p.id for p in placements} == {"a"}
    # positive control: the identical node is chosen once it advertises hosting
    go.capability.services = ["batch", "storage", "hosting"]
    both = bin_pack_replicas([py, go], count=3, cpu_pct=40, ram_mb=1000)
    assert {p.id for p in both} == {"a", "b"}


def test_nodes_registered_before_services_existed_keep_every_service():
    """Backwards compatibility is load-bearing: the field can only ever narrow.

    An empty list means "did not narrow", not "can do nothing" -- otherwise a
    single old agent build would silently drop out of every placement decision.
    """
    legacy = mknode("legacy", 8, 16000, 10)
    legacy.capability.services = []
    assert legacy.provides("shards") and legacy.provides("hosting")
    assert topology_partition([legacy], n_layers=8, per_layer_mb=1000)
    assert bin_pack_replicas([legacy], count=1, cpu_pct=40, ram_mb=1000)


def test_null_list_fields_do_not_brick_registration():
    """A JSON ``null`` for a list capability must not 422 the node out of the pool.

    Regression for a real fleet-bricking bug, found when a freshly built ``meshd``
    could not register at all: Go marshals a nil slice as ``null``, not ``[]``, so
    an agent that leaves ``services`` unset sends ``"services": null``. Pydantic
    rejects null for ``list[str]``, registration answers 422, and the agent retries
    forever with no path to recovery. Adding a list field to Capability would
    otherwise be enough to lock out every older agent binary mid-rollout.

    ``ollama_models`` had this hazard from the day it was added; it escaped notice
    only because the Python agent always sets it.
    """
    from meshpool.common.models import Capability, Node

    cap = Capability(**{"cpu_cores": 8, "services": None,
                        "runtimes": None, "ollama_models": None})
    assert cap.services == ["batch", "hosting", "shards", "storage"]
    assert cap.runtimes == ["builtin", "wasm"]
    assert cap.ollama_models == []

    # ...through the real registration entry point, which is a nested dict.
    node = Node(**{"id": "n1", "name": "n1", "url": "http://127.0.0.1:0/n1",
                   "capability": {"cpu_cores": 8, "services": None}})
    assert node.provides("shards") and node.provides("hosting")

    # ...and straight from the wire.
    assert Capability.model_validate_json(
        '{"cpu_cores":4,"services":null}').services == [
            "batch", "hosting", "shards", "storage"]

    # Non-vacuity: an explicit list must still be honoured, or the coercion above
    # could be "ignore this field entirely", which would silently undo the filter.
    narrowed = Capability(**{"cpu_cores": 8, "services": ["batch", "storage"]})
    assert narrowed.services == ["batch", "storage"]
    narrow_node = Node(id="n2", name="n2", url="http://127.0.0.1:0/n2",
                       capability=narrowed)
    assert not narrow_node.provides("shards")
    assert narrow_node.provides("batch")

def test_every_list_capability_field_tolerates_null():
    """Derived, not hand-listed: adding a list field must not re-brick registration.

    The previous guard named three fields explicitly, so the *next* list field
    added to Capability would silently fall outside it and lock every agent
    build that leaves the slice nil out of the pool. This iterates the model, so
    a new field is covered the moment it exists.
    """
    import json

    from meshpool.common.models import Capability

    list_fields = Capability._null_tolerant_list_fields()
    assert "services" in list_fields and "gpu_names" in list_fields, list_fields

    for name in list_fields:
        payload = json.dumps({"cpu_cores": 4, name: None})
        cap = Capability.model_validate_json(payload)
        value = getattr(cap, name)
        assert isinstance(value, list), f"{name}: null produced {value!r}, not a list"

    # All of them nil at once, which is what a freshly written agent in a
    # language with nil slices actually sends.
    every = json.dumps({"cpu_cores": 4, **{n: None for n in list_fields}})
    cap = Capability.model_validate_json(every)
    for name in list_fields:
        assert isinstance(getattr(cap, name), list)
