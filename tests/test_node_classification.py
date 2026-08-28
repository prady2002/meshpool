"""Automatic node classification (plan §3.6) and what it protects.

Every machine in the pool is not equally suited to every job, and the scheduler
currently cannot tell the difference: ``_usable_shard_nodes`` filters on advertised
service and memory alone. A battery-powered laptop behind CGNAT with a history of
dropping out therefore gets handed an inference pipeline stage purely for having
RAM -- and ``inference/engine.py`` already documents why that is expensive: a
pipeline is a chain, so its availability is the *product* of its stages'.

Two things are tested here, and they are deliberately separate:

  1. ``classify()`` is a pure function of measured signals, so its rules can be
     pinned exactly rather than inferred from scheduler behaviour.
  2. The scheduler actually *uses* it. A classification nothing consumes is a
     decorative field, so every rule below is paired with a placement assertion.

The signals are all pre-existing and all measured: ``network_class`` and
``battery_powered`` come from the real hardware probe, ``uptime_ratio_ema`` and
``latency_consistency_ema`` from the trust engine, ``samples`` from how much
evidence backs them.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from meshpool.common.models import (  # noqa: E402
    Capability, DonationPolicy, Node, NodeLoad, NodeStatus, TrustScore,
)
from meshpool.controlplane.nodeclass import (  # noqa: E402
    MIN_SAMPLES_FOR_PROMOTION, NodeClass, classify,
)


def _node(node_id: str, *, ram_mb: int = 32_000, cores: int = 8,
          network_class: str = "public", battery: bool | None = False,
          thermal: str = "ok", gpu: bool = False, vram_mb: int = 0) -> Node:
    return Node(
        id=node_id, name=node_id, url=f"http://10.0.0.1:9000/{node_id}",
        capability=Capability(
            cpu_cores=cores, ram_mb=ram_mb, vram_mb=vram_mb,
            gpu=gpu, gpu_names=["RTX 4090"] if gpu else [],
            network_class=network_class, battery_powered=battery,
            thermal_state=thermal, flops_estimate=float(cores),
        ),
        policy=DonationPolicy(cpu_pct=80, ram_mb=ram_mb),
        load=NodeLoad(), status=NodeStatus.ONLINE,
    )


def _trust(node_id: str, *, uptime: float, samples: int = 500,
           latency_consistency: float = 1.0) -> TrustScore:
    return TrustScore(node_id=node_id, uptime_ratio_ema=uptime, samples=samples,
                      latency_consistency_ema=latency_consistency)


# --------------------------------------------------------------------------- #
# The four classes
# --------------------------------------------------------------------------- #

def test_public_stable_server_is_an_anchor():
    n = _node("n1", ram_mb=128_000, cores=32, network_class="public")
    c = classify(n, _trust("n1", uptime=0.99))
    assert c.node_class is NodeClass.ANCHOR, c.reasons
    assert c.volatility < 0.1, c.reasons


def test_public_but_only_moderately_reliable_is_a_relay():
    """Relay duty needs reachability, not a perfect record: a relay that drops a
    connection costs one reconnect, whereas an anchor that drops loses metadata."""
    n = _node("n2", network_class="public")
    c = classify(n, _trust("n2", uptime=0.85))
    assert c.node_class is NodeClass.RELAY, c.reasons


def test_residential_gpu_box_with_decent_uptime_is_constellation_class():
    n = _node("n3", network_class="restricted", gpu=True, vram_mb=24_000)
    c = classify(n, _trust("n3", uptime=0.90))
    assert c.node_class is NodeClass.CONSTELLATION, c.reasons


def test_cgnat_laptop_on_battery_is_a_drifter():
    n = _node("n4", network_class="cgnat", battery=True)
    c = classify(n, _trust("n4", uptime=0.92))
    assert c.node_class is NodeClass.DRIFTER, c.reasons

    # Volatility is asserted against a matched control rather than a magic
    # threshold. This node has a *measured* 0.92 uptime, so demanding some
    # absolute "about to vanish" number would be asking the model to overrule its
    # own observation. What must be true is narrower and actually meaningful:
    # holding uptime fixed, portable power and unreachability each cost something.
    mains = classify(_node("n4", network_class="public", battery=False),
                     _trust("n4", uptime=0.92))
    assert c.volatility > mains.volatility, (mains.reasons, c.reasons)


def test_a_drifter_is_kept_out_of_pipeline_stages_and_the_others_are_not():
    """The classification has to change a decision, or it is a decorative field."""
    drifter = classify(_node("d", network_class="cgnat", battery=True),
                       _trust("d", uptime=0.92))
    assert not drifter.suitable_for_pipeline_stage(), drifter.reasons
    for network_class, uptime in (("public", 0.99), ("public", 0.85),
                                  ("restricted", 0.90)):
        ok = classify(_node("k", network_class=network_class), _trust("k", uptime=uptime))
        assert ok.suitable_for_pipeline_stage(), (network_class, ok.reasons)



# --------------------------------------------------------------------------- #
# The rules that are easy to get wrong
# --------------------------------------------------------------------------- #

def test_unreachable_node_can_never_be_anchor_or_relay_however_stable():
    """Both roles exist to be *dialled by peers*. A node behind CGNAT cannot be,
    no matter how good its record -- and its record is exactly what makes this
    tempting to get wrong, because uptime says nothing about addressability."""
    for network_class in ("cgnat", "symmetric"):
        n = _node("nx", network_class=network_class)
        c = classify(n, _trust("nx", uptime=1.0, samples=10_000))
        assert c.node_class not in (NodeClass.ANCHOR, NodeClass.RELAY), (
            f"{network_class} node classified {c.node_class}: {c.reasons}")


def test_a_brand_new_node_is_not_promoted_on_default_optimism():
    """Every trust EMA *defaults to 1.0*, so a node with no history looks perfect.

    Reading those defaults as evidence would hand a stranger's first-ever
    heartbeat an anchor role -- control-plane replicas and metadata -- on zero
    observations. Absence of evidence is not evidence of reliability, and the
    Capability model's own docstring makes the same argument about probe fields:
    a confident wrong number is worse than a known absence.
    """
    n = _node("new", network_class="public")
    fresh = classify(n, _trust("new", uptime=1.0, samples=0))
    assert fresh.node_class not in (NodeClass.ANCHOR, NodeClass.RELAY), fresh.reasons
    assert fresh.confidence < 0.5, fresh.reasons
    assert any("evidence" in r or "sample" in r for r in fresh.reasons), fresh.reasons

    # Non-vacuity: the *only* difference is accumulated evidence.
    proven = classify(n, _trust("new", uptime=1.0, samples=MIN_SAMPLES_FOR_PROMOTION))
    assert proven.node_class is NodeClass.ANCHOR, proven.reasons


def test_thermal_throttling_raises_volatility_without_changing_reachability():
    cool = classify(_node("t", network_class="restricted"), _trust("t", uptime=0.9))
    hot = classify(_node("t", network_class="restricted", thermal="throttling"),
                   _trust("t", uptime=0.9))
    assert hot.volatility > cool.volatility, (cool.reasons, hot.reasons)


def test_classification_always_explains_itself():
    """The console has to be able to say *why* a machine was or was not chosen."""
    for n, t in [
        (_node("a", network_class="public"), _trust("a", uptime=0.99)),
        (_node("b", network_class="cgnat", battery=True), _trust("b", uptime=0.5)),
    ]:
        c = classify(n, t)
        assert c.reasons, "a classification with no reasons cannot be explained"
        assert all(isinstance(r, str) and r for r in c.reasons)


def test_unknown_network_class_is_not_treated_as_public():
    """``network_class`` defaults to "unknown" for any agent that never probed.

    Guessing "public" would promote an unmeasured node into a role that requires
    peers to dial it, and the failure only shows up when a peer tries.
    """
    n = _node("u", network_class="unknown")
    c = classify(n, _trust("u", uptime=1.0, samples=10_000))
    assert c.node_class not in (NodeClass.ANCHOR, NodeClass.RELAY), c.reasons


def test_battery_powered_is_distinguished_from_unknown_battery_state():
    """``battery_powered`` is Optional[bool]: None means "not probed", not "desktop".

    Collapsing None into False silently treats every unprobed laptop as mains
    powered, which is the population most likely to vanish mid-pipeline.
    """
    desktop = classify(_node("d", network_class="restricted", battery=False),
                       _trust("d", uptime=0.9))
    unknown = classify(_node("d", network_class="restricted", battery=None),
                       _trust("d", uptime=0.9))
    assert unknown.volatility > desktop.volatility, (desktop.reasons, unknown.reasons)


# --------------------------------------------------------------------------- #
# The scheduler must actually consume it
# --------------------------------------------------------------------------- #

def _trust_lookup(mapping):
    return lambda node_id: mapping.get(node_id)


def test_a_drifter_is_dropped_from_placement_when_the_model_still_fits_without_it():
    from meshpool.controlplane.scheduler import stable_nodes_for_model

    solid = [_node(f"s{i}", ram_mb=32_000, network_class="restricted")
             for i in range(3)]
    laptop = _node("laptop", ram_mb=32_000, network_class="cgnat", battery=True)
    trust = _trust_lookup({
        **{f"s{i}": _trust(f"s{i}", uptime=0.95) for i in range(3)},
        "laptop": _trust("laptop", uptime=0.95),
    })

    kept = stable_nodes_for_model(solid + [laptop], trust, required_memory_mb=1_000)
    assert [n.id for n in kept] == ["s0", "s1", "s2"], [n.id for n in kept]


def test_a_drifter_is_kept_when_dropping_it_would_make_the_model_unhostable():
    """Refusing to place is worse than placing on a volatile node.

    The whole product promise is running models that fit on no single machine, so
    a purity rule that turns a working deployment into "insufficient pooled
    memory" is a regression dressed up as a safety feature. Preference, not
    prohibition.
    """
    from meshpool.controlplane.scheduler import stable_nodes_for_model

    solid = [_node("s0", ram_mb=8_000, network_class="restricted")]
    laptop = _node("laptop", ram_mb=32_000, network_class="cgnat", battery=True)
    trust = _trust_lookup({"s0": _trust("s0", uptime=0.95),
                           "laptop": _trust("laptop", uptime=0.95)})

    kept = stable_nodes_for_model([*solid, laptop], trust, required_memory_mb=30_000)
    assert {n.id for n in kept} == {"s0", "laptop"}, [n.id for n in kept]


def test_placement_is_unchanged_when_no_trust_data_is_available():
    """LAN mode and fresh coordinators have no trust history at all. Classification
    must degrade to the previous behaviour rather than emptying the pool."""
    from meshpool.controlplane.scheduler import stable_nodes_for_model

    nodes = [_node(f"n{i}", network_class="cgnat", battery=True) for i in range(3)]
    kept = stable_nodes_for_model(nodes, lambda _nid: None, required_memory_mb=1_000)
    assert [n.id for n in kept] == [n.id for n in nodes]


def test_topology_partition_avoids_a_drifter_when_it_can():
    """End-to-end: the property has to survive all the way to a layer assignment."""
    from meshpool.controlplane.scheduler import topology_partition

    solid = [_node(f"s{i}", ram_mb=16_000, network_class="restricted")
             for i in range(4)]
    laptop = _node("laptop", ram_mb=64_000, network_class="cgnat", battery=True)
    trust = _trust_lookup({
        **{f"s{i}": _trust(f"s{i}", uptime=0.95) for i in range(4)},
        "laptop": _trust("laptop", uptime=0.99),
    })

    before = topology_partition([*solid, laptop], n_layers=8, per_layer_mb=1_000)
    assert any(a.node_id == "laptop" for a in before), (
        "baseline: without classification the laptop is chosen, since it has the "
        "most memory -- this is the defect being fixed")

    after = topology_partition([*solid, laptop], n_layers=8, per_layer_mb=1_000,
                               trust_lookup=trust)
    assert after, "classification must not make the model unhostable"
    assert not any(a.node_id == "laptop" for a in after), [a.node_id for a in after]


def test_inference_engine_plumbs_trust_into_placement():
    """The wiring, not just the function.

    stable_nodes_for_model can be perfect and still change nothing if the engine
    never passes a lookup. This asserts on the argument that actually reaches the
    scheduler, because that is the only thing standing between a correct helper
    and a decorative one.
    """
    from unittest.mock import MagicMock

    from meshpool.inference.engine import InferenceManager

    eng = InferenceManager.__new__(InferenceManager)
    eng.trust = MagicMock()
    eng.trust.snapshot = lambda nid: _trust(nid, uptime=0.95)

    backend = MagicMock()
    backend.native_generate = True
    backend.plan_partition = MagicMock(return_value=[])

    cfg = MagicMock()
    cfg.layer_bytes = None
    cfg.n_layer = 8
    cfg.per_layer_mb = 1_000

    solid = [_node(f"s{i}", ram_mb=16_000, network_class="restricted")
             for i in range(4)]
    laptop = _node("laptop", ram_mb=64_000, network_class="cgnat", battery=True)

    eng._plan_partition(backend, [*solid, laptop], cfg)

    passed_nodes = backend.plan_partition.call_args.args[0]
    assert "laptop" not in {n.id for n in passed_nodes}, [n.id for n in passed_nodes]

    # Non-vacuity: with no trust handle the pool must reach the backend untouched,
    # which is what LAN mode and every existing test depend on.
    eng.trust = None
    backend.plan_partition.reset_mock()
    eng._plan_partition(backend, [*solid, laptop], cfg)
    untouched = backend.plan_partition.call_args.args[0]
    assert "laptop" in {n.id for n in untouched}, [n.id for n in untouched]
