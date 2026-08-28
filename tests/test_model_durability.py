"""Tests for ``InferenceManager.durability`` — how many machines hold each layer.

On borrowed hardware the question a user actually has about a deployment is not
"is it up" (they can see that) but "what happens when these machines go away".
For a pooled model that reduces to one number: the least-held weight group. Lose
that many machines and the model cannot be re-formed at all, because the layers
to rebuild it from are gone. Every other stage can be replaced from a peer.

The arithmetic is easy to get subtly, silently wrong in the optimistic direction
— counting offline holders, counting a chunked group as held by anyone who has
*any* chunk, or averaging instead of taking the minimum. All three produce a
number that looks healthy and is not, on the one screen a user consults before
trusting the platform with something that matters. So each one has a test.
"""
from __future__ import annotations

import asyncio
import types

from meshpool.common.models import NodeStatus
from meshpool.inference.engine import InferenceManager


def _node(nid: str, online: bool = True):
    return types.SimpleNamespace(
        id=nid, url=f"http://{nid}:9",
        status=NodeStatus.ONLINE if online else NodeStatus.OFFLINE)


class _Registry:
    def __init__(self, nodes):
        self._nodes = {n.id: n for n in nodes}

    def get(self, nid):
        return self._nodes.get(nid)

    def online(self):
        return [n for n in self._nodes.values() if n.status is NodeStatus.ONLINE]


class _Storage:
    def __init__(self, metas, blobs=None):
        self._metas = metas
        self._blobs = blobs or {}

    def get_meta(self, blob_id):
        return self._metas.get(blob_id)

    async def get(self, blob_id):
        return self._blobs.get(blob_id)


class _Store:
    def __init__(self, dep):
        self._dep = dep

    def get_model(self, mid):
        return self._dep if (self._dep is not None and self._dep.id == mid) else None


def _meta(replicas, chunks=()):
    return types.SimpleNamespace(replicas=list(replicas), chunks=list(chunks))


def _mgr(dep, metas, nodes, manifest=None, blobs=None):
    mgr = InferenceManager(_Registry(nodes), store=_Store(dep), ledger=None,
                           events=None, transport=None)
    mgr.storage = _Storage(metas, blobs)
    if manifest is not None:
        mgr._manifest_cache[dep.weights_blob_id] = (manifest, None)
    return mgr


def _dep(mid="mdl-1", blob="manifest-blob"):
    return types.SimpleNamespace(id=mid, weights_blob_id=blob)


def _run(mgr, mid="mdl-1"):
    return asyncio.run(mgr.durability(mid))


# --------------------------------------------------------------------------- #
def test_unknown_deployment_is_none_not_an_empty_report():
    """A 404 and 'held by zero machines' are opposite answers; returning the
    latter for a typo'd id would be alarming and wrong."""
    mgr = _mgr(None, {}, [])
    assert asyncio.run(mgr.durability("nope")) is None


def test_builtin_model_reports_nothing_to_lose():
    """A built-in checkpoint ships with the agent, so every capable node already
    has the weights and no machine leaving can take them away."""
    dep = types.SimpleNamespace(id="mdl-1", weights_blob_id=None)
    mgr = _mgr(dep, {}, [_node("A")])
    d = _run(mgr)
    assert d["pooled"] is False
    assert d["min_holders"] is None and d["survives_machine_losses"] is None


def test_the_weakest_group_governs_not_the_average():
    """A pipeline needs every layer. Nine groups on five machines and one group
    on one machine is a model that dies with a single departure -- but the mean
    is 4.6, which reads as comfortable. This is the whole point of the metric."""
    metas = {"g0": _meta(["A", "B", "C", "D", "E"]), "g1": _meta(["A"])}
    mgr = _mgr(_dep(), metas, [_node(x) for x in "ABCDE"],
               manifest={"groups": {"layer0": "g0", "layer1": "g1"}})
    d = _run(mgr)
    assert d["min_holders"] == 1, d
    assert d["max_holders"] == 5
    assert d["survives_machine_losses"] == 0, (
        "a model with a single-homed layer must not report that it can lose a "
        f"machine: {d}")


def test_survivable_losses_is_one_less_than_the_weakest_group():
    metas = {"g0": _meta(["A", "B", "C"]), "g1": _meta(["A", "B"])}
    mgr = _mgr(_dep(), metas, [_node(x) for x in "ABC"],
               manifest={"groups": {"layer0": "g0", "layer1": "g1"}})
    d = _run(mgr)
    assert d["min_holders"] == 2 and d["survives_machine_losses"] == 1


def test_offline_holders_do_not_count():
    """The single most dangerous way to be wrong here: a machine that has been
    off for a week still counted as a copy. The number stays green right up
    until the model cannot be rebuilt."""
    metas = {"g0": _meta(["A", "B", "C"])}
    mgr = _mgr(_dep(), metas,
               [_node("A"), _node("B", online=False), _node("C", online=False)],
               manifest={"groups": {"layer0": "g0"}})
    d = _run(mgr)
    assert d["min_holders"] == 1, f"offline machines were counted as copies: {d}"
    assert d["per_group"][0]["nodes"] == ["A"]
    assert d["survives_machine_losses"] == 0


def test_unknown_node_ids_do_not_count():
    """A replica list can outlive the registry entry it names."""
    metas = {"g0": _meta(["A", "ghost"])}
    mgr = _mgr(_dep(), metas, [_node("A")],
               manifest={"groups": {"layer0": "g0"}})
    assert _run(mgr)["min_holders"] == 1


def test_a_chunked_group_is_only_as_held_as_its_scarcest_chunk():
    """Holding half of a layer is holding none of it. Counting anyone with any
    chunk would report 2 holders for a group that exactly one machine can
    actually serve."""
    metas = {
        "g0": _meta([], chunks=["c0", "c1"]),
        "c0": _meta(["A", "B"]),
        "c1": _meta(["A"]),
    }
    mgr = _mgr(_dep(), metas, [_node("A"), _node("B")],
               manifest={"groups": {"layer0": "g0"}})
    d = _run(mgr)
    assert d["min_holders"] == 1, f"a partial holder was counted as a copy: {d}"
    assert d["per_group"][0]["nodes"] == ["A"]


def test_a_node_holding_the_reassembled_whole_counts_for_a_chunked_group():
    """A node that pulled the group ends up with one object, not the chunks it
    was assembled from. It can serve the whole thing in one piece and must be
    counted -- otherwise swarm growth is invisible for exactly the large groups
    that get chunked, which are the ones where it matters most."""
    metas = {
        "g0": _meta(["Z"], chunks=["c0", "c1"]),
        "c0": _meta(["A"]),
        "c1": _meta(["A"]),
    }
    mgr = _mgr(_dep(), metas, [_node("A"), _node("Z")],
               manifest={"groups": {"layer0": "g0"}})
    d = _run(mgr)
    assert d["per_group"][0]["nodes"] == ["A", "Z"], d
    assert d["min_holders"] == 2


def test_a_group_nobody_holds_is_reported_as_zero_not_omitted():
    """Silently dropping an unheld group would make the minimum come from the
    groups that DO survive -- the report would look best precisely when the
    model is least recoverable."""
    metas = {"g0": _meta(["A"]), "g1": _meta([])}
    mgr = _mgr(_dep(), metas, [_node("A")],
               manifest={"groups": {"layer0": "g0", "layer1": "g1"}})
    d = _run(mgr)
    assert d["groups"] == 2
    assert d["min_holders"] == 0 and d["survives_machine_losses"] == 0
    assert {g["group"]: g["holders"] for g in d["per_group"]} == {"layer0": 1, "layer1": 0}


def test_a_group_whose_metadata_is_gone_is_reported_as_zero():
    mgr = _mgr(_dep(), {}, [_node("A")], manifest={"groups": {"layer0": "vanished"}})
    d = _run(mgr)
    assert d["min_holders"] == 0 and d["per_group"][0]["holders"] == 0


def test_distinct_holders_counts_machines_not_placements():
    """total_placements and distinct_holders answer different questions: how many
    copies exist, versus how many machines the model depends on."""
    metas = {"g0": _meta(["A", "B"]), "g1": _meta(["A", "B"]), "g2": _meta(["A", "C"])}
    mgr = _mgr(_dep(), metas, [_node(x) for x in "ABC"],
               manifest={"groups": {"l0": "g0", "l1": "g1", "l2": "g2"}})
    d = _run(mgr)
    assert d["total_placements"] == 6
    assert d["distinct_holders"] == 3


def test_groups_are_reported_in_a_stable_order():
    """The UI reads this straight through; an unstable order makes every poll
    look like a change."""
    metas = {f"g{i}": _meta(["A"]) for i in range(4)}
    manifest = {"groups": {"layer3": "g3", "layer1": "g1", "layer0": "g0", "layer2": "g2"}}
    mgr = _mgr(_dep(), metas, [_node("A")], manifest=manifest)
    names = [g["group"] for g in _run(mgr)["per_group"]]
    assert names == sorted(names) == ["layer0", "layer1", "layer2", "layer3"]
