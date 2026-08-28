"""Weight-group redundancy: a layer must never be one machine away from lost.

Placement puts each weight group on the nodes that are *serving* it. That is
enough to answer requests and not enough to survive. When a machine leaves, the
model re-partitions onto survivors and pulls the layers it needs from whoever
still has them -- but if the departing machine was the last holder of a group,
there is nobody to pull from and the model can never be rebuilt. The weights are
simply gone, and no amount of spare capacity brings them back.

Measured on the sim mesh before this existed: after the three machines a
checkpoint was uploaded to had all left, every layer group was down to a single
holder. The model was serving perfectly and was one departure from unrecoverable,
with nothing anywhere reporting a problem.

The fix is warm residency: keep each group on at least ``target`` live machines,
whether or not they are serving it, by having under-held groups pulled onto nodes
that do not have them. It is deliberately peer-to-peer -- a redundancy mechanism
that pushes bytes through the coordinator would make durability maintenance the
one thing that cannot scale, which is precisely backwards.
"""
from __future__ import annotations

import asyncio
import types

import pytest

from meshpool.common.models import NodeStatus
from meshpool.common.transport import TransportError
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

    def schedulable_nodes(self):
        return self.online()


class _Storage:
    """Tracks holders the way StorageManager does, so redundancy is observable."""

    def __init__(self, metas):
        self._metas = metas

    def get_meta(self, blob_id):
        return self._metas.get(blob_id)

    async def get(self, blob_id):
        return b"bytes-of-" + blob_id.encode()

    def note_replica(self, blob_id, node_id):
        meta = self._metas.get(blob_id)
        if meta is None or node_id in meta.replicas:
            return False
        meta.replicas = [*meta.replicas, node_id]
        return True


class _Transport:
    def __init__(self, peer_ok=True, present=(), pull_fails=False):
        self.peer_ok = peer_ok
        self.present = set(present)
        self.pull_fails = pull_fails
        self.calls: list[tuple] = []

    def peer_pull_ok(self, node_id):
        return self.peer_ok

    async def call(self, node_id, path, payload, timeout=None):
        self.calls.append((node_id, path, payload.get("blob_id")))
        if path == "blob_has":
            return {"has": (node_id, payload["blob_id"]) in self.present}
        if path == "blob_pull":
            if self.pull_fails:
                raise TransportError("simulated pull failure")
            return {"stored": True}
        if path == "blob_put":
            return {"stored": True}
        raise TransportError(f"unexpected {path}")

    def of(self, path):
        return [(n, b) for n, p, b in self.calls if p == path]


class _Store:
    def __init__(self, dep):
        self._dep = dep

    def get_model(self, mid):
        return self._dep if self._dep is not None and self._dep.id == mid else None

    def put_model(self, dep):
        self._dep = dep


def _meta(replicas, chunks=()):
    return types.SimpleNamespace(replicas=list(replicas), chunks=list(chunks))


def _dep(mid="mdl-1"):
    return types.SimpleNamespace(id=mid, weights_blob_id="manifest")


def _mgr(nodes, metas, manifest, transport, dep=None):
    dep = dep or _dep()
    mgr = InferenceManager(_Registry(nodes), store=_Store(dep), ledger=None,
                           events=None, transport=transport)
    mgr.storage = _Storage(metas)
    mgr._manifest_cache["manifest"] = (manifest, None)
    return mgr, dep


# --------------------------------------------------------------------------- #
def test_a_single_homed_group_is_replicated_onto_another_machine():
    """The failure this exists to prevent: one machine holds the only copy of a
    layer, leaves, and the model becomes permanently unrecoverable while every
    health indicator was green."""
    metas = {"g0": _meta(["A"]), "g1": _meta(["A", "B"])}
    tr = _Transport()
    mgr, dep = _mgr([_node(x) for x in "ABC"], metas,
                    {"groups": {"l0": "g0", "l1": "g1"}}, tr)

    moved = asyncio.run(mgr.ensure_group_redundancy(dep, target=2))

    assert moved == 1, f"expected exactly the under-held group to move: {tr.calls}"
    assert len(metas["g0"].replicas) == 2, metas["g0"].replicas
    assert asyncio.run(mgr.durability("mdl-1"))["survives_machine_losses"] == 1


def test_it_does_not_touch_groups_that_are_already_safe():
    """Redundancy work is not free -- it is real bytes over somebody's home
    connection. Copying a group that already meets the target burns a
    contributor's bandwidth to achieve nothing."""
    metas = {"g0": _meta(["A", "B"]), "g1": _meta(["A", "B", "C"])}
    tr = _Transport()
    mgr, dep = _mgr([_node(x) for x in "ABCD"], metas,
                    {"groups": {"l0": "g0", "l1": "g1"}}, tr)

    moved = asyncio.run(mgr.ensure_group_redundancy(dep, target=2))

    assert moved == 0
    assert tr.of("blob_pull") == [] and tr.of("blob_put") == []


def test_redundancy_travels_peer_to_peer_not_through_the_coordinator():
    """A durability mechanism that relays every byte through one machine makes
    the coordinator the bottleneck for exactly the workload that never stops --
    every group, every departure, forever."""
    metas = {"g0": _meta(["A"])}
    tr = _Transport(peer_ok=True)
    mgr, dep = _mgr([_node(x) for x in "AB"], metas, {"groups": {"l0": "g0"}}, tr)

    asyncio.run(mgr.ensure_group_redundancy(dep, target=2))

    assert tr.of("blob_pull"), "no peer pull was attempted"
    assert tr.of("blob_put") == [], "the coordinator carried the weight bytes"


def test_it_copies_onto_a_machine_that_does_not_already_hold_the_group():
    """Two copies on one machine is one copy. Picking a target that already has
    it would satisfy the counter while changing nothing about survival."""
    metas = {"g0": _meta(["A"])}
    tr = _Transport()
    mgr, dep = _mgr([_node("A"), _node("B")], metas, {"groups": {"l0": "g0"}}, tr)

    asyncio.run(mgr.ensure_group_redundancy(dep, target=2))

    targets = [n for n, _ in tr.of("blob_pull")]
    assert targets == ["B"], f"copied onto a machine that already held it: {targets}"


def test_offline_machines_do_not_count_toward_the_target():
    """Counting a machine that has been off for a week is how this silently
    stops working: the target looks met and no copy is ever made."""
    metas = {"g0": _meta(["A", "gone"])}
    tr = _Transport()
    mgr, dep = _mgr([_node("A"), _node("gone", online=False), _node("B")],
                    metas, {"groups": {"l0": "g0"}}, tr)

    moved = asyncio.run(mgr.ensure_group_redundancy(dep, target=2))

    assert moved == 1, "an offline holder was counted as a live copy"
    assert [n for n, _ in tr.of("blob_pull")] == ["B"]


def test_it_asks_for_no_more_than_the_pool_can_hold():
    """A two-machine pool cannot store three copies. Retrying forever, or
    reporting failure every cycle, would turn a small healthy mesh into a
    permanent alarm."""
    metas = {"g0": _meta(["A"])}
    tr = _Transport()
    mgr, dep = _mgr([_node("A"), _node("B")], metas, {"groups": {"l0": "g0"}}, tr)

    moved = asyncio.run(mgr.ensure_group_redundancy(dep, target=5))

    assert moved == 1
    assert len(metas["g0"].replicas) == 2


def test_a_group_nobody_holds_cannot_be_recovered_and_is_not_faked():
    """There is no honest way to conjure bytes nobody has.

    It must not be recorded as placed, because a phantom holder makes the next
    repair skip it and turns a recoverable outage into a permanent one. And it
    must not be *attempted*: without the guard, every pass asks every machine in
    the pool whether it has a group that no machine has -- 1000 pointless round
    trips per pass, forever, for a group that is already lost. So the assertion
    is that nothing was asked at all, not merely that nothing was copied. That
    is also what makes this test capable of failing: delete the guard and every
    other assertion here still holds, because the pull correctly finds no source.

    The operator-facing log line for a lost layer is deliberately NOT asserted.
    This project's logger binds its stream at import and reaches neither caplog
    nor capfd, so such an assertion would test pytest's capture plumbing rather
    than the product. The machine-readable form of the same fact is durability()
    reporting holders=0 for the group, asserted in tests/test_model_durability.py.
    """
    metas = {"g0": _meta([])}
    tr = _Transport()
    mgr, dep = _mgr([_node("A"), _node("B")], metas, {"groups": {"l0": "g0"}}, tr)

    moved = asyncio.run(mgr.ensure_group_redundancy(dep, target=2))

    assert moved == 0
    assert metas["g0"].replicas == []
    assert tr.calls == [], (
        f"probed the pool for a group nobody holds: {tr.calls}")


def test_a_failed_pull_is_not_recorded_as_a_copy():
    """Recording a holder that does not hold it is worse than having no copy:
    the next repair believes the group is safe and skips it."""
    metas = {"g0": _meta(["A"])}
    tr = _Transport(pull_fails=True)
    mgr, dep = _mgr([_node("A"), _node("B")], metas, {"groups": {"l0": "g0"}}, tr)

    moved = asyncio.run(mgr.ensure_group_redundancy(dep, target=2))

    assert moved == 0
    assert metas["g0"].replicas == ["A"], metas["g0"].replicas


def test_a_node_that_already_has_the_bytes_is_recorded_without_transferring():
    """A node can hold a group the pool has forgotten about -- it served that
    layer under an older partition. Re-downloading it would be pure waste."""
    metas = {"g0": _meta(["A"])}
    tr = _Transport(present=[("B", "g0")])
    mgr, dep = _mgr([_node("A"), _node("B")], metas, {"groups": {"l0": "g0"}}, tr)

    moved = asyncio.run(mgr.ensure_group_redundancy(dep, target=2))

    assert moved == 1
    assert tr.of("blob_pull") == [], "re-fetched bytes the node already had"
    assert sorted(metas["g0"].replicas) == ["A", "B"]


def test_builtin_models_need_no_redundancy_work():
    """Every capable node already ships with the checkpoint."""
    tr = _Transport()
    dep = types.SimpleNamespace(id="mdl-1", weights_blob_id=None)
    mgr, _ = _mgr([_node("A")], {}, {}, tr, dep=dep)

    assert asyncio.run(mgr.ensure_group_redundancy(dep, target=3)) == 0
    assert tr.calls == []


def test_redundancy_never_raises_into_the_caller():
    """It runs on deploy and after every re-partition. A model that is serving
    correctly must not be failed by a bookkeeping problem in a background
    durability improvement."""
    class _Broken(_Storage):
        def get_meta(self, blob_id):
            raise RuntimeError("metadata store is down")

    tr = _Transport()
    mgr, dep = _mgr([_node("A"), _node("B")], {}, {"groups": {"l0": "g0"}}, tr)
    mgr.storage = _Broken({})

    assert asyncio.run(mgr.ensure_group_redundancy(dep, target=2)) == 0


@pytest.mark.parametrize("target", [0, 1])
def test_a_target_below_two_does_nothing(target):
    """Guards against a misconfiguration silently disabling durability while
    still appearing to run."""
    metas = {"g0": _meta(["A"])}
    tr = _Transport()
    mgr, dep = _mgr([_node("A"), _node("B")], metas, {"groups": {"l0": "g0"}}, tr)

    assert asyncio.run(mgr.ensure_group_redundancy(dep, target=target)) == 0
    assert tr.calls == []
