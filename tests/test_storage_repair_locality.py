"""Does the storage repair loop move weight bytes through the control plane?

The Weight Swarm gate asserts ``weight_group_pushes == 0`` and reports "zero
coordinator relays". That counter is incremented in exactly one place: the
inference engine's fallback path. Nothing stops a *different* component from
moving the same bytes through the same coordinator, and if one does, the gate
cannot see it and the headline claim is not true.

StorageManager._reconcile re-replicates any under-replicated blob, and every
weight group is a blob. So the question these tests answer empirically rather
than by reading is: when a machine holding a weight group leaves, do the
replacement bytes travel peer to peer, or through the control plane's memory?

The distinction does not matter at 6 MB. It decides whether the product exists
at 600 GB: a coordinator that must hold each weight group in RAM, base64 it
(+33%), and push it out again is a hard bandwidth and memory bottleneck that
every node departure in the pool queues behind -- the exact centralisation the
peer-to-peer placement path was built to avoid.

The fakes are imported from tests/test_storage.py rather than rewritten, because
those verify digests the way a real agent does. A fake that accepts bytes it was
never sent cannot tell you who sent them.
"""
from __future__ import annotations

import asyncio
import base64
import hashlib

import pytest

from test_storage import (FakeEvents, FakeRegistry, FakeStore, FakeTransport,
                          _sha)
from meshpool.common.transport import TransportError
from meshpool.storage.engine import StorageManager

WEIGHTS = b"pretend-this-is-a-layer-of-kimi-k2\n" * 512


class RelayCountingTransport(FakeTransport):
    """FakeTransport that also records what the *coordinator itself* carried.

    ``blob_get`` is the control plane pulling an object into its own memory.
    ``blob_put`` carrying a ``data`` payload is it pushing those bytes back out.
    A genuine peer-to-peer transfer would be neither: it would name a source and
    let the receiving node do the fetching, so only ids cross the control plane.
    """

    def __init__(self, node_ids, unreachable=None):
        super().__init__(node_ids, unreachable=unreachable)
        self.relayed_in = 0
        self.relayed_out = 0
        self.peer_pulls = 0

    async def call(self, node_id, method, payload, timeout=None):
        if method == "blob_pull" and node_id in self.disks:
            # A real plan carries node *URLs* and the receiving agent fetches from
            # them over HTTP. FakeRegistry mints urls as http://127.0.0.1:0/<node_id>,
            # so the trailing segment resolves a url back to the disk it names.
            for url in payload.get("sources") or []:
                src = str(url).rstrip("/").rsplit("/", 1)[-1]
                raw = self.disks.get(src, {}).get(payload["blob_id"])
                if raw is None or src in self.unreachable:
                    continue
                if hashlib.sha256(raw).hexdigest() != payload["blob_id"]:
                    continue          # an agent verifies before it stores
                self.calls.append((node_id, method))
                self.disks[node_id][payload["blob_id"]] = raw
                self.peer_pulls += 1
                return {"ok": True, "bytes": len(raw)}
            self.calls.append((node_id, method))
            raise TransportError(f"no reachable source for {payload['blob_id'][:12]}")
        res = await super().call(node_id, method, payload, timeout)
        if method == "blob_get" and res.get("found"):
            self.relayed_in += len(base64.b64decode(res["data"]))
        elif method == "blob_put" and payload.get("data"):
            self.relayed_out += len(base64.b64decode(payload["data"]))
        return res

    def reset_counters(self):
        self.relayed_in = self.relayed_out = self.peer_pulls = 0
        self.calls.clear()


def _manager(node_ids, unreachable=None):
    reg = FakeRegistry(node_ids)
    store = FakeStore()
    transport = RelayCountingTransport(node_ids, unreachable=unreachable)
    mgr = StorageManager(reg, store, None, FakeEvents(), transport)
    return mgr, reg, store, transport


def _seed(mgr, transport, data=WEIGHTS, replication=2):
    meta = asyncio.run(mgr.put(data, name="g-layer-0", replication=replication))
    transport.reset_counters()
    return meta


def test_repair_after_a_departure_does_not_route_bytes_through_the_coordinator():
    """The claim the Weight Swarm gate makes, tested where that gate cannot see it.

    A weight group sits on two machines. One leaves. Something must place a
    second copy, and the only question that matters at scale is whether the
    replacement bytes pass through the control plane on the way there.
    """
    mgr, reg, store, tr = _manager(["A", "B", "C"])
    meta = _seed(mgr, tr)
    victim = meta.replicas[0]

    reg.kill(victim)
    tr.unreachable.add(victim)
    asyncio.run(mgr._reconcile())

    after = store.get_blob(meta.id)
    assert len(after.replicas) == 2, (
        f"the group was left on {after.replicas}; one more departure loses the "
        f"layer, and a layer cannot be rebuilt from the other layers")
    assert tr.relayed_in == 0 and tr.relayed_out == 0, (
        f"the control plane carried {tr.relayed_in} bytes in and "
        f"{tr.relayed_out} bytes out to restore a single replica. At nano-gpt "
        f"scale that is invisible. For a 600 GB model it means the coordinator "
        f"holds each layer group in RAM, base64-encodes it (+33%), and pushes it "
        f"out again, serialising every node departure in the pool behind one "
        f"uplink. Calls: {tr.calls}")
    assert (mgr.repair_peer_pulls, mgr.repair_relays) == (1, 0), (
        f"the manager's own accounting disagrees with the wire: "
        f"peer_pulls={mgr.repair_peer_pulls} relays={mgr.repair_relays}. A gate "
        f"reads these counters, so a counter that does not match what actually "
        f"crossed the transport is worse than no counter at all")
    assert mgr.repair_relayed_bytes == 0


def test_when_no_peer_is_reachable_the_relay_still_saves_the_data():
    """The fallback must remain a real fallback, not decoration.

    Peer-to-peer is an optimisation of the *route*. It must never become a reason
    to leave an object under-replicated: a pool that refuses to repair unless it
    can do so cheaply is less durable than one that repairs expensively.
    """
    mgr, reg, store, tr = _manager(["A", "B", "C"])
    meta = _seed(mgr, tr)
    victim, survivor = meta.replicas[0], meta.replicas[1]

    reg.kill(victim)
    tr.unreachable.add(victim)
    # Strip the surviving holder's advertised URL: reachable by the coordinator,
    # not addressable by a peer. This is the CGNAT case, and section 2.5 budgets
    # for 25-40% of the pool being in it.
    reg.nodes[survivor].url = ""

    asyncio.run(mgr._reconcile())

    after = store.get_blob(meta.id)
    live = [r for r in after.replicas if r in reg.online_ids]
    assert len(live) == 2, (
        f"gave up on an object it could still have repaired: {after.replicas}")
    assert mgr.repair_relays == 1 and mgr.repair_peer_pulls == 0
    assert mgr.repair_relayed_bytes == len(WEIGHTS)
    for nid in live:
        assert tr.disks[nid][meta.id] == WEIGHTS


def test_a_peer_pull_that_fails_falls_back_rather_than_losing_the_replica():
    """A refused pull is not a lost object.

    The peer path can fail for reasons the coordinator cannot predict -- the
    source went away between the plan being built and the pull being attempted,
    the target's disk is full, a firewall ate it. Every one of those must degrade
    to the slow route, never to silence.
    """
    mgr, reg, store, tr = _manager(["A", "B", "C"])
    meta = _seed(mgr, tr)
    victim = meta.replicas[0]
    reg.kill(victim)
    tr.unreachable.add(victim)

    real_call = tr.call

    async def refusing_call(node_id, method, payload, timeout=None):
        if method == "blob_pull":
            tr.calls.append((node_id, method))
            raise TransportError("peer refused")
        return await real_call(node_id, method, payload, timeout)

    tr.call = refusing_call
    asyncio.run(mgr._reconcile())

    after = store.get_blob(meta.id)
    assert len([r for r in after.replicas if r in reg.online_ids]) == 2, (
        f"a failed peer pull cost the object a replica: {after.replicas}")
    assert mgr.repair_relays == 1 and mgr.repair_peer_pulls == 0
    assert any(m == "blob_pull" for _, m in tr.calls), (
        "the peer path was never even attempted, so this proves nothing about "
        "the fallback")


def test_the_repair_actually_happened_so_zero_relay_cannot_pass_vacuously():
    """Paired non-vacuity: prove the loss was real and the repair was needed.

    Without this, the assertion above passes trivially the day _reconcile decides
    there is nothing to do -- zero bytes relayed because zero bytes moved.
    """
    mgr, reg, store, tr = _manager(["A", "B", "C"])
    meta = _seed(mgr, tr)
    victim, survivor = meta.replicas[0], meta.replicas[1]

    reg.kill(victim)
    tr.unreachable.add(victim)
    live_before = [r for r in store.get_blob(meta.id).replicas
                   if r in reg.online_ids]
    assert live_before == [survivor], (
        f"the departure did not reduce live holders: {live_before}")

    asyncio.run(mgr._reconcile())

    after = store.get_blob(meta.id)
    fresh = [r for r in after.replicas if r != survivor]
    assert fresh, (
        f"nothing was repaired, so a zero-relay assertion would be measuring an "
        f"absence of work rather than an absence of relaying: {after.replicas}")
    assert tr.disks[fresh[0]][meta.id] == WEIGHTS, (
        f"{fresh[0]} is recorded as a replica but does not hold the bytes")


def test_a_repair_with_no_live_source_relays_nothing_and_claims_nothing():
    """If every holder is gone the bytes are gone. Nothing may be invented.

    A phantom holder is worse than an acknowledged loss: it makes the next repair
    pass skip the group, turning a recoverable outage into a permanent one.
    """
    mgr, reg, store, tr = _manager(["A", "B", "C"])
    meta = asyncio.run(mgr.put(WEIGHTS, name="g-layer-0", replication=1))
    tr.reset_counters()
    holder = meta.replicas[0]

    reg.kill(holder)
    tr.unreachable.add(holder)
    asyncio.run(mgr._reconcile())

    after = store.get_blob(meta.id)
    assert [r for r in after.replicas if r in reg.online_ids] == [], (
        f"a live node is recorded as holding a group no live machine has: "
        f"{after.replicas}")
    for nid in reg.online_ids:
        assert meta.id not in tr.disks[nid], (
            f"{nid} somehow acquired bytes that no reachable machine held")


@pytest.mark.parametrize("desired,live_holders,expect_growth", [
    (2, 1, True),
    (2, 2, False),
    (3, 2, True),
])
def test_repair_moves_bytes_only_when_the_object_is_actually_short(
        desired, live_holders, expect_growth):
    """A repair loop that copies when nothing is missing is a permanent leak.

    It runs forever, on every object, on every pass. On a pool holding weight
    groups that is the difference between a background task and an outage.
    """
    mgr, reg, store, tr = _manager(["A", "B", "C", "D"])
    meta = asyncio.run(mgr.put(WEIGHTS, name="g-layer-0", replication=2))
    meta.replication_desired = desired
    store.put_blob(meta)
    tr.reset_counters()

    if live_holders < 2:
        victim = meta.replicas[0]
        reg.kill(victim)
        tr.unreachable.add(victim)

    before = len([r for r in store.get_blob(meta.id).replicas
                  if r in reg.online_ids])
    asyncio.run(mgr._reconcile())
    after = len([r for r in store.get_blob(meta.id).replicas
                 if r in reg.online_ids])

    assert (after > before) is expect_growth, (
        f"desired={desired} live_holders={live_holders}: live replicas went "
        f"{before} -> {after}; expected growth={expect_growth}")
    if not expect_growth:
        assert tr.relayed_in == 0 and tr.relayed_out == 0, (
            f"moved {tr.relayed_out} bytes for an object already at its desired "
            f"replication")


def test_a_repaired_replica_holds_the_exact_bytes_not_merely_the_right_count():
    """A counter is not an identity.

    ``len(replicas) == 2`` is satisfied by a node holding the wrong bytes under
    the right id. Content addressing is supposed to make that impossible, so
    assert the digest and the bytes rather than the count.
    """
    mgr, reg, store, tr = _manager(["A", "B", "C"])
    meta = _seed(mgr, tr)
    victim = meta.replicas[0]

    reg.kill(victim)
    tr.unreachable.add(victim)
    asyncio.run(mgr._reconcile())

    after = store.get_blob(meta.id)
    live = [r for r in after.replicas if r in reg.online_ids]
    assert len(live) == 2
    for nid in live:
        raw = tr.disks[nid][meta.id]
        assert hashlib.sha256(raw).hexdigest() == meta.id, (
            f"{nid} holds bytes that do not hash to {meta.id[:12]}")
        assert raw == WEIGHTS
    assert _sha(WEIGHTS) == meta.id
