"""Does erasure repair funnel k shards through the coordinator?

Replication repair was fixed to send a byte-free plan naming peer URLs
(``tests/test_storage_repair_locality.py``). Erasure repair is the harder half of
the same problem and was deliberately left relaying: to rebuild one lost shard the
coordinator pulled **k** surviving shards into its own memory, base64-decoded them,
reconstructed the object, re-encoded it, and pushed the replacement back out.

That is worse than replication repair by a factor of k. At the cold profile RS(29,80)
one node departure drags 29 shards through a single uplink and the coordinator holds
the whole reconstructed object in RAM. Repair is the *steady-state* cost of a churning
pool -- one per object per departure -- so this is a hard bottleneck exactly at the
scale the product exists to serve.

Nothing about correctness can see this. The relay path reconstructs perfectly and
every durability assertion passes either way. Only counting who carried the bytes
distinguishes them, which is what these tests do.

The fakes come from tests/test_storage.py because they verify digests the way a real
agent does; a fake that accepts bytes it was never sent cannot tell you who sent them.
"""
from __future__ import annotations

import asyncio
import base64
import hashlib

import pytest

from test_storage import FakeEvents, FakeRegistry, FakeStore, FakeTransport
from meshpool.common.transport import TransportError
from meshpool.storage.engine import StorageManager
from meshpool.storage.erasure import RSCodec

PAYLOAD = b"pretend-this-is-an-erasure-coded-object\n" * 512


class ECRelayCountingTransport(FakeTransport):
    """Counts what the coordinator carried, and reconstructs like a real agent would.

    ``blob_get`` is the control plane pulling shards into its own memory; ``blob_put``
    with a payload is it pushing bytes back out. ``blob_ec_repair`` is the delegated
    path: it must carry only ids and peer URLs, and the *node* does the fetching,
    decoding and re-encoding.
    """

    def __init__(self, node_ids, unreachable=None, refuse_ec=False):
        super().__init__(node_ids, unreachable=unreachable)
        self.relayed_in = 0
        self.relayed_out = 0
        self.ec_plans: list[dict] = []
        self.refuse_ec = refuse_ec

    def _resolve(self, url: str) -> str:
        # FakeRegistry mints http://127.0.0.1:0/<node_id>, so the trailing path
        # segment resolves a url back to the disk it names.
        return str(url).rstrip("/").rsplit("/", 1)[-1]

    async def call(self, node_id, method, payload, timeout=None):
        if method != "blob_ec_repair":
            res = await super().call(node_id, method, payload, timeout)
            if method == "blob_get" and res.get("found"):
                self.relayed_in += len(base64.b64decode(res["data"]))
            elif method == "blob_put" and payload.get("data"):
                self.relayed_out += len(base64.b64decode(payload["data"]))
            return res

        self.calls.append((node_id, method))
        self.ec_plans.append(payload)
        if self.refuse_ec:
            raise TransportError("node refuses delegated EC repair")
        if node_id in self.unreachable or node_id not in self.disks:
            raise TransportError(f"node {node_id} unreachable")

        k, m = int(payload["k"]), int(payload["m"])
        shards: list[bytes | None] = [None] * (k + m)
        for entry in payload["shards"]:
            for url in entry.get("sources") or []:
                src = self._resolve(url)
                if src in self.unreachable:
                    continue
                raw = self.disks.get(src, {}).get(entry["id"])
                if raw is None:
                    continue
                if hashlib.sha256(raw).hexdigest() != entry["id"]:
                    continue          # an agent verifies before it trusts
                shards[int(entry["index"])] = raw
                break
        if sum(1 for s in shards if s is not None) < k:
            raise TransportError("not enough reachable shards to reconstruct")

        codec = RSCodec(k, m)
        rebuilt = codec.encode(codec.decode(shards))
        stored = []
        for want in payload["want"]:
            raw = rebuilt[int(want["index"])]
            if hashlib.sha256(raw).hexdigest() != want["id"]:
                raise TransportError("reconstructed shard failed integrity check")
            self.disks[node_id][want["id"]] = raw
            stored.append(want["id"])
        return {"ok": True, "stored": stored}

    def ops(self, method: str):
        return [c for c in self.calls if c[1] == method]

    def reset_counters(self):
        self.relayed_in = self.relayed_out = 0
        self.calls.clear()
        self.ec_plans.clear()


def _manager(node_ids, unreachable=None, refuse_ec=False):
    reg = FakeRegistry(node_ids)
    store = FakeStore()
    tr = ECRelayCountingTransport(node_ids, unreachable=unreachable, refuse_ec=refuse_ec)
    mgr = StorageManager(reg, store, None, FakeEvents(), tr)
    return mgr, reg, store, tr


def _ec_put(mgr, data=PAYLOAD, k=3, m=2) -> str:
    meta = asyncio.run(mgr.put(data, owner="u", name="obj",
                               erasure=True, ec_k=k, ec_m=m))
    assert mgr._is_ec(meta), "test requires a genuinely EC-coded object"
    return meta.id


def _a_shard_holder(mgr, blob_id: str) -> str:
    meta = mgr.store.get_blob(blob_id)
    for sid in meta.chunks:
        sm = mgr.store.get_blob(sid)
        if sm and sm.replicas:
            return sm.replicas[0]
    raise AssertionError("no shard holder found")


NODES = [f"n{i}" for i in range(9)]


def test_ec_repair_does_not_relay_shards_through_the_coordinator():
    mgr, reg, _store, tr = _manager(NODES)
    blob_id = _ec_put(mgr)

    tr.reset_counters()
    reg.kill(_a_shard_holder(mgr, blob_id))
    asyncio.run(mgr._reconcile())

    assert tr.ops("blob_get") == [], (
        f"coordinator pulled shards into its own memory: {tr.ops('blob_get')}")
    assert (tr.relayed_in, tr.relayed_out) == (0, 0), (
        f"coordinator relayed {tr.relayed_in} bytes in / {tr.relayed_out} out")


def test_ec_repair_delegates_reconstruction_to_a_node():
    mgr, reg, _store, tr = _manager(NODES)
    blob_id = _ec_put(mgr)

    tr.reset_counters()
    reg.kill(_a_shard_holder(mgr, blob_id))
    asyncio.run(mgr._reconcile())

    assert tr.ops("blob_ec_repair"), "no node was asked to reconstruct"
    plan = tr.ec_plans[0]
    assert plan["k"] == 3 and plan["m"] == 2
    assert plan["want"], "plan must name the shard(s) to rebuild"
    for entry in plan["shards"]:
        assert "data" not in entry, "plan carried payload bytes"


def test_ec_repair_plan_names_enough_live_sources_to_reconstruct():
    """A plan naming fewer than k reachable shards cannot possibly succeed."""
    mgr, reg, _store, tr = _manager(NODES)
    blob_id = _ec_put(mgr)

    tr.reset_counters()
    reg.kill(_a_shard_holder(mgr, blob_id))
    asyncio.run(mgr._reconcile())

    plan = tr.ec_plans[0]
    usable = [s for s in plan["shards"] if s.get("sources")]
    assert len(usable) >= plan["k"], (
        f"plan named {len(usable)} reachable shards but reconstruction needs {plan['k']}")


def test_ec_repair_never_names_the_target_as_its_own_source():
    mgr, reg, _store, tr = _manager(NODES)
    blob_id = _ec_put(mgr)

    tr.reset_counters()
    reg.kill(_a_shard_holder(mgr, blob_id))
    asyncio.run(mgr._reconcile())

    target = tr.ops("blob_ec_repair")[0][0]
    for entry in tr.ec_plans[0]["shards"]:
        for url in entry.get("sources") or []:
            assert tr._resolve(url) != target, "node told to fetch from itself"


def test_ec_repair_restores_the_object_and_it_still_reads_back():
    """Locality is an optimisation of the route; durability is the invariant."""
    mgr, reg, _store, _tr = _manager(NODES)
    blob_id = _ec_put(mgr)

    reg.kill(_a_shard_holder(mgr, blob_id))
    asyncio.run(mgr._reconcile())

    assert asyncio.run(mgr.get(blob_id)) == PAYLOAD


def test_ec_repair_falls_back_to_relay_when_no_node_will_reconstruct():
    """Peer reconstruction is an optimisation, never a reason to lose durability.

    If every candidate refuses the delegated repair, the object must still be
    restored by the old relay path rather than left a shard short.
    """
    mgr, reg, _store, tr = _manager(NODES, refuse_ec=True)
    blob_id = _ec_put(mgr)

    tr.reset_counters()
    reg.kill(_a_shard_holder(mgr, blob_id))
    asyncio.run(mgr._reconcile())

    assert tr.ops("blob_get"), "fallback relay never ran"
    assert asyncio.run(mgr.get(blob_id)) == PAYLOAD


def test_ec_repair_is_not_attempted_when_nothing_is_missing():
    """Non-vacuity: a healthy object must not generate repair traffic at all."""
    mgr, _reg, _store, tr = _manager(NODES)
    _ec_put(mgr)

    tr.reset_counters()
    asyncio.run(mgr._reconcile())

    assert tr.ops("blob_ec_repair") == []
    assert tr.ops("blob_get") == []
