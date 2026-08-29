"""Unit tests for the pooled, content-addressed object store (spec §P3).

Two layers are exercised in isolation:
  * ``BlobStore`` — the agent's on-disk, content-addressed store (real tmp dirs).
  * ``StorageManager`` — the control-plane replicator/repairer, driven against an
    in-memory fake cluster so replication and self-healing are deterministic.
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import types

import pytest

from meshpool.common import models as M
from meshpool.common.transport import TransportError
from meshpool.storage.blobstore import BlobStore, StorageFull
from meshpool.storage.engine import StorageManager, StorageError


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# --------------------------------------------------------------------------- #
# BlobStore (agent-side, real disk)
# --------------------------------------------------------------------------- #
def test_blobstore_roundtrips_and_dedupes(tmp_path):
    bs = BlobStore(str(tmp_path / "blobs"), cap_bytes=1_000_000)
    data = b"hello pooled storage" * 10
    bid = _sha(data)

    bs.put(bid, data)
    assert bs.has(bid)
    assert bs.get(bid) == data
    assert bs.used_bytes() == len(data)

    bs.put(bid, data)                       # idempotent: same content, no growth
    assert bs.used_bytes() == len(data)


def test_blobstore_rejects_hash_mismatch(tmp_path):
    bs = BlobStore(str(tmp_path / "blobs"), cap_bytes=1_000_000)
    with pytest.raises(ValueError):
        bs.put(_sha(b"claimed"), b"actually different bytes")


def test_blobstore_enforces_donated_cap(tmp_path):
    bs = BlobStore(str(tmp_path / "blobs"), cap_bytes=10)
    ten = b"0123456789"
    bs.put(_sha(ten), ten)                  # exactly fills the cap
    one = b"x"
    with pytest.raises(StorageFull):
        bs.put(_sha(one), one)


def test_blobstore_delete_frees_space(tmp_path):
    bs = BlobStore(str(tmp_path / "blobs"), cap_bytes=1_000)
    data = b"deletable"
    bid = _sha(data)
    bs.put(bid, data)
    assert bs.used_bytes() == len(data)
    bs.delete(bid)
    assert not bs.has(bid)
    assert bs.get(bid) is None
    assert bs.used_bytes() == 0


def test_blobstore_survives_reopen(tmp_path):
    root = str(tmp_path / "blobs")
    data = b"persisted across restarts"
    bid = _sha(data)
    BlobStore(root, cap_bytes=1_000).put(bid, data)
    # a fresh instance rescans the directory and accounts for existing bytes
    bs2 = BlobStore(root, cap_bytes=1_000)
    assert bs2.has(bid)
    assert bs2.used_bytes() == len(data)


# --------------------------------------------------------------------------- #
# Fakes for the control-plane StorageManager
# --------------------------------------------------------------------------- #
def _node(nid: str, disk_mb: int = 100_000, services: list | None = None):
    """A *real* Node, not a look-alike.

    This used to be a ``SimpleNamespace`` carrying only ``id`` and
    ``capability.disk_mb``. That is precisely the kind of fake that lets
    placement bugs through: when the storage planner learned to ask
    ``node.provides("storage")`` the fake could not answer, and a fake that
    cannot answer the questions the real code asks is not testing the real code.
    """
    cap = M.Capability(cpu_cores=4, ram_mb=4096, disk_mb=disk_mb)
    if services is not None:
        cap.services = services
    return M.Node(id=nid, name=nid, url=f"http://127.0.0.1:0/{nid}",
                  capability=cap, status=M.NodeStatus.ONLINE)


class FakeRegistry:
    def __init__(self, node_ids):
        self.nodes = {n: _node(n) for n in node_ids}
        self.online_ids = set(node_ids)

    def online(self):
        return [self.nodes[n] for n in self.nodes if n in self.online_ids]

    def schedulable_nodes(self):
        return self.online()

    def kill(self, nid):
        self.online_ids.discard(nid)


class FakeStore:
    """Mimics the SQLite store's serialize/deserialize (deep-copies on every op) so
    the manager can never rely on shared object identity."""

    def __init__(self):
        self._rows = {}

    def put_blob(self, meta):
        self._rows[meta.id] = meta.model_copy(deep=True)

    def get_blob(self, blob_id):
        m = self._rows.get(blob_id)
        return m.model_copy(deep=True) if m else None

    def list_blobs(self):
        return [m.model_copy(deep=True) for m in self._rows.values()]

    def delete_blob(self, blob_id):
        self._rows.pop(blob_id, None)


class FakeTransport:
    """An in-memory cluster: one blob dict per node, verifying digests like a real
    agent. Nodes in ``unreachable`` raise TransportError to simulate a down box."""

    def __init__(self, node_ids, unreachable=None):
        self.disks = {n: {} for n in node_ids}
        self.unreachable = set(unreachable or ())
        self.calls = []

    async def call(self, node_id, method, payload, timeout=None):
        self.calls.append((node_id, method))
        if node_id in self.unreachable or node_id not in self.disks:
            raise TransportError(f"node {node_id} unreachable")
        disk = self.disks[node_id]
        if method == "blob_put":
            raw = base64.b64decode(payload["data"])
            if hashlib.sha256(raw).hexdigest() != payload["blob_id"]:
                raise TransportError("hash mismatch at agent")
            disk[payload["blob_id"]] = raw
            return {"blob_id": payload["blob_id"], "stored": True, "size": len(raw)}
        if method == "blob_get":
            raw = disk.get(payload["blob_id"])
            if raw is None:
                return {"blob_id": payload["blob_id"], "found": False, "data": ""}
            return {"blob_id": payload["blob_id"], "found": True,
                    "data": base64.b64encode(raw).decode("ascii")}
        if method == "blob_delete":
            disk.pop(payload["blob_id"], None)
            return {"deleted": True}
        raise TransportError(f"unknown method {method}")


class FakeEvents:
    def __init__(self):
        self.events = []

    def publish(self, kind, data):
        self.events.append((kind, data))


def _manager(node_ids, unreachable=None):
    reg = FakeRegistry(node_ids)
    store = FakeStore()
    transport = FakeTransport(node_ids, unreachable=unreachable)
    events = FakeEvents()
    mgr = StorageManager(reg, store, None, events, transport)
    return mgr, reg, store, transport, events


# --------------------------------------------------------------------------- #
# StorageManager: replication, content-addressing, integrity, durability
# --------------------------------------------------------------------------- #
def test_put_replicates_to_two_distinct_nodes():
    mgr, _reg, store, transport, _ev = _manager(["A", "B", "C"])
    data = b"replicate me across the pool"
    meta = asyncio.run(mgr.put(data, owner="alice", name="doc.txt", replication=2))

    assert meta.id == _sha(data)
    assert len(meta.replicas) == 2
    assert len(set(meta.replicas)) == 2          # two DISTINCT nodes
    assert meta.status == "stored"
    # both chosen nodes actually hold the bytes
    for nid in meta.replicas:
        assert transport.disks[nid][meta.id] == data
    assert store.get_blob(meta.id) is not None


def test_put_is_content_addressed_and_idempotent():
    mgr, _reg, store, _t, _ev = _manager(["A", "B", "C"])
    data = b"same bytes twice"
    m1 = asyncio.run(mgr.put(data, replication=2))
    m2 = asyncio.run(mgr.put(data, replication=2))

    assert m1.id == m2.id                        # identical content => identical id
    assert len(store.list_blobs()) == 1          # not duplicated
    assert len(m2.replicas) == 2                 # topped up, not over-replicated


def test_get_returns_bytes():
    mgr, _reg, _s, _t, _ev = _manager(["A", "B"])
    data = b"fetch me back"
    meta = asyncio.run(mgr.put(data, replication=2))
    assert asyncio.run(mgr.get(meta.id)) == data


def test_get_unknown_blob_is_none():
    mgr, _reg, _s, _t, _ev = _manager(["A", "B"])
    assert asyncio.run(mgr.get(_sha(b"never stored"))) is None


def test_get_skips_a_tampered_copy_and_serves_a_good_one():
    mgr, _reg, _s, transport, _ev = _manager(["A", "B"])
    data = b"authentic content"
    meta = asyncio.run(mgr.put(data, replication=2))
    # a faulty/dishonest node returns wrong bytes under the right id
    transport.disks[meta.replicas[0]][meta.id] = b"tampered!"
    # the manager verifies the digest and falls through to the intact replica
    assert asyncio.run(mgr.get(meta.id)) == data


def test_get_returns_none_when_every_copy_is_corrupt():
    mgr, _reg, _s, transport, _ev = _manager(["A", "B"])
    data = b"all copies rot"
    meta = asyncio.run(mgr.put(data, replication=2))
    for nid in meta.replicas:
        transport.disks[nid][meta.id] = b"corrupt"
    assert asyncio.run(mgr.get(meta.id)) is None


def test_put_raises_when_no_node_available():
    mgr, reg, _s, _t, _ev = _manager(["A"])
    reg.kill("A")
    with pytest.raises(StorageError):
        asyncio.run(mgr.put(b"nowhere to go", replication=2))


def test_delete_removes_from_every_node_and_the_store():
    mgr, _reg, store, transport, _ev = _manager(["A", "B", "C"])
    meta = asyncio.run(mgr.put(b"temporary", replication=2))
    assert asyncio.run(mgr.delete(meta.id)) is True
    assert store.get_blob(meta.id) is None
    for nid in meta.replicas:
        assert meta.id not in transport.disks[nid]
    assert asyncio.run(mgr.delete(meta.id)) is False   # already gone


def test_on_node_offline_prunes_replica_and_degrades():
    mgr, _reg, store, _t, _ev = _manager(["A", "B", "C"])
    meta = asyncio.run(mgr.put(b"loses a home", replication=2))
    victim = meta.replicas[0]
    mgr.on_node_offline(victim)
    after = store.get_blob(meta.id)
    assert victim not in after.replicas
    assert len(after.replicas) == 1
    assert after.status == "degraded"


def test_reconcile_re_replicates_after_a_node_is_lost():
    # The headline durability property: lose a holding node, and the repair loop
    # rebuilds the missing copy onto a surviving node from the intact replica.
    async def scenario():
        mgr, reg, store, transport, _ev = _manager(["A", "B", "C"])
        meta = await mgr.put(b"must survive a machine failure", replication=2)
        assert len(meta.replicas) == 2

        victim = meta.replicas[0]
        survivor = meta.replicas[1]
        reg.kill(victim)                 # the node goes offline for good
        mgr.on_node_offline(victim)      # reaper hook prunes the dead copy
        assert store.get_blob(meta.id).status == "degraded"

        await mgr._reconcile()           # repair loop runs
        healed = store.get_blob(meta.id)
        return meta, healed, survivor, victim, transport

    meta, healed, survivor, victim, transport = asyncio.run(scenario())
    assert len(healed.replicas) == 2                 # back to full replication
    assert victim not in healed.replicas
    assert survivor in healed.replicas
    assert healed.status == "stored"
    # the newly-placed replica physically holds the bytes
    new_home = next(n for n in healed.replicas if n != survivor)
    assert transport.disks[new_home][meta.id] is not None


def test_reconcile_cannot_heal_when_no_live_copy_remains():
    # If every holder dies at once there is nothing to copy from — the object is
    # marked lost rather than silently resurrected.
    async def scenario():
        mgr, reg, store, _t, _ev = _manager(["A", "B", "C"])
        meta = await mgr.put(b"total loss", replication=2)
        for nid in list(meta.replicas):
            reg.kill(nid)
            mgr.on_node_offline(nid)
        await mgr._reconcile()
        return store.get_blob(meta.id)

    lost = asyncio.run(scenario())
    assert lost.replicas == []
    assert lost.status == "lost"


# --------------------------------------------------------------------------- #
# StorageManager: chunked large objects (bigger than any single machine could hold)
# --------------------------------------------------------------------------- #
def test_large_object_is_split_into_chunks_and_reassembled(monkeypatch):
    # An object larger than the chunk size is spread across the pool as many small
    # content-addressed chunks, so it can exceed what any one node could store.
    monkeypatch.setenv("MESHPOOL_CHUNK_SIZE", "64")
    mgr, _reg, store, _t, _ev = _manager(["A", "B", "C"])
    data = bytes(range(256)) * 4                     # 1024 bytes -> 16 chunks of 64
    meta = asyncio.run(mgr.put(data, owner="alice", name="big.bin", replication=2))

    assert meta.id == _sha(data)
    assert meta.size == len(data)
    assert len(meta.chunks) == len(data) // 64       # exactly split
    assert meta.replicas == []                       # the manifest itself holds no bytes
    assert meta.status == "stored"
    # the user-facing listing shows ONLY the manifest, never the internal chunks
    assert [b.id for b in mgr.list()] == [meta.id]
    # every chunk is a replicated, integrity-checkable blob in its own right
    holders = set()
    for cid in meta.chunks:
        cm = store.get_blob(cid)
        assert cm.internal is True
        assert len(cm.replicas) == 2 and len(set(cm.replicas)) == 2
        holders.update(cm.replicas)
    assert len(holders) >= 2                          # spread across the pool, not one box
    # and the whole thing round-trips byte-for-byte through reassembly
    assert asyncio.run(mgr.get(meta.id)) == data


def test_chunked_object_dedupes_on_reput(monkeypatch):
    monkeypatch.setenv("MESHPOOL_CHUNK_SIZE", "64")
    mgr, _reg, _s, _t, _ev = _manager(["A", "B", "C"])
    data = bytes(range(256)) * 4
    m1 = asyncio.run(mgr.put(data, replication=2))
    m2 = asyncio.run(mgr.put(data, replication=2))

    assert m1.id == m2.id
    assert m2.chunks == m1.chunks                     # same content => same chunk ids
    assert [b.id for b in mgr.list()] == [m1.id]      # one manifest, not duplicated


def test_chunked_object_reuses_shared_chunks_between_related_models(monkeypatch):
    """Dedup is about bytes not transferred, not whether the second object reads.

    Two fine-tunes of the same base should share the base chunks. This asserts the
    second upload only places the one changed chunk; if chunk ids include an offset,
    random salt, or anything other than content, the byte-transfer count jumps.
    """
    monkeypatch.setenv("MESHPOOL_CHUNK_SIZE", "64")
    mgr, _reg, _s, t, _ev = _manager(["A", "B", "C"])
    base = b"A" * 64 + b"B" * 64 + b"C" * 64
    tuned = b"A" * 64 + b"B" * 64 + b"D" * 64

    m1 = asyncio.run(mgr.put(base, replication=2))
    puts_after_base = sum(1 for _nid, method in t.calls if method == "blob_put")
    m2 = asyncio.run(mgr.put(tuned, replication=2))
    new_puts = sum(1 for _nid, method in t.calls if method == "blob_put") - puts_after_base

    assert m1.chunks[:2] == m2.chunks[:2]             # base chunks genuinely shared
    assert m1.chunks[2] != m2.chunks[2]               # changed chunk non-vacuity
    assert new_puts == 2                              # one changed chunk × 2 replicas


def test_chunked_upload_resumes_from_existing_chunks(monkeypatch):
    """A resumed upload must not restart from byte zero; count chunk writes."""
    monkeypatch.setenv("MESHPOOL_CHUNK_SIZE", "64")
    mgr, _reg, store, t, _ev = _manager(["A", "B", "C"])
    data = b"R" * 64 + b"S" * 64 + b"T" * 64
    first, second, third = data[:64], data[64:128], data[128:]

    asyncio.run(mgr.put(first + second, replication=2))
    initial_puts = sum(1 for _nid, method in t.calls if method == "blob_put")
    meta = asyncio.run(mgr.put(data, replication=2))
    resumed_puts = sum(1 for _nid, method in t.calls if method == "blob_put") - initial_puts

    assert meta.chunks[:2] == [_sha(first), _sha(second)]
    assert store.get_blob(_sha(first)).replicas
    assert store.get_blob(_sha(second)).replicas
    assert resumed_puts == 2                          # only the missing final chunk


def test_chunked_object_survives_node_loss_and_reassembles(monkeypatch):
    # The headline durability property at object scale: lose a machine holding chunks
    # and the repair loop rebuilds every under-replicated chunk from a surviving copy.
    monkeypatch.setenv("MESHPOOL_CHUNK_SIZE", "64")

    async def scenario():
        mgr, reg, store, _t, _ev = _manager(["A", "B", "C"])
        data = bytes(range(256)) * 8                  # 2048 bytes -> 32 chunks
        meta = await mgr.put(data, replication=2)
        victim = "A"
        reg.kill(victim)
        mgr.on_node_offline(victim)                   # reaper prunes the dead copies
        await mgr._reconcile()                        # repair loop heals the chunks
        whole = await mgr.get(meta.id)
        return meta, whole, data, store, victim

    meta, whole, data, store, victim = asyncio.run(scenario())
    assert whole == data                              # still byte-perfect
    for cid in meta.chunks:
        cm = store.get_blob(cid)
        assert victim not in cm.replicas
        assert len(cm.replicas) == 2                  # back to full replication
    assert store.get_blob(meta.id).status == "stored"  # manifest health restored


def test_deleting_a_chunked_object_removes_every_chunk(monkeypatch):
    monkeypatch.setenv("MESHPOOL_CHUNK_SIZE", "64")
    mgr, _reg, store, transport, _ev = _manager(["A", "B", "C"])
    data = bytes(range(256)) * 4
    meta = asyncio.run(mgr.put(data, replication=2))
    chunk_ids = list(meta.chunks)

    assert asyncio.run(mgr.delete(meta.id)) is True
    assert store.get_blob(meta.id) is None            # manifest gone
    for cid in chunk_ids:
        assert store.get_blob(cid) is None            # chunk metadata gone
        for nid in transport.disks:                   # and the bytes are gone everywhere
            assert cid not in transport.disks[nid]


# --------------------------------------------------------------------------- #
# StorageManager: internal objects (machinery like a hosted app's deploy bundle)
# --------------------------------------------------------------------------- #
def test_internal_put_is_hidden_from_listing_but_still_readable():
    # A large app's deploy bundle is stored internal=True so it is replicated and
    # fetchable like any object, yet never clutters the user's own Files listing.
    mgr, _reg, store, _t, _ev = _manager(["A", "B", "C"])
    data = b"deploy bundle bytes for a hosted app"
    meta = asyncio.run(mgr.put(data, owner="alice", name="bundle:site",
                               replication=2, internal=True))

    assert meta.internal is True
    assert len(meta.replicas) == 2                     # replicated like any object
    assert mgr.list() == []                            # hidden from the user listing
    assert mgr.list(owner="alice") == []               # even for its own owner
    assert store.get_blob(meta.id) is not None         # but the metadata exists
    assert asyncio.run(mgr.get(meta.id)) == data       # and it round-trips

