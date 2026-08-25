"""Comprehensive tests for the Reed–Solomon erasure coding implementation.

Exercises:
  * GF(2^8) arithmetic correctness
  * RSCodec encode/decode round-trips: all edge cases
  * Corruption detection (bad shard hash)
  * Property: every combination of up to m lost shards recovers exactly
  * Placement diversity: no two shards of same object on same node
  * Repair: fail-and-repair cycles preserve data exactly
  * StorageManager integration with EC
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import itertools
import os
import random
import struct
import types
from typing import Optional

import pytest

from meshpool.storage.erasure import (
    RSCodec, _gf_mul, _gf_div, _gf_inv, _gf_pow,
    _mat_mul_np, _mat_invert, _cauchy_matrix,
    _pack_shard, _unpack_shard, _HEADER_SIZE,
    erasure_overhead, erasure_crossover_bytes,
)
from meshpool.storage.engine import (
    StorageManager, StorageError, DEFAULT_EC_K, DEFAULT_EC_M,
    _pack_ec_params, _unpack_ec_params,
)
from meshpool.common import models as M
from meshpool.common.transport import TransportError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _node(nid: str, disk_mb: int = 100_000, services: list | None = None):
    """A *real* Node — see the identical helper in test_storage.py for why a
    ``SimpleNamespace`` look-alike is not good enough here."""
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


def _manager_ec(n_nodes: int = 12, k: int = 3, m: int = 2):
    """Build a manager with enough nodes for EC (k,m)."""
    node_ids = [f"N{i:02d}" for i in range(n_nodes)]
    reg = FakeRegistry(node_ids)
    store = FakeStore()
    transport = FakeTransport(node_ids)
    events = FakeEvents()
    mgr = StorageManager(reg, store, None, events, transport)
    return mgr, reg, store, transport, events, node_ids, k, m


# ---------------------------------------------------------------------------
# GF(2^8) arithmetic
# ---------------------------------------------------------------------------
class TestGFArithmetic:
    def test_mul_by_zero(self):
        for x in [0, 1, 127, 255]:
            assert _gf_mul(x, 0) == 0
            assert _gf_mul(0, x) == 0

    def test_mul_by_one(self):
        for x in [1, 13, 99, 255]:
            assert _gf_mul(x, 1) == x
            assert _gf_mul(1, x) == x

    def test_mul_commutative(self):
        rng = random.Random(42)
        for _ in range(200):
            a, b = rng.randint(0, 255), rng.randint(0, 255)
            assert _gf_mul(a, b) == _gf_mul(b, a)

    def test_mul_distributive(self):
        """a*(b XOR c) == a*b XOR a*c"""
        rng = random.Random(99)
        for _ in range(100):
            a, b, c = rng.randint(1, 255), rng.randint(0, 255), rng.randint(0, 255)
            assert _gf_mul(a, b ^ c) == _gf_mul(a, b) ^ _gf_mul(a, c)

    def test_div_inverse_of_mul(self):
        rng = random.Random(7)
        for _ in range(200):
            a = rng.randint(1, 255)
            b = rng.randint(1, 255)
            assert _gf_div(_gf_mul(a, b), b) == a

    def test_inv_times_orig_is_one(self):
        for x in range(1, 256):
            assert _gf_mul(x, _gf_inv(x)) == 1

    def test_pow_zero_is_one(self):
        for x in [1, 5, 200]:
            assert _gf_pow(x, 0) == 1

    def test_pow_one_is_x(self):
        for x in range(1, 256):
            assert _gf_pow(x, 1) == x


# ---------------------------------------------------------------------------
# RSCodec: encode/decode correctness
# ---------------------------------------------------------------------------
class TestRSCodecRoundtrip:
    @pytest.mark.parametrize("k,m", [(3, 2), (6, 3), (4, 4)])
    def test_empty_input(self, k, m):
        codec = RSCodec(k, m)
        shards = codec.encode(b"")
        assert len(shards) == k + m
        result = codec.decode(shards)
        assert result == b""

    @pytest.mark.parametrize("k,m", [(3, 2), (6, 3)])
    def test_single_byte(self, k, m):
        codec = RSCodec(k, m)
        data = b"\xAB"
        shards = codec.encode(data)
        assert codec.decode(shards) == data

    @pytest.mark.parametrize("k,m", [(3, 2), (6, 3)])
    def test_all_256_byte_values(self, k, m):
        codec = RSCodec(k, m)
        data = bytes(range(256))
        shards = codec.encode(data)
        assert codec.decode(shards) == data

    @pytest.mark.parametrize("size", [0, 1, 5, 100, 720, 1023, 1024, 10000])
    def test_various_sizes(self, size):
        codec = RSCodec(6, 3)
        data = bytes(range(256)) * (size // 256 + 1)
        data = data[:size]
        shards = codec.encode(data)
        assert len(shards) == 9
        assert codec.decode(shards) == data

    def test_random_binary_data(self):
        rng = random.Random(1234)
        codec = RSCodec(6, 3)
        for _ in range(20):
            size = rng.randint(0, 5000)
            data = bytes(rng.getrandbits(8) for _ in range(size))
            shards = codec.encode(data)
            assert codec.decode(shards) == data

    def test_shard_count_is_k_plus_m(self):
        for k, m in [(3, 2), (6, 3), (10, 5)]:
            codec = RSCodec(k, m)
            shards = codec.encode(b"hello world, this is some test data padded well")
            assert len(shards) == k + m

    def test_shard_sizes_equal(self):
        codec = RSCodec(6, 3)
        data = b"equal shard sizes for distributed storage" * 10
        shards = codec.encode(data)
        sizes = [len(s) for s in shards]
        # All shards include the same 40-byte header + same-sized payload
        assert all(s == sizes[0] for s in sizes), f"unequal sizes: {sizes}"


# ---------------------------------------------------------------------------
# RSCodec: loss tolerance
# ---------------------------------------------------------------------------
class TestRSCodecLoss:
    @pytest.mark.parametrize("k,m", [(3, 2), (6, 3)])
    def test_recover_from_exactly_m_losses(self, k, m):
        """Every combination of exactly m lost shards must recover exactly."""
        codec = RSCodec(k, m)
        data = b"durability under loss: " + bytes(range(64))
        original_shards = codec.encode(data)

        for lost_indices in itertools.combinations(range(k + m), m):
            shards = list(original_shards)
            for i in lost_indices:
                shards[i] = None
            recovered = codec.decode(shards)
            assert recovered == data, (
                f"Failed to recover with lost={lost_indices}")

    @pytest.mark.parametrize("k,m", [(3, 2), (6, 3)])
    def test_cannot_recover_from_m_plus_1_losses(self, k, m):
        """m+1 losses must raise ValueError, not silently produce wrong data."""
        codec = RSCodec(k, m)
        data = b"irrecoverable loss: " + bytes(range(64))
        original_shards = codec.encode(data)
        # Lose the first m+1 shards
        shards = list(original_shards)
        for i in range(m + 1):
            shards[i] = None
        with pytest.raises(ValueError):
            codec.decode(shards)

    def test_random_loss_patterns(self):
        """Property test: random binary data, random ≤m losses, always recovers."""
        rng = random.Random(8888)
        k, m = 6, 3
        codec = RSCodec(k, m)
        for trial in range(30):
            size = rng.randint(1, 3000)
            data = bytes(rng.getrandbits(8) for _ in range(size))
            shards = codec.encode(data)
            n_lost = rng.randint(0, m)
            lost = rng.sample(range(k + m), n_lost)
            for i in lost:
                shards[i] = None
            recovered = codec.decode(shards)
            assert recovered == data, (
                f"trial={trial} size={size} lost={lost}: recovery wrong")


# ---------------------------------------------------------------------------
# RSCodec: corruption detection
# ---------------------------------------------------------------------------
class TestRSCodecCorruption:
    def test_corrupt_shard_detected_and_excluded(self):
        """A single bit flip in a shard's payload is detected via its hash,
        and the remaining good shards are used for recovery."""
        k, m = 6, 3
        codec = RSCodec(k, m)
        data = b"corruption must be detected and routed around: " + bytes(range(64))
        shards = list(codec.encode(data))

        # Flip a bit in shard 0's payload (after the 40-byte header)
        raw = bytearray(shards[0])
        raw[_HEADER_SIZE] ^= 0x01   # flip one bit
        shards[0] = bytes(raw)

        # Decoding with the corrupted shard as-is should succeed using the other k shards
        recovered = codec.decode(shards)
        assert recovered == data, "Recovery failed even with enough good shards"

    def test_corrupt_shard_header_detected(self):
        """Corruption in the header (claimed hash) is also caught."""
        k, m = 6, 3
        codec = RSCodec(k, m)
        data = b"header corruption: " + bytes(range(64))
        shards = list(codec.encode(data))

        # Corrupt the hash bytes in shard 2's header
        raw = bytearray(shards[2])
        raw[8] ^= 0xFF   # corrupt the hash field
        shards[2] = bytes(raw)

        recovered = codec.decode(shards)
        assert recovered == data

    def test_too_many_corrupt_shards_raises(self):
        """If m+1 shards are corrupted (treated as missing), must raise."""
        k, m = 6, 3
        codec = RSCodec(k, m)
        data = b"too many corrupted shards" + bytes(range(64))
        shards = list(codec.encode(data))

        for i in range(m + 1):
            raw = bytearray(shards[i])
            raw[_HEADER_SIZE] ^= 0x01
            shards[i] = bytes(raw)

        with pytest.raises(ValueError):
            codec.decode(shards)


# ---------------------------------------------------------------------------
# RSCodec: shard header round-trip
# ---------------------------------------------------------------------------
class TestShardHeader:
    def test_header_roundtrip(self):
        payload = b"test payload content for header verification"
        packed = _pack_shard(12345, 7, payload)
        ol, idx, valid, p = _unpack_shard(packed)
        assert ol == 12345
        assert idx == 7
        assert valid is True
        assert p == payload

    def test_short_data_invalid(self):
        _, _, valid, _ = _unpack_shard(b"short")
        assert valid is False


# ---------------------------------------------------------------------------
# Placement diversity
# ---------------------------------------------------------------------------
class TestPlacementDiversity:
    def test_no_two_shards_on_same_node(self):
        """The fundamental placement invariant: every shard of an EC-coded object
        must live on a distinct node."""
        k, m = 3, 2
        mgr, reg, store, transport, events, node_ids, k, m = _manager_ec(
            n_nodes=k + m + 2, k=k, m=m)
        # Use data well above the crossover (~600 bytes for k=3,m=2)
        data = b"placement diversity test: " + bytes(range(200)) + b"\x00" * 500
        meta = asyncio.run(mgr.put(data, erasure=True, ec_k=k, ec_m=m))

        # Collect which node holds each shard
        shard_nodes = []
        for sid in meta.chunks:
            sm = store.get_blob(sid)
            assert sm is not None, f"shard blob {sid[:8]} missing from store"
            assert len(sm.replicas) == 1, "each shard must be on exactly one node"
            shard_nodes.append(sm.replicas[0])

        assert len(shard_nodes) == k + m, "wrong number of shards"
        assert len(set(shard_nodes)) == k + m, (
            f"shards share nodes! placement={shard_nodes}")

    def test_shards_spread_across_distinct_nodes(self):
        """No node should hold more than one shard of the same object."""
        k, m = 6, 3
        n_nodes = k + m + 3
        mgr, reg, store, transport, events, node_ids, k, m = _manager_ec(
            n_nodes=n_nodes, k=k, m=m)
        data = bytes(range(256)) * 10   # 2560 bytes — well above crossover

        meta = asyncio.run(mgr.put(data, erasure=True, ec_k=k, ec_m=m))
        assert meta.chunks, "should have shard IDs in chunks"

        shard_nodes = []
        for sid in meta.chunks:
            sm = store.get_blob(sid)
            assert sm is not None
            shard_nodes.extend(sm.replicas)

        node_counts = {}
        for n in shard_nodes:
            node_counts[n] = node_counts.get(n, 0) + 1
        overloaded = {n: c for n, c in node_counts.items() if c > 1}
        assert not overloaded, (
            f"Nodes hold multiple shards of the same object: {overloaded}")


# ---------------------------------------------------------------------------
# StorageManager EC integration
# ---------------------------------------------------------------------------
class TestStorageManagerEC:
    # Data must be above the EC crossover threshold (~600 bytes for k=3,m=2)
    _DATA = b"EC storage manager test data: " + bytes(range(256)) * 4   # ~1054 bytes

    def _make_mgr(self, k=3, m=2, n_extra=2):
        node_ids = [f"N{i:02d}" for i in range(k + m + n_extra)]
        reg = FakeRegistry(node_ids)
        store = FakeStore()
        transport = FakeTransport(node_ids)
        events = FakeEvents()
        mgr = StorageManager(reg, store, None, events, transport)
        return mgr, reg, store, transport, events, node_ids

    def test_ec_put_and_get_roundtrip(self):
        k, m = 3, 2
        mgr, _, store, _, _, _ = self._make_mgr(k=k, m=m)
        data = self._DATA
        meta = asyncio.run(mgr.put(data, erasure=True, ec_k=k, ec_m=m))
        assert meta.status == "stored"
        assert len(meta.chunks) == k + m
        recovered = asyncio.run(mgr.get(meta.id))
        assert recovered == data

    def test_ec_get_survives_m_node_failures(self):
        """Data is recoverable after losing exactly m nodes."""
        k, m = 3, 2
        mgr, reg, store, transport, _, node_ids = self._make_mgr(k=k, m=m)
        data = self._DATA
        meta = asyncio.run(mgr.put(data, erasure=True, ec_k=k, ec_m=m))

        # Find which m nodes hold shards and kill them
        shard_holders = []
        for sid in meta.chunks:
            sm = store.get_blob(sid)
            if sm and sm.replicas:
                shard_holders.extend(sm.replicas)
        shard_holders = list(dict.fromkeys(shard_holders))  # unique, ordered

        for victim in shard_holders[:m]:
            transport.unreachable.add(victim)
            reg.kill(victim)

        recovered = asyncio.run(mgr.get(meta.id))
        assert recovered == data, "failed to recover after m node failures"

    def test_ec_fails_cleanly_with_m_plus_1_losses(self):
        """If m+1 shards are gone, get() must return None, not wrong data."""
        k, m = 3, 2
        mgr, reg, store, transport, _, node_ids = self._make_mgr(k=k, m=m)
        data = self._DATA
        meta = asyncio.run(mgr.put(data, erasure=True, ec_k=k, ec_m=m))

        # Kill m+1 shard holders
        shard_holders = []
        for sid in meta.chunks:
            sm = store.get_blob(sid)
            if sm and sm.replicas:
                for nid in sm.replicas:
                    if nid not in shard_holders:
                        shard_holders.append(nid)

        for victim in shard_holders[:m + 1]:
            transport.unreachable.add(victim)
            reg.kill(victim)
            # Also erase from disk to ensure unavailability
            transport.disks[victim].clear()

        result = asyncio.run(mgr.get(meta.id))
        assert result is None, "should return None, not corrupt data"

    def test_ec_corrupted_shard_routed_around(self):
        """A node returning corrupted shard bytes does not cause wrong output."""
        k, m = 3, 2
        mgr, reg, store, transport, _, node_ids = self._make_mgr(k=k, m=m)
        data = self._DATA
        meta = asyncio.run(mgr.put(data, erasure=True, ec_k=k, ec_m=m))

        # Corrupt one shard on its node
        first_shard_id = meta.chunks[0]
        sm = store.get_blob(first_shard_id)
        holder = sm.replicas[0]
        raw = transport.disks[holder][first_shard_id]
        corrupted = bytearray(raw)
        corrupted[_HEADER_SIZE] ^= 0x5A   # flip bits in shard payload
        transport.disks[holder][first_shard_id] = bytes(corrupted)

        recovered = asyncio.run(mgr.get(meta.id))
        assert recovered == data, "should recover past a corrupted shard"

    def test_ec_idempotent_put(self):
        """Re-putting identical bytes does not create duplicate shard blobs."""
        k, m = 3, 2
        mgr, _, store, _, _, _ = self._make_mgr(k=k, m=m)
        data = self._DATA
        m1 = asyncio.run(mgr.put(data, erasure=True, ec_k=k, ec_m=m))
        m2 = asyncio.run(mgr.put(data, erasure=True, ec_k=k, ec_m=m))
        assert m1.id == m2.id
        assert m2.chunks == m1.chunks   # same shards reused

    def test_ec_internal_flag_hidden_from_listing(self):
        k, m = 3, 2
        mgr, _, store, _, _, _ = self._make_mgr(k=k, m=m)
        data = self._DATA
        meta = asyncio.run(mgr.put(data, erasure=True, ec_k=k, ec_m=m,
                                   internal=True))
        assert meta.internal is True
        assert mgr.list() == []
        assert asyncio.run(mgr.get(meta.id)) == data

    def test_ec_delete_removes_all_shards(self):
        k, m = 3, 2
        mgr, _, store, transport, _, _ = self._make_mgr(k=k, m=m)
        data = self._DATA
        meta = asyncio.run(mgr.put(data, erasure=True, ec_k=k, ec_m=m))
        shard_ids = list(meta.chunks)

        assert asyncio.run(mgr.delete(meta.id)) is True
        assert store.get_blob(meta.id) is None
        for sid in shard_ids:
            assert store.get_blob(sid) is None
            for disk in transport.disks.values():
                assert sid not in disk


# ---------------------------------------------------------------------------
# Repair path
# ---------------------------------------------------------------------------
class TestRepairPath:
    # Data above crossover for k=3,m=2 (~600 bytes)
    _DATA = b"repair test data: " + bytes(range(256)) * 4   # ~1042 bytes

    def _make_mgr(self, k=3, m=2, n_extra=2):
        node_ids = [f"N{i:02d}" for i in range(k + m + n_extra)]
        reg = FakeRegistry(node_ids)
        store = FakeStore()
        transport = FakeTransport(node_ids)
        events = FakeEvents()
        mgr = StorageManager(reg, store, None, events, transport)
        return mgr, reg, store, transport, events, node_ids

    def test_repair_restores_full_redundancy(self):
        """After one node dies and repair runs, the object has k+m shards again."""
        k, m = 3, 2

        async def scenario():
            mgr, reg, store, transport, events, node_ids = self._make_mgr(k=k, m=m)
            data = self._DATA
            meta = await mgr.put(data, erasure=True, ec_k=k, ec_m=m)
            assert len(meta.chunks) == k + m

            # Kill one shard holder
            first_shard_id = meta.chunks[0]
            sm = store.get_blob(first_shard_id)
            victim = sm.replicas[0]
            reg.kill(victim)
            # Remove from victim's disk so _fetch fails
            transport.disks[victim].clear()
            # Remove replica record so engine sees it as missing
            sm.replicas = []
            store.put_blob(sm)

            # Run reconcile (repair)
            async with mgr._lock:
                await mgr._reconcile()

            # Verify data still recoverable
            recovered = await mgr.get(meta.id)
            assert recovered == data
            return store, meta.id, k + m

        store, blob_id, total = asyncio.run(scenario())
        # Verify the manifest shows full redundancy again
        meta = store.get_blob(blob_id)
        live_shards = sum(
            1 for sid in meta.chunks
            if (sm := store.get_blob(sid)) and sm.replicas)
        assert live_shards == total, (
            f"repair did not restore full redundancy: {live_shards}/{total}")

    def test_repeated_fail_and_repair_cycles(self):
        """Multiple fail-repair cycles must leave data identical every time."""
        k, m = 3, 2

        async def scenario():
            mgr, reg, store, transport, events, node_ids = self._make_mgr(
                k=k, m=m, n_extra=6)
            data = self._DATA
            meta = await mgr.put(data, erasure=True, ec_k=k, ec_m=m)

            for cycle in range(3):
                # Kill one random shard holder
                meta_now = store.get_blob(meta.id)
                for sid in meta_now.chunks:
                    sm = store.get_blob(sid)
                    if sm and sm.replicas:
                        victim = sm.replicas[0]
                        reg.kill(victim)
                        transport.disks[victim].clear()
                        sm.replicas = []
                        store.put_blob(sm)
                        break

                # Repair
                async with mgr._lock:
                    await mgr._reconcile()

                recovered = await mgr.get(meta.id)
                assert recovered == data, f"cycle {cycle}: data corrupted!"

            return data

        data = asyncio.run(scenario())
        assert data == self._DATA

    def test_repair_respects_node_diversity(self):
        """After repair, no node should hold two shards of the same object."""
        k, m = 3, 2

        async def scenario():
            mgr, reg, store, transport, events, node_ids = self._make_mgr(
                k=k, m=m, n_extra=4)
            data = self._DATA
            meta = await mgr.put(data, erasure=True, ec_k=k, ec_m=m)

            # Kill first shard holder
            first_sm = store.get_blob(meta.chunks[0])
            victim = first_sm.replicas[0]
            reg.kill(victim)
            transport.disks[victim].clear()
            first_sm.replicas = []
            store.put_blob(first_sm)

            async with mgr._lock:
                await mgr._reconcile()

            meta = store.get_blob(meta.id)
            shard_nodes = []
            for sid in meta.chunks:
                sm = store.get_blob(sid)
                if sm:
                    shard_nodes.extend(sm.replicas)
            return shard_nodes

        shard_nodes = asyncio.run(scenario())
        assert len(set(shard_nodes)) == len(shard_nodes), (
            f"repair placed two shards on same node: {shard_nodes}")


# ---------------------------------------------------------------------------
# EC pack/unpack params
# ---------------------------------------------------------------------------
def test_pack_unpack_ec_params():
    for k in [1, 3, 6, 10, 15]:
        for m in [1, 2, 3, 5, 10]:
            packed = _pack_ec_params(k, m)
            k2, m2 = _unpack_ec_params(packed)
            assert k2 == k and m2 == m


# ---------------------------------------------------------------------------
# Crossover and overhead calculations
# ---------------------------------------------------------------------------
def test_erasure_overhead_calculation():
    assert erasure_overhead(6, 3) == pytest.approx(1.5)
    assert erasure_overhead(1, 0) == pytest.approx(1.0)
    assert erasure_overhead(4, 4) == pytest.approx(2.0)


def test_crossover_is_positive():
    """The crossover should be a positive number of bytes."""
    x = erasure_crossover_bytes(6, 3, replication=2)
    assert x > 0


def test_crossover_ec_cheaper_above_threshold():
    """Above the crossover point, EC uses fewer bytes than replication."""
    k, m = 6, 3
    rep = 2
    x = erasure_crossover_bytes(k, m, replication=rep)
    from meshpool.storage.erasure import _HEADER_SIZE, erasure_overhead

    # Just above crossover: EC should be cheaper
    size = x + 100
    shard_payload = (size + k - 1) // k   # ceil(size/k)
    ec_cost = (k + m) * (shard_payload + _HEADER_SIZE)
    repl_cost = rep * size
    assert ec_cost <= repl_cost, (
        f"EC still more expensive than replication at size={size}: "
        f"ec={ec_cost} repl={repl_cost}")
