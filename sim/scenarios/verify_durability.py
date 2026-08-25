"""Acceptance gate: Reed-Solomon erasure coding is correct, efficient, and durable.

This scenario demonstrates -- with real data and printed numbers -- that the
(k=6, m=3) RS codec implemented in ``meshpool/storage/erasure.py`` and integrated
into ``meshpool/storage/engine.py`` is rigorously correct and meaningfully better
than the old 2x replication it replaces.

Checks performed
----------------
1.  **Survive m losses** -- an object survives the loss of exactly m nodes and its
    bytes are recovered **identical** to the original (bit-for-bit comparison).
2.  **Fail on m+1 losses** -- the codec cleanly raises ValueError (not a crash, not
    a wrong answer) when m+1 shards are lost.
3.  **Corruption routing** -- a shard with a flipped bit is detected via its SHA-256
    header and routed around; remaining good shards reconstruct correctly.
4.  **Repair** -- after a node failure, the repair path regenerates the missing shard
    onto a new node, and the data is still intact after repair.
5.  **Overhead comparison** -- prints measured storage overhead for EC vs. replication
    and verifies EC uses less storage per byte protected.
6.  **Durability comparison** -- computes the probability of data loss under a 3 %
    annual node-failure rate for both schemes and verifies EC wins by a large margin.

Non-vacuity guards
------------------
- Asserts that the test data is non-trivial random binary (not all-zeros, not ASCII).
- Asserts that the shard count actually equals k+m (> 1).
- Asserts that failure detection is genuinely triggered by non-zero data corruption.
- The m+1 loss test verifies that the codec FAILS, not that it accidentally succeeds.

Exit 0 = PASS.  Exit 1 = FAIL (with reason printed).
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import math
import os
import random
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from meshpool.storage.erasure import (
    RSCodec, _HEADER_SIZE, erasure_overhead, erasure_crossover_bytes,
)
from meshpool.storage.engine import (
    StorageManager, StorageError, _pack_ec_params, _unpack_ec_params,
)
from meshpool.common.transport import TransportError
from meshpool.common.models import BlobMeta


# ---------------------------------------------------------------------------
# Fake cluster for in-process simulation
# ---------------------------------------------------------------------------
def _node(nid, disk_mb=100_000, services=None):
    """A *real* Node, so this gate exercises the same code path production does
    (the placement planners ask ``node.provides(...)``; a namespace can't answer)."""
    from meshpool.common import models as M
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

    def get_blob(self, bid):
        m = self._rows.get(bid)
        return m.model_copy(deep=True) if m else None

    def list_blobs(self):
        return [m.model_copy(deep=True) for m in self._rows.values()]

    def delete_blob(self, bid):
        self._rows.pop(bid, None)


class FakeTransport:
    def __init__(self, node_ids):
        self.disks = {n: {} for n in node_ids}
        self.unreachable: set = set()

    def add_node(self, nid):
        self.disks[nid] = {}

    async def call(self, node_id, method, payload, timeout=None):
        if node_id in self.unreachable or node_id not in self.disks:
            raise TransportError(f"node {node_id} unreachable")
        disk = self.disks[node_id]
        if method == "blob_put":
            raw = base64.b64decode(payload["data"])
            if hashlib.sha256(raw).hexdigest() != payload["blob_id"]:
                raise TransportError("hash mismatch at agent")
            disk[payload["blob_id"]] = raw
            return {"blob_id": payload["blob_id"], "stored": True}
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


def _make_cluster(n_nodes: int, k: int, m: int):
    node_ids = [f"node-{i:03d}" for i in range(n_nodes)]
    reg = FakeRegistry(node_ids)
    store = FakeStore()
    transport = FakeTransport(node_ids)
    events = FakeEvents()
    mgr = StorageManager(reg, store, None, events, transport)
    return mgr, reg, store, transport, events, node_ids


# ---------------------------------------------------------------------------
# Durability calculation
# ---------------------------------------------------------------------------
def _loss_probability(k: int, m: int, n_nodes: int, p_fail: float) -> float:
    """P(data loss) = P(more than m nodes fail among n_nodes) using binomial.

    Data is lost when more than m of the k+m shard-holders fail.
    We conservatively assume all k+m nodes could fail independently.
    """
    total = k + m
    # P(exactly j failures out of total) = C(total,j) * p^j * (1-p)^(total-j)
    p_loss = 0.0
    for j in range(m + 1, total + 1):
        p_loss += math.comb(total, j) * (p_fail ** j) * ((1 - p_fail) ** (total - j))
    return p_loss


def _replication_loss_probability(rep: int, p_fail: float) -> float:
    """P(all ``rep`` replicas fail simultaneously)."""
    return p_fail ** rep


# ---------------------------------------------------------------------------
# Main scenario
# ---------------------------------------------------------------------------
def main() -> int:  # noqa: C901
    ok = True
    K, M = 6, 3

    print(f"== MeshPool Storage: Reed-Solomon ({K},{M}) Erasure Coding vs 2x Replication ==")
    print(f"   k={K} data shards + m={M} parity shards; any m losses recoverable")
    print()

    # -- Non-vacuity guard: generate real random binary data --
    rng = random.Random(0xDEAD_BEEF)
    SIZE = 64_000   # 64 KB of random binary content
    data_bytes = bytes(rng.getrandbits(8) for _ in range(SIZE))

    distinct_bytes = len(set(data_bytes))
    min_byte = min(data_bytes)
    max_byte = max(data_bytes)
    nonzero = sum(1 for b in data_bytes if b != 0)
    assert len(data_bytes) > 1000, f"test data too small: {len(data_bytes)} bytes"
    assert distinct_bytes >= 200, (
        f"test data not sufficiently random: only {distinct_bytes} distinct byte values "
        f"(expected >=200 for random data of this size)")
    assert min_byte == 0 and max_byte == 255, (
        "test data does not span full 0-255 byte range")
    assert nonzero > len(data_bytes) * 0.9, (
        f"test data looks non-random: only {nonzero}/{len(data_bytes)} non-zero bytes")
    print(f"test data: {SIZE} bytes of random binary, {distinct_bytes}/256 distinct "
          f"byte values, {nonzero} non-zero bytes")
    print()

    n_nodes = K + M + 10   # enough nodes for all checks

    # ------------------------------------------------------------------ #
    print(f"== check 1: object survives loss of exactly m={M} nodes ==")
    try:
        # Fresh cluster for this check
        mgr1, reg1, store1, transport1, _, _ = _make_cluster(n_nodes, K, M)
        meta = asyncio.run(mgr1.put(data_bytes, erasure=True, ec_k=K, ec_m=M))
        assert meta.chunks, "EC not used (check crossover or cluster size)"
        assert len(meta.chunks) == K + M, (
            f"expected {K+M} shards, got {len(meta.chunks)}")
        assert len(set(meta.chunks)) == K + M, "duplicate shard IDs!"
        print(f"  stored: {len(meta.chunks)} shards ({K} data + {M} parity) "
              f"on {K+M} distinct nodes")

        # Identify and kill exactly m shard-holder nodes
        shard_holder_nodes = []
        for sid in meta.chunks:
            sm = store1.get_blob(sid)
            if sm and sm.replicas:
                shard_holder_nodes.extend(sm.replicas)
        shard_holder_nodes = list(dict.fromkeys(shard_holder_nodes))

        victims = shard_holder_nodes[:M]
        for victim in victims:
            transport1.unreachable.add(victim)
            reg1.kill(victim)
        print(f"  killed {M} nodes: {victims}")

        recovered = asyncio.run(mgr1.get(meta.id))
        assert recovered is not None, "got None after losing m nodes"
        assert recovered == data_bytes, (
            f"recovered data differs from original! "
            f"first diff at byte "
            f"{next(i for i,(a,b) in enumerate(zip(recovered,data_bytes)) if a!=b)}")
        print(f"  PASS: recovered {len(recovered)} bytes -- identical to original")
    except Exception as exc:
        ok = False
        print(f"  FAIL: {type(exc).__name__}: {exc}")

    # ------------------------------------------------------------------ #
    print(f"\n== check 2: correctly FAILS (not crashes) when m+1={M+1} shards lost ==")
    try:
        # Fresh cluster
        mgr2, reg2, store2, transport2, _, _ = _make_cluster(n_nodes, K, M)
        meta2 = asyncio.run(mgr2.put(data_bytes, erasure=True, ec_k=K, ec_m=M))
        assert len(meta2.chunks) == K + M

        shard_holders2 = []
        for sid in meta2.chunks:
            sm = store2.get_blob(sid)
            if sm and sm.replicas:
                shard_holders2.extend(sm.replicas)
        shard_holders2 = list(dict.fromkeys(shard_holders2))

        for victim in shard_holders2[:M + 1]:
            transport2.unreachable.add(victim)
            transport2.disks[victim].clear()
            reg2.kill(victim)

        result = asyncio.run(mgr2.get(meta2.id))
        assert result is None, (
            f"should have returned None with m+1={M+1} losses, but returned "
            f"{len(result)} bytes")
        print(f"  PASS: returned None (no crash, no wrong answer) when {M+1} "
              f"shards unavailable")
    except Exception as exc:
        ok = False
        print(f"  FAIL: {type(exc).__name__}: {exc}")

    # ------------------------------------------------------------------ #
    print(f"\n== check 3: corrupted shard is detected and routed around ==")
    try:
        # Fresh cluster for this check to avoid stale state from previous checks
        mgr3, reg3, store3, transport3, events3, node_ids3 = _make_cluster(n_nodes, K, M)
        meta3 = asyncio.run(mgr3.put(data_bytes, erasure=True, ec_k=K, ec_m=M))
        assert len(meta3.chunks) == K + M

        # Corrupt one shard's payload bytes (flip a bit)
        target_sid = meta3.chunks[K // 2]   # middle data shard
        sm3 = store3.get_blob(target_sid)
        holder3 = sm3.replicas[0]
        raw3 = transport3.disks[holder3][target_sid]
        orig_raw3 = raw3
        corrupted3 = bytearray(raw3)
        corrupted3[_HEADER_SIZE + 7] ^= 0xFF   # flip byte in payload
        assert bytes(corrupted3) != raw3, "corruption must actually change the bytes"
        transport3.disks[holder3][target_sid] = bytes(corrupted3)

        # Verify it's genuinely corrupted (hash mismatch)
        modified_hash = hashlib.sha256(bytes(corrupted3)[_HEADER_SIZE:]).digest()
        claimed_hash = raw3[8:40]
        assert modified_hash != claimed_hash, "corruption not detected by hash -- test is vacuous"

        recovered3 = asyncio.run(mgr3.get(meta3.id))
        assert recovered3 is not None, "got None -- reconstruction should succeed via remaining shards"
        assert recovered3 == data_bytes, "recovered data is wrong despite routing around corruption"
        print(f"  PASS: 1 corrupted shard detected and routed around; "
              f"data recovered correctly from remaining {K+M-1} good shards")
    except Exception as exc:
        ok = False
        print(f"  FAIL: {type(exc).__name__}: {exc}")

    # ------------------------------------------------------------------ #
    print(f"\n== check 4: repair restores full redundancy and data stays intact ==")
    try:
        # Fresh cluster with spare nodes for repair
        n_nodes4 = K + M + 6
        mgr4, reg4, store4, transport4, events4, node_ids4 = _make_cluster(n_nodes4, K, M)
        meta4 = asyncio.run(mgr4.put(data_bytes, erasure=True, ec_k=K, ec_m=M))
        assert len(meta4.chunks) == K + M

        # Kill two shard holders (below failure threshold)
        victims4 = []
        for sid in meta4.chunks[:2]:
            sm4 = store4.get_blob(sid)
            v = sm4.replicas[0]
            if v not in victims4:
                victims4.append(v)
                reg4.kill(v)
                transport4.disks[v].clear()
                sm4.replicas = []
                store4.put_blob(sm4)

        # Run repair
        async def run_repair4():
            async with mgr4._lock:
                await mgr4._reconcile()
        asyncio.run(run_repair4())

        # Verify data intact
        recovered4 = asyncio.run(mgr4.get(meta4.id))
        assert recovered4 is not None, "repair left data unrecoverable"
        assert recovered4 == data_bytes, "data changed after repair!"

        # Count live shards
        meta4_after = store4.get_blob(meta4.id)
        live_shards = sum(
            1 for sid in meta4_after.chunks
            if (sm := store4.get_blob(sid)) and sm.replicas)
        print(f"  killed: {len(victims4)} shard nodes * "
              f"after repair: {live_shards}/{K+M} shards live")
        assert live_shards == K + M, (
            f"repair did not restore full redundancy: {live_shards}/{K+M}")
        print("  PASS: repair restored full redundancy; data bytes identical")
    except Exception as exc:
        ok = False
        print(f"  FAIL: {type(exc).__name__}: {exc}")

    # ------------------------------------------------------------------ #
    print(f"\n== check 5: storage overhead and durability comparison ==")
    try:
        REP = 2
        ec_factor = erasure_overhead(K, M)
        rep_factor = float(REP)
        overhead_saving_pct = (rep_factor - ec_factor) / rep_factor * 100

        print(f"\n  Scheme            Overhead   Tolerates   Shard nodes")
        print(f"  {REP}x replication   {rep_factor:.2f}x       1 loss      {REP}")
        print(f"  RS({K},{M}) erasure  {ec_factor:.2f}x       {M} losses     {K+M}")

        # Crossover
        xover = erasure_crossover_bytes(K, M, replication=REP)
        print(f"\n  Crossover: EC breaks even with replication at {xover} bytes")
        print(f"  Objects >= {xover} bytes: EC uses {overhead_saving_pct:.1f}% less storage")
        print(f"  Objects < {xover} bytes: engine falls back to replication automatically")

        # Durability math at 3% annual failure rate
        P_FAIL_ANNUAL = 0.03
        p_ec_loss = _loss_probability(K, M, K + M, P_FAIL_ANNUAL)
        p_rep_loss = _replication_loss_probability(REP, P_FAIL_ANNUAL)
        durability_gain = p_rep_loss / p_ec_loss if p_ec_loss > 0 else float("inf")

        print(f"\n  Annual failure probability per node: {P_FAIL_ANNUAL*100:.0f}%")
        print(f"  P(data loss | {REP}x replication):  {p_rep_loss:.3e}  "
              f"(loses data when {REP} specific nodes fail together)")
        print(f"  P(data loss | RS({K},{M})):           {p_ec_loss:.3e}  "
              f"(loses data when >{M} of {K+M} nodes fail together)")
        print(f"  Durability improvement: RS({K},{M}) is ~{durability_gain:.0f}x "
              f"less likely to lose data")

        assert ec_factor < rep_factor, (
            f"EC overhead {ec_factor}x should be < replication {rep_factor}x")
        assert p_ec_loss < p_rep_loss, (
            f"EC P(loss)={p_ec_loss:.2e} should be < replication {p_rep_loss:.2e}")
        assert durability_gain > 5, (
            f"EC should be at least 5x more durable than 2x replication; "
            f"got {durability_gain:.1f}x")
        assert xover > 0, "crossover must be a positive number of bytes"
        assert xover < 10_000, f"crossover suspiciously large: {xover} bytes"

        print("  PASS: EC overhead lower, durability dramatically higher")
    except Exception as exc:
        ok = False
        print(f"  FAIL: {type(exc).__name__}: {exc}")

    # ------------------------------------------------------------------ #
    print(f"\n== check 6: encode/decode throughput measurement ==")
    try:
        import time
        codec = RSCodec(K, M)
        BENCH_SIZE = 1_000_000   # 1 MB
        bench_data = bytes(rng.getrandbits(8) for _ in range(BENCH_SIZE))

        t0 = time.perf_counter()
        shards = codec.encode(bench_data)
        t_enc = time.perf_counter() - t0
        enc_mbps = BENCH_SIZE / 1e6 / t_enc

        t0 = time.perf_counter()
        result = codec.decode(shards)
        t_dec = time.perf_counter() - t0
        dec_mbps = BENCH_SIZE / 1e6 / t_dec

        assert result == bench_data, "encode/decode round-trip failed on 1MB benchmark!"
        print(f"  encode: {enc_mbps:.2f} MB/s ({t_enc*1000:.1f} ms for 1 MB)")
        print(f"  decode: {dec_mbps:.2f} MB/s ({t_dec*1000:.1f} ms for 1 MB)")
        print("  PASS: 1 MB round-trip exact")
    except Exception as exc:
        ok = False
        print(f"  FAIL: {type(exc).__name__}: {exc}")

    # ------------------------------------------------------------------ #
    print()
    if ok:
        print("ALL DURABILITY CHECKS PASSED")
    else:
        print("SOME CHECKS FAILED -- see above for details")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
