"""Unit tests for proof-of-storage audits, repair, and escrow.

Five groups of tests:
  1. Random-offset challenge: a challenge issued for chunk N verifies correctly
     against the expected chunk hash.
  2. Memorised-answer cheater: a node that always returns the hash of chunk 0
     fails any challenge for a different chunk index.
  3. Repair: when an audit fails, the shard is dropped from the node's placement
     set, and reconcile re-places it on a surviving node, restoring full durability.
  4. Escrow held then released: storage credits are held in escrow and converted
     to "donated" ledger entries only after N consecutive passing audits.
  5. Escrow forfeited on a failed audit: a failed audit resets the consecutive
     pass streak and forfeits all held escrow (no credit granted).
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import random
import types
from typing import Optional

import pytest

from meshpool.common.db import Store
from meshpool.common.models import BlobMeta, now_ms
from meshpool.common.transport import TransportError
from meshpool.common.trust import ContributionLedger
from meshpool.storage.audit import (
    AUDIT_CHUNK_SIZE,
    CONSECUTIVE_PASSES_TO_RELEASE,
    ESCROW_PER_AUDIT_UNIT,
    StorageAuditEngine,
)
from meshpool.storage.engine import StorageManager


# ---------------------------------------------------------------------------
# Shared fakes (copied from test_storage.py so this file is self-contained)
# ---------------------------------------------------------------------------
def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _node(nid: str, disk_mb: int = 100_000):
    from meshpool.common import models as M
    cap = M.Capability(cpu_cores=4, ram_mb=4096, disk_mb=disk_mb)
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
    """In-memory cluster: one blob dict per node, verifying digests like a real agent.

    Set ``audit_fn`` to override how the node responds to blob_audit challenges
    (used to simulate an honest or dishonest node in audit tests).
    """

    def __init__(self, node_ids, unreachable=None):
        self.disks: dict[str, dict[str, bytes]] = {n: {} for n in node_ids}
        self.unreachable: set = set(unreachable or ())
        self.calls: list = []
        # Override: node_id -> fn(challenge_dict) -> response_dict
        self.audit_overrides: dict[str, object] = {}

    def add_node(self, nid):
        self.disks[nid] = {}

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
        if method == "blob_audit":
            # Use node-specific override if set, else compute honest response.
            override = self.audit_overrides.get(node_id)
            if override is not None:
                return override(payload)
            # Honest response: SHA-256 of the challenged chunk.
            shard_id = payload["shard_id"]
            chunk_index = payload["chunk_index"]
            chunk_size = payload["chunk_size"]
            raw = disk.get(shard_id)
            if raw is None:
                return {"challenge_id": payload["challenge_id"], "hash": ""}
            start = chunk_index * chunk_size
            end = start + chunk_size
            chunk = raw[start:end]
            if not chunk:
                return {"challenge_id": payload["challenge_id"], "hash": ""}
            h = hashlib.sha256(chunk).hexdigest()
            return {"challenge_id": payload["challenge_id"], "hash": h}
        raise TransportError(f"unknown method {method}")


class FakeEvents:
    def __init__(self):
        self.events = []

    def publish(self, kind, data):
        self.events.append((kind, data))


def _cluster(node_ids, k=6, m=3, unreachable=None):
    reg = FakeRegistry(node_ids)
    store = FakeStore()
    transport = FakeTransport(node_ids, unreachable=unreachable)
    events = FakeEvents()
    return reg, store, transport, events


def _manager_with_audit(node_ids, k=6, m=3, rng=None, unreachable=None):
    reg, store, transport, events = _cluster(node_ids, k, m, unreachable)
    store_obj = Store(":memory:")
    ledger = ContributionLedger(store_obj)
    audit = StorageAuditEngine(ledger, rng=rng)
    mgr = StorageManager(reg, store, ledger, events, transport, audit=audit)
    return mgr, reg, store, transport, events, audit, ledger


# ---------------------------------------------------------------------------
# Group 1: Random-offset challenge (correct chunk passes, wrong chunk fails)
# ---------------------------------------------------------------------------

class TestRandomOffsetChallenge:
    """The challenge asks for a RANDOM chunk so the coordinator can verify without
    the full shard bytes, and a node that memorised one specific answer fails any
    challenge for a different chunk index."""

    def test_correct_chunk_passes(self, tmp_path):
        # Build a shard with multiple auditable chunks.
        rng = random.Random(42)
        shard_size = AUDIT_CHUNK_SIZE * 5   # 5 chunks
        shard_data = bytes(rng.getrandbits(8) for _ in range(shard_size))
        shard_id = _sha(shard_data)

        store = Store(str(tmp_path / "test.db"))
        ledger = ContributionLedger(store)
        engine = StorageAuditEngine(ledger)
        engine.record_shard_hashes(shard_id, shard_data)

        # Issue a challenge for a specific chunk and manually verify
        # that the coordinator's stored hash matches the actual bytes.
        for chunk_index in range(5):
            challenge = engine.issue_challenge("node-A", shard_id)
            assert challenge is not None
            # Simulate the honest node computing the response.
            actual_ci = challenge["chunk_index"]
            start = actual_ci * AUDIT_CHUNK_SIZE
            end = start + AUDIT_CHUNK_SIZE
            response = hashlib.sha256(shard_data[start:end]).hexdigest()
            # Coordinator verifies.
            assert engine.verify_response(challenge["challenge_id"], response), (
                f"correct chunk hash for index {actual_ci} should verify")

    def test_wrong_hash_fails(self, tmp_path):
        rng = random.Random(99)
        shard_data = bytes(rng.getrandbits(8) for _ in range(AUDIT_CHUNK_SIZE * 4))
        shard_id = _sha(shard_data)

        store = Store(str(tmp_path / "test.db"))
        ledger = ContributionLedger(store)
        engine = StorageAuditEngine(ledger)
        engine.record_shard_hashes(shard_id, shard_data)

        challenge = engine.issue_challenge("node-B", shard_id)
        assert challenge is not None
        bad_response = hashlib.sha256(b"wrong bytes").hexdigest()
        assert not engine.verify_response(challenge["challenge_id"], bad_response), (
            "wrong hash must not verify")

    def test_challenge_is_consumed_on_first_use(self, tmp_path):
        """Replay protection: a challenge can only be verified once."""
        shard_data = bytes(range(256)) * (AUDIT_CHUNK_SIZE // 256)
        shard_id = _sha(shard_data)

        store = Store(str(tmp_path / "test.db"))
        ledger = ContributionLedger(store)
        engine = StorageAuditEngine(ledger)
        engine.record_shard_hashes(shard_id, shard_data)

        challenge = engine.issue_challenge("node-C", shard_id)
        ci = challenge["chunk_index"]
        start = ci * AUDIT_CHUNK_SIZE
        response = hashlib.sha256(shard_data[start:start + AUDIT_CHUNK_SIZE]).hexdigest()

        assert engine.verify_response(challenge["challenge_id"], response)
        # Second call with the same challenge_id must fail (challenge consumed).
        assert not engine.verify_response(challenge["challenge_id"], response), (
            "challenge must be one-shot — replay should fail")

    def test_no_material_returns_none_challenge(self, tmp_path):
        """issue_challenge returns None for a shard with no recorded material."""
        store = Store(str(tmp_path / "test.db"))
        ledger = ContributionLedger(store)
        engine = StorageAuditEngine(ledger)
        challenge = engine.issue_challenge("node-X", "a" * 64)
        assert challenge is None


# ---------------------------------------------------------------------------
# Group 2: Memorised-answer cheater fails a fresh challenge
# ---------------------------------------------------------------------------

class TestMemorisedAnswerCheater:
    """A node that memorised only chunk 0's hash fails any challenge for chunk N > 0.

    This is the core non-vacuity property: a fixed challenge would let a cheater
    delete the bytes and store just one hash; random selection defeats that.
    """

    def _audit_with_cheater(self, shard_data: bytes, memorised_chunk_index: int,
                             tmp_path) -> tuple[bool, int]:
        """Run one audit against a cheater that always returns hash of chunk ``memorised_chunk_index``.

        Returns (passed, challenged_chunk_index).
        """
        shard_id = _sha(shard_data)
        store = Store(str(tmp_path / "test.db"))
        ledger = ContributionLedger(store)
        # Use a seeded RNG so the test is deterministic but covers different chunks.
        engine = StorageAuditEngine(ledger)
        engine.record_shard_hashes(shard_id, shard_data)

        # The cheater always returns the hash of chunk ``memorised_chunk_index``.
        start = memorised_chunk_index * AUDIT_CHUNK_SIZE
        memorised_hash = hashlib.sha256(shard_data[start:start + AUDIT_CHUNK_SIZE]).hexdigest()

        challenge = engine.issue_challenge("cheater", shard_id)
        assert challenge is not None
        challenged_index = challenge["chunk_index"]
        # Cheater ignores the challenged index and returns memorised hash.
        passed = engine.verify_response(challenge["challenge_id"], memorised_hash)
        return passed, challenged_index

    def test_cheater_fails_when_asked_different_chunk(self, tmp_path):
        """A cheater that memorised chunk 0 fails any challenge for chunk != 0."""
        rng = random.Random(2024)
        n_chunks = 8
        shard_data = bytes(rng.getrandbits(8) for _ in range(AUDIT_CHUNK_SIZE * n_chunks))
        memorised_index = 0

        # Run multiple challenges with different seeds until we get a challenge != 0.
        failed_at_least_once = False
        for seed in range(100):
            engine_rng = random.Random(seed)
            shard_id = _sha(shard_data)
            store = Store(str(tmp_path / f"db_{seed}.db"))
            ledger = ContributionLedger(store)
            engine = StorageAuditEngine(ledger, rng=engine_rng)
            engine.record_shard_hashes(shard_id, shard_data)

            start = memorised_index * AUDIT_CHUNK_SIZE
            memorised_hash = hashlib.sha256(
                shard_data[start:start + AUDIT_CHUNK_SIZE]).hexdigest()

            challenge = engine.issue_challenge("cheater", shard_id)
            assert challenge is not None
            if challenge["chunk_index"] != memorised_index:
                passed = engine.verify_response(challenge["challenge_id"], memorised_hash)
                assert not passed, (
                    f"Cheater should FAIL when asked for chunk {challenge['chunk_index']} "
                    f"but submitted answer for chunk {memorised_index}")
                failed_at_least_once = True
                break

        assert failed_at_least_once, (
            "Expected at least one challenge for chunk != 0 in 100 seeds (random should vary)")

    def test_correct_chunk_passes_same_cheater(self, tmp_path):
        """Baseline: the same node passes when it happens to know the right answer."""
        rng = random.Random(7)
        shard_data = bytes(rng.getrandbits(8) for _ in range(AUDIT_CHUNK_SIZE * 3))
        shard_id = _sha(shard_data)

        # Use a fixed-seed RNG that will produce chunk_index=0 deterministically.
        fixed_rng = random.Random(0)
        store = Store(str(tmp_path / "test.db"))
        ledger = ContributionLedger(store)
        engine = StorageAuditEngine(ledger, rng=fixed_rng)
        engine.record_shard_hashes(shard_id, shard_data)

        challenge = engine.issue_challenge("lucky_cheater", shard_id)
        ci = challenge["chunk_index"]
        # "Lucky" cheater memorised the right chunk by coincidence.
        start = ci * AUDIT_CHUNK_SIZE
        correct_hash = hashlib.sha256(shard_data[start:start + AUDIT_CHUNK_SIZE]).hexdigest()
        assert engine.verify_response(challenge["challenge_id"], correct_hash)

    def test_cheater_with_canned_hash_fails_end_to_end_audit(self, tmp_path):
        """End-to-end: a cheater whose FakeTransport always returns chunk 0's hash
        fails audit_shard when the challenge asks for a chunk != 0.
        """
        rng = random.Random(2025)
        n_chunks = 6
        shard_data = bytes(rng.getrandbits(8) for _ in range(AUDIT_CHUNK_SIZE * n_chunks))
        node_ids = [f"n{i}" for i in range(9)]   # k=6 + m=3

        # Set up full manager with audit engine.
        mgr, reg, store, transport, events, audit, ledger = _manager_with_audit(
            node_ids, k=6, m=3, rng=random.Random(0))

        # Store a blob that produces EC shards.
        data = bytes(rng.getrandbits(8) for _ in range(4096 * 7))  # > EC crossover
        meta = asyncio.run(mgr.put(data, erasure=True, ec_k=6, ec_m=3))
        assert meta.chunks, "Must have EC shards"

        # Pick one shard and its node; make that node a cheater.
        first_shard_id = meta.chunks[0]
        first_shard_meta = store.get_blob(first_shard_id)
        cheater_node = first_shard_meta.replicas[0]

        # Cheater always returns hash of chunk 0, ignoring the challenged index.
        raw_shard = transport.disks[cheater_node][first_shard_id]
        memorised_hash = hashlib.sha256(raw_shard[:AUDIT_CHUNK_SIZE]).hexdigest()

        fail_count = 0
        pass_count = 0
        # Run multiple challenges; the cheater should fail for non-zero chunk indices.
        for _ in range(20):
            # Override: always return memorised_hash regardless of chunk_index.
            transport.audit_overrides[cheater_node] = (
                lambda p, _h=memorised_hash: {"challenge_id": p["challenge_id"], "hash": _h})

            result = asyncio.run(
                audit.audit_shard(
                    cheater_node, first_shard_id, meta.id, mgr, transport))
            if result:
                pass_count += 1
            else:
                fail_count += 1

        # The cheater MUST fail at least once in 20 challenges
        # (the probability that all 20 random indices equal 0 is < 0.006%)
        assert fail_count >= 1, (
            f"Cheater returned memorised chunk-0 hash and should have failed "
            f"at least once in 20 challenges (passed={pass_count}, failed={fail_count})")


# ---------------------------------------------------------------------------
# Group 3: Repair restores durability after audit failure
# ---------------------------------------------------------------------------

class TestRepairAfterAuditFail:
    """on_audit_fail drops the node from the shard's placement and repair
    (reconcile) re-places the shard onto a surviving node."""

    def test_audit_fail_drops_node_from_shard(self, tmp_path):
        rng = random.Random(5)
        node_ids = [f"n{i}" for i in range(9)]
        mgr, reg, store, transport, events, audit, ledger = _manager_with_audit(
            node_ids, rng=rng)

        data = bytes(rng.getrandbits(8) for _ in range(4096 * 7))
        meta = asyncio.run(mgr.put(data, erasure=True, ec_k=6, ec_m=3))
        assert meta.chunks

        target_shard = meta.chunks[0]
        shard_meta = store.get_blob(target_shard)
        bad_node = shard_meta.replicas[0]
        assert bad_node in {n.id for n in reg.online()}

        audit.on_audit_fail(bad_node, target_shard, meta.id, mgr)

        updated = store.get_blob(target_shard)
        assert bad_node not in updated.replicas, (
            "audit_fail must remove the failing node from the shard's replica list")

    def test_repair_restores_shard_after_audit_fail(self, tmp_path):
        """After audit_fail drops a shard, reconcile places it on a new node."""
        rng = random.Random(11)
        node_ids = [f"n{i}" for i in range(12)]   # extra spare nodes for repair
        mgr, reg, store, transport, events, audit, ledger = _manager_with_audit(
            node_ids, rng=rng)

        data = bytes(rng.getrandbits(8) for _ in range(4096 * 7))
        meta = asyncio.run(mgr.put(data, erasure=True, ec_k=6, ec_m=3))

        # Fail one shard's audit (not killing the node, just dropping the shard)
        target_shard = meta.chunks[0]
        shard_meta = store.get_blob(target_shard)
        bad_node = shard_meta.replicas[0]
        audit.on_audit_fail(bad_node, target_shard, meta.id, mgr)

        # Run reconcile (the repair loop)
        async def run_reconcile():
            async with mgr._lock:
                await mgr._reconcile()
        asyncio.run(run_reconcile())

        # Verify the shard is now held by a different node
        after_shard_meta = store.get_blob(target_shard)
        # The shard ID may have changed because repair re-encodes; check blob health.
        after_meta = store.get_blob(meta.id)
        k, m = 6, 3
        total = k + m
        live_shards = sum(
            1 for sid in after_meta.chunks
            if (sm := store.get_blob(sid)) and sm.replicas)
        assert live_shards >= k, (
            f"Repair should have restored at least k={k} shards; "
            f"got {live_shards}/{total}")

        # Data must still be recoverable
        recovered = asyncio.run(mgr.get(meta.id))
        assert recovered is not None, "Data must still be recoverable after repair"
        assert recovered == data, "Recovered data must be identical to original"

    def test_repair_is_anti_thrash(self, tmp_path):
        """Repair cooldown prevents the same blob from being repaired in a tight loop."""
        rng = random.Random(17)
        node_ids = [f"n{i}" for i in range(9)]
        mgr, reg, store, transport, events, audit, ledger = _manager_with_audit(
            node_ids, rng=rng)
        audit._repair_cooldown_s = 3600.0  # 1 hour — should not fire again

        data = bytes(rng.getrandbits(8) for _ in range(4096 * 7))
        meta = asyncio.run(mgr.put(data, erasure=True, ec_k=6, ec_m=3))

        target_shard = meta.chunks[0]
        shard_meta = store.get_blob(target_shard)
        bad_node = shard_meta.replicas[0]

        # First fail: should publish the repair event
        audit.on_audit_fail(bad_node, target_shard, meta.id, mgr)
        first_repair_events = [e for e in events.events if e[0] == "audit_fail_repair"]

        # Second fail: should NOT publish another repair event (cooldown active)
        audit.on_audit_fail(bad_node, target_shard, meta.id, mgr)
        second_repair_events = [e for e in events.events if e[0] == "audit_fail_repair"]

        assert len(first_repair_events) == 1, "First fail must trigger repair event"
        assert len(second_repair_events) == 1, (
            "Second fail within cooldown must NOT trigger another repair event")


# ---------------------------------------------------------------------------
# Group 4: Escrow held then released after N consecutive passes
# ---------------------------------------------------------------------------

class TestEscrowHeldThenReleased:
    """Storage credits are held in escrow and released to the ledger only after
    CONSECUTIVE_PASSES_TO_RELEASE consecutive passing audits."""

    def test_escrow_held_before_threshold(self, tmp_path):
        store = Store(str(tmp_path / "db.db"))
        ledger = ContributionLedger(store)
        engine = StorageAuditEngine(ledger)

        node_id = "honest-node"
        shard_id = "a" * 64

        # Pass CONSECUTIVE_PASSES_TO_RELEASE - 1 audits (not yet enough to release)
        for _ in range(CONSECUTIVE_PASSES_TO_RELEASE - 1):
            released = engine.on_audit_pass(node_id, shard_id)
            assert released == 0.0, "No release expected before threshold"

        # Escrow is held but NOT yet in the ledger as donated credits
        assert engine.total_held_escrow() > 0
        state = engine.escrow_state(node_id)
        assert state["held_escrow"] > 0
        assert state["consecutive_passes"] == CONSECUTIVE_PASSES_TO_RELEASE - 1
        # Ledger should show no donated credits yet
        assert ledger.credits(node_id) == 0.0 or ledger.credits(node_id) < ESCROW_PER_AUDIT_UNIT

    def test_escrow_released_at_threshold(self, tmp_path):
        store = Store(str(tmp_path / "db.db"))
        ledger = ContributionLedger(store)
        engine = StorageAuditEngine(ledger)

        node_id = "honest-node"
        shard_id = "b" * 64

        released_total = 0.0
        for i in range(CONSECUTIVE_PASSES_TO_RELEASE):
            released = engine.on_audit_pass(node_id, shard_id, owner=node_id)
            released_total += released

        # After exactly CONSECUTIVE_PASSES_TO_RELEASE passes, escrow should be released
        expected = ESCROW_PER_AUDIT_UNIT * CONSECUTIVE_PASSES_TO_RELEASE
        assert abs(released_total - expected) < 1e-9, (
            f"Expected {expected} credits released, got {released_total}")

        # Credits appear in the existing ledger as "donated"
        credits = ledger.credits(node_id)
        assert abs(credits - expected) < 1e-9, (
            f"Ledger credits should match released escrow: "
            f"expected {expected}, got {credits}")

        # State is reset for the next cycle
        state = engine.escrow_state(node_id)
        assert state["consecutive_passes"] == 0
        assert state["held_escrow"] == 0.0

    def test_escrow_accumulates_over_multiple_cycles(self, tmp_path):
        """After release, the next audit cycle starts a new escrow accumulation."""
        store = Store(str(tmp_path / "db.db"))
        ledger = ContributionLedger(store)
        engine = StorageAuditEngine(ledger)

        node_id = "multi-cycle"
        shard_id = "c" * 64
        N = CONSECUTIVE_PASSES_TO_RELEASE
        n_cycles = 3

        for cycle in range(n_cycles):
            for i in range(N):
                engine.on_audit_pass(node_id, shard_id, owner=node_id)

        # After n_cycles × N passes, total donated = n_cycles × N × ESCROW_PER_AUDIT_UNIT
        expected = n_cycles * N * ESCROW_PER_AUDIT_UNIT
        credits = ledger.credits(node_id)
        assert abs(credits - expected) < 1e-9, (
            f"Expected {expected} total donated credits over {n_cycles} cycles, "
            f"got {credits}")

    def test_end_to_end_audit_pass_releases_escrow(self, tmp_path):
        """Full audit cycle via audit_shard: honest node passes and escrow releases."""
        rng = random.Random(42)
        node_ids = [f"n{i}" for i in range(9)]
        mgr, reg, store, transport, events, audit, ledger = _manager_with_audit(
            node_ids, rng=rng)

        data = bytes(rng.getrandbits(8) for _ in range(4096 * 7))
        meta = asyncio.run(mgr.put(data, erasure=True, ec_k=6, ec_m=3))
        assert meta.chunks

        first_shard = meta.chunks[0]
        shard_meta = store.get_blob(first_shard)
        node_id = shard_meta.replicas[0]

        # Run enough passing audits to trigger release
        N = CONSECUTIVE_PASSES_TO_RELEASE
        for _ in range(N):
            result = asyncio.run(
                audit.audit_shard(node_id, first_shard, meta.id, mgr, transport,
                                  owner=node_id))
            assert result is True, "Honest node should pass audit"

        # Credits released into ledger
        credits = ledger.credits(node_id)
        expected = ESCROW_PER_AUDIT_UNIT * N
        assert credits >= expected, (
            f"Expected at least {expected} credits; got {credits}")


# ---------------------------------------------------------------------------
# Group 5: Escrow forfeited on a failed audit
# ---------------------------------------------------------------------------

class TestEscrowForfeitedOnFail:
    """A failed audit resets the pass streak and forfeits all held escrow."""

    def test_fail_resets_consecutive_passes(self, tmp_path):
        store = Store(str(tmp_path / "db.db"))
        ledger = ContributionLedger(store)
        engine = StorageAuditEngine(ledger)

        node_id = "intermittent-node"
        shard_id = "d" * 64

        # Build up some passes
        for _ in range(CONSECUTIVE_PASSES_TO_RELEASE - 1):
            engine.on_audit_pass(node_id, shard_id)

        state_before = engine.escrow_state(node_id)
        assert state_before["consecutive_passes"] > 0
        assert state_before["held_escrow"] > 0.0

        # Fake storage manager with empty store
        fake_sm = types.SimpleNamespace(
            store=FakeStore(),
            events=FakeEvents(),
        )

        forfeited = engine.on_audit_fail(node_id, shard_id, shard_id, fake_sm)

        assert abs(forfeited - state_before["held_escrow"]) < 1e-9, (
            f"All held escrow must be forfeited on fail: "
            f"expected {state_before['held_escrow']}, got {forfeited}")

        state_after = engine.escrow_state(node_id)
        assert state_after["consecutive_passes"] == 0
        assert state_after["held_escrow"] == 0.0

        # No credits appear in the ledger (escrow forfeited = no donation)
        assert ledger.credits(node_id) == 0.0, (
            "Forfeited escrow must not appear as donated credits in the ledger")

    def test_fail_after_full_cycle_forfeits_only_new_escrow(self, tmp_path):
        """Credits from a completed cycle are already in the ledger; only the
        new (post-release) escrow is forfeited on the next fail."""
        store = Store(str(tmp_path / "db.db"))
        ledger = ContributionLedger(store)
        engine = StorageAuditEngine(ledger)

        node_id = "then-failed"
        shard_id = "e" * 64
        N = CONSECUTIVE_PASSES_TO_RELEASE
        fake_sm = types.SimpleNamespace(store=FakeStore(), events=FakeEvents())

        # Complete one release cycle
        for _ in range(N):
            engine.on_audit_pass(node_id, shard_id, owner=node_id)
        released_credits = ledger.credits(node_id)
        assert released_credits > 0.0, "First cycle must release credits"

        # Partial second cycle
        for _ in range(N - 1):
            engine.on_audit_pass(node_id, shard_id, owner=node_id)
        held_before_fail = engine.escrow_state(node_id)["held_escrow"]
        assert held_before_fail > 0.0

        # Fail: forfeit only the new (partial) escrow
        forfeited = engine.on_audit_fail(node_id, shard_id, shard_id, fake_sm)
        assert abs(forfeited - held_before_fail) < 1e-9, (
            "Only the newly held escrow should be forfeited")

        # Released credits from the FIRST cycle survive in the ledger
        assert abs(ledger.credits(node_id) - released_credits) < 1e-9, (
            "Previously released credits must not be clawed back on audit fail")

    def test_dishonest_node_end_to_end_forfeits_escrow(self, tmp_path):
        """End-to-end: a dishonest node fails the audit and its held escrow is forfeited."""
        rng = random.Random(88)
        node_ids = [f"n{i}" for i in range(9)]
        mgr, reg, store, transport, events, audit, ledger = _manager_with_audit(
            node_ids, rng=rng)

        data = bytes(rng.getrandbits(8) for _ in range(4096 * 7))
        meta = asyncio.run(mgr.put(data, erasure=True, ec_k=6, ec_m=3))

        # Build up some escrow for a target node
        first_shard = meta.chunks[0]
        shard_meta = store.get_blob(first_shard)
        target_node = shard_meta.replicas[0]

        N = CONSECUTIVE_PASSES_TO_RELEASE
        # Pass N-1 audits honestly to build held escrow
        for _ in range(N - 1):
            asyncio.run(
                audit.audit_shard(target_node, first_shard, meta.id, mgr, transport,
                                  owner=target_node))
        held = audit.escrow_state(target_node)["held_escrow"]
        assert held > 0.0, "Must have held escrow before the fail"

        # Now the node goes dishonest: return a wrong hash
        transport.audit_overrides[target_node] = (
            lambda p: {"challenge_id": p["challenge_id"],
                       "hash": hashlib.sha256(b"wrong").hexdigest()})

        result = asyncio.run(
            audit.audit_shard(target_node, first_shard, meta.id, mgr, transport,
                              owner=target_node))
        assert result is False, "Dishonest node must fail the audit"

        # Escrow is forfeited
        state = audit.escrow_state(target_node)
        assert state["held_escrow"] == 0.0, "Held escrow must be forfeited on fail"
        assert state["consecutive_passes"] == 0, "Pass streak must be reset"
        # No credits in the ledger
        assert ledger.credits(target_node) == 0.0, (
            "Forfeited escrow must not appear as donated credits")


# ---------------------------------------------------------------------------
# Group 6: A blinded auditor must never masquerade as a healthy one
# ---------------------------------------------------------------------------

class TestBlindedAuditorNeverReportsPasses:
    """``_chunk_hashes`` is in-memory only, so a control-plane restart wipes every
    shard's verification material.

    Before the fix, ``audit_shard`` returned True for a shard it had no material
    for and ``run_audit_pass`` counted that as ``passed`` -- so a restarted
    coordinator reported "100% of audits passing" while verifying precisely
    nothing.  That is the worst possible failure mode for a trust component,
    because total loss of the security property is indistinguishable from
    success.

    These tests pin both halves of the fix:
      1. material is rebaselined from digest-verified bytes when reachable, and
      2. when it cannot be rebaselined the shard is ``skipped``, never ``passed``.
    """

    def test_restart_rebaselines_from_digest_verified_bytes(self, tmp_path):
        rng = random.Random(101)
        node_ids = [f"n{i}" for i in range(9)]
        mgr, reg, store, transport, events, audit, ledger = _manager_with_audit(
            node_ids, rng=rng)

        data = bytes(rng.getrandbits(8) for _ in range(4096 * 7))
        meta = asyncio.run(mgr.put(data, erasure=True, ec_k=6, ec_m=3))
        assert meta.chunks

        # Simulate a control-plane restart: all verification material is gone.
        audit._chunk_hashes.clear()
        assert not any(audit.has_shard_info(s) for s in meta.chunks)

        summary = asyncio.run(audit.run_audit_pass(mgr, transport))

        assert summary["passed"] > 0, (
            "auditor should rebaseline from digest-verified shard bytes and "
            f"then really audit; got {summary}")
        assert summary["failed"] == 0, f"honest nodes must not fail: {summary}"
        assert all(audit.has_shard_info(s) for s in meta.chunks), (
            "every audited shard should have material again after rebaseline")

    def test_unrebaselinable_shard_is_skipped_not_passed(self, tmp_path):
        """NON-VACUITY: when the bytes cannot be re-fetched the auditor must
        report ``skipped``, and must NOT inflate ``passed``."""
        rng = random.Random(202)
        node_ids = [f"n{i}" for i in range(9)]
        mgr, reg, store, transport, events, audit, ledger = _manager_with_audit(
            node_ids, rng=rng)

        data = bytes(rng.getrandbits(8) for _ in range(4096 * 7))
        meta = asyncio.run(mgr.put(data, erasure=True, ec_k=6, ec_m=3))

        # Restart wipes material, AND every holder refuses to serve the bytes
        # (still "online", so they are not simply skipped as offline).
        audit._chunk_hashes.clear()
        for nid in node_ids:
            transport.disks[nid].clear()

        summary = asyncio.run(audit.run_audit_pass(mgr, transport))

        assert summary["passed"] == 0, (
            "a blinded auditor must never report a pass -- this is exactly the "
            f"bug that makes total verification loss look like success: {summary}")
        assert summary["skipped"] > 0, (
            f"unauditable shards must be reported as skipped: {summary}")

    def test_forget_shard_releases_material_on_delete(self, tmp_path):
        """Deleted blobs must not leak verification material forever."""
        rng = random.Random(303)
        node_ids = [f"n{i}" for i in range(9)]
        mgr, reg, store, transport, events, audit, ledger = _manager_with_audit(
            node_ids, rng=rng)

        data = bytes(rng.getrandbits(8) for _ in range(4096 * 7))
        meta = asyncio.run(mgr.put(data, erasure=True, ec_k=6, ec_m=3))
        shard_ids = list(meta.chunks)
        assert any(audit.has_shard_info(s) for s in shard_ids)

        assert asyncio.run(mgr.delete(meta.id)) is True

        assert not any(audit.has_shard_info(s) for s in shard_ids), (
            "delete must release per-shard audit material, otherwise "
            "_chunk_hashes grows without bound")
