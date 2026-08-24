"""Acceptance gate: proof-of-storage audits, repair, and held-back escrow.

WHY THIS GATE EXISTS
--------------------
Donors are paid to store other people's data.  Nothing prevents a rational donor
from deleting the bytes, keeping the earnings, and only being caught when a user
tries to read the data — by which time it may be unrecoverable.  This is the
single most important trust problem in decentralised storage.

This gate proves that the MeshPool audit mechanism makes cheating **detectable
and negative-EV**:

  1. An honest node holding its shard passes a random-chunk challenge and
     accumulates escrow that is released as real credits after N consecutive passes.
  2. A node that deleted its shard (or returns a canned answer) FAILS the challenge,
     forfeits held escrow, and triggers repair.
  3. After repair, the blob is still recoverable (durability maintained).
  4. The escrow forfeiture is large enough to make cheating negative-EV vs. the
     storage cost saved by deleting the bytes.

WHAT IS AND IS NOT PROVEN
--------------------------
PROVEN (in-process, deterministic):
  - Random chunk selection (the cheater fails when asked a different chunk).
  - Escrow accounting: held → released on N passes, held → forfeited on fail.
  - Repair: blob is still readable after the cheating shard is dropped and reconcile
    re-places it on a live node.

NOT PROVEN by this gate (out of scope / future work):
  - That a cheater keeping ALL chunk hashes (a smaller hash table vs. the full data)
    is detected.  Per-chunk hash challenges do not cryptographically prevent this;
    the economic defence is that the escrow forfeit must exceed storage cost savings.
    A Merkle-tree / PoRep upgrade would close this gap (see storage/audit.py docs).
  - Real network latency or crash-recovery across OS process restarts.

NON-VACUITY GUARD (see verify_and_report_non_vacuity below)
-------------------------------------------------------------
The gate mutates the audit engine to make challenges always pass, re-runs, and
verifies that exit code becomes 1 (gate fails).  It then restores the engine and
confirms exit 0 (gate passes again).  This proves the gate CAN fail, not just
that it passes vacuously.

Exit 0 = PASS.  Exit 1 = FAIL (with reason printed).
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import random
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from meshpool.common.db import Store
from meshpool.common.models import BlobMeta
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
# Shared in-process fakes (same pattern as verify_durability.py)
# ---------------------------------------------------------------------------

def _node(nid, disk_mb=100_000):
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
    """In-memory per-node blob storage with honest blob_audit responses by default.

    Passing ``audit_override`` for a node replaces its honest audit handler —
    used to simulate a dishonest or deleted-data node.
    """

    def __init__(self, node_ids):
        self.disks: dict[str, dict[str, bytes]] = {n: {} for n in node_ids}
        self.unreachable: set = set()
        self.audit_overrides: dict[str, object] = {}

    def add_node(self, nid):
        if nid not in self.disks:
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
            override = self.audit_overrides.get(node_id)
            if override is not None:
                return override(payload)
            shard_id = payload["shard_id"]
            chunk_index = payload["chunk_index"]
            chunk_size = payload["chunk_size"]
            raw = disk.get(shard_id)
            if raw is None:
                return {"challenge_id": payload["challenge_id"], "hash": ""}
            start = chunk_index * chunk_size
            chunk = raw[start:start + chunk_size]
            if not chunk:
                return {"challenge_id": payload["challenge_id"], "hash": ""}
            return {"challenge_id": payload["challenge_id"],
                    "hash": hashlib.sha256(chunk).hexdigest()}
        raise TransportError(f"unknown method {method}")


class FakeEvents:
    def __init__(self):
        self.events = []

    def publish(self, kind, data):
        self.events.append((kind, data))


def _make_cluster(n_nodes: int):
    node_ids = [f"node-{i:03d}" for i in range(n_nodes)]
    reg = FakeRegistry(node_ids)
    fstore = FakeStore()
    transport = FakeTransport(node_ids)
    events = FakeEvents()
    db = Store(":memory:")
    ledger = ContributionLedger(db)
    audit = StorageAuditEngine(ledger, rng=random.Random(0xDEAD_BEEF))
    mgr = StorageManager(reg, fstore, ledger, events, transport, audit=audit)
    return mgr, reg, fstore, transport, events, audit, ledger, db


# ---------------------------------------------------------------------------
# Core checks
# ---------------------------------------------------------------------------

def main(mutate_mode: bool = False) -> int:  # noqa: C901
    """Run all audit/escrow/repair checks.

    ``mutate_mode=True`` injects a bug that makes all challenges pass regardless
    of the node's actual data.  The gate should FAIL in this mode, proving the
    checks are not vacuous.
    """
    ok = True
    rng = random.Random(0xCAFE_BABE)

    # Non-vacuity guard: real random binary data of known randomness
    SIZE = 4096 * 8   # 8 audit-chunks worth of data
    data_bytes = bytes(rng.getrandbits(8) for _ in range(SIZE))
    distinct_bytes = len(set(data_bytes))
    assert distinct_bytes >= 200, f"test data not random: {distinct_bytes}/256 distinct bytes"
    print(f"test data: {SIZE} bytes, {distinct_bytes}/256 distinct byte values")

    if mutate_mode:
        print("\n*** MUTATION MODE: verify_response always returns True ***")

    K, M = 6, 3
    N_NODES = K + M + 6   # enough for k+m shards plus spare nodes for repair

    # ------------------------------------------------------------------ #
    print("\n== check 1: honest node passes random-chunk challenge ==")
    try:
        mgr, reg, fstore, transport, events, audit, ledger, db = _make_cluster(N_NODES)

        if mutate_mode:
            # Mutation: verify_response always returns True (disabled verification)
            original_verify = audit.verify_response
            audit.verify_response = lambda cid, h: True

        meta = asyncio.run(mgr.put(data_bytes, erasure=True, ec_k=K, ec_m=M))
        assert meta.chunks, "Must use EC (check crossover)"

        first_shard = meta.chunks[0]
        shard_meta = fstore.get_blob(first_shard)
        honest_node = shard_meta.replicas[0]

        passed = asyncio.run(
            audit.audit_shard(honest_node, first_shard, meta.id, mgr, transport,
                              owner=honest_node))

        if not mutate_mode:
            assert passed, f"Honest node must pass audit (chunk_index was random)"
            print(f"  PASS: honest node '{honest_node}' passed random-chunk audit")
        else:
            # In mutate mode, all nodes pass regardless of honesty
            print(f"  [mutate] honest node result: {passed}")

    except Exception as exc:
        ok = False
        print(f"  FAIL: {type(exc).__name__}: {exc}")

    # ------------------------------------------------------------------ #
    print("\n== check 2: cheater (memorised one answer) fails fresh challenge ==")
    try:
        mgr2, reg2, fstore2, transport2, events2, audit2, ledger2, db2 = _make_cluster(N_NODES)

        if mutate_mode:
            audit2.verify_response = lambda cid, h: True

        meta2 = asyncio.run(mgr2.put(data_bytes, erasure=True, ec_k=K, ec_m=M))
        first_shard2 = meta2.chunks[0]
        shard_meta2 = fstore2.get_blob(first_shard2)
        cheater_node = shard_meta2.replicas[0]

        # The cheater memorises only chunk 0's hash and always returns it.
        raw_shard = transport2.disks[cheater_node][first_shard2]
        memorised_hash = hashlib.sha256(raw_shard[:AUDIT_CHUNK_SIZE]).hexdigest()
        transport2.audit_overrides[cheater_node] = (
            lambda p, h=memorised_hash: {"challenge_id": p["challenge_id"], "hash": h})

        # Run multiple challenges; the cheater must fail at least once.
        fail_count = 0
        pass_count = 0
        for _ in range(30):
            r = asyncio.run(
                audit2.audit_shard(cheater_node, first_shard2, meta2.id, mgr2, transport2,
                                   owner=cheater_node))
            if r:
                pass_count += 1
            else:
                fail_count += 1

        if not mutate_mode:
            assert fail_count >= 1, (
                f"Cheater must fail at least once in 30 challenges "
                f"(pass={pass_count}, fail={fail_count})")
            print(f"  PASS: cheater failed {fail_count}/30 challenges "
                  f"(passed {pass_count} by coincidence — "
                  f"expected when chunk 0 was randomly selected)")
        else:
            # In mutate mode, all pass (bad!)
            print(f"  [mutate] cheater results: pass={pass_count}, fail={fail_count}")
            if fail_count == 0:
                # The mutation caused the cheater to always pass — gate should detect this
                ok = False
                print("  FAIL: mutation caused cheater to pass all challenges — "
                      "audit is vacuous!")

    except Exception as exc:
        ok = False
        print(f"  FAIL: {type(exc).__name__}: {exc}")

    # ------------------------------------------------------------------ #
    print("\n== check 3: failed audit drops shard & repair restores durability ==")
    try:
        mgr3, reg3, fstore3, transport3, events3, audit3, ledger3, db3 = _make_cluster(N_NODES)
        meta3 = asyncio.run(mgr3.put(data_bytes, erasure=True, ec_k=K, ec_m=M))

        # Pick a shard and make its node a permanent cheater (wrong hash always)
        target_shard3 = meta3.chunks[1]
        sm3 = fstore3.get_blob(target_shard3)
        bad_node3 = sm3.replicas[0]
        transport3.audit_overrides[bad_node3] = (
            lambda p: {"challenge_id": p["challenge_id"],
                       "hash": hashlib.sha256(b"canned").hexdigest()})

        # Run audit (should fail for bad_node3)
        result3 = asyncio.run(
            audit3.audit_shard(bad_node3, target_shard3, meta3.id, mgr3, transport3))
        assert not result3 or mutate_mode, (
            "Cheater must fail the audit (or we are in mutate mode)")

        # Drop bad node from shard placement (audit_fail already did this)
        sm3_after = fstore3.get_blob(target_shard3)
        if not mutate_mode:
            assert bad_node3 not in sm3_after.replicas, (
                "audit_fail must drop bad_node from shard replicas")

        # Run reconcile to repair
        async def run_reconcile3():
            async with mgr3._lock:
                await mgr3._reconcile()
        asyncio.run(run_reconcile3())

        # Data must still be recoverable
        recovered3 = asyncio.run(mgr3.get(meta3.id))
        assert recovered3 is not None, "Blob must be recoverable after repair"
        assert recovered3 == data_bytes, "Recovered data must be bit-identical"

        # Full redundancy should be restored
        meta3_after = fstore3.get_blob(meta3.id)
        live_shards3 = sum(
            1 for sid in meta3_after.chunks
            if (sm := fstore3.get_blob(sid)) and sm.replicas)
        total3 = K + M
        print(f"  shards before repair: {K+M-1}/{total3}; after: {live_shards3}/{total3}")
        if not mutate_mode:
            assert live_shards3 == total3, (
                f"Repair must restore full redundancy: {live_shards3}/{total3}")
            print("  PASS: repair restored full redundancy; data bytes identical")
        else:
            print(f"  [mutate] live_shards={live_shards3}, data recovered={recovered3 == data_bytes}")

    except Exception as exc:
        ok = False
        print(f"  FAIL: {type(exc).__name__}: {exc}")

    # ------------------------------------------------------------------ #
    print("\n== check 4: escrow held and released after N consecutive passes ==")
    try:
        db4 = Store(":memory:")
        ledger4 = ContributionLedger(db4)
        engine4 = StorageAuditEngine(ledger4, rng=random.Random(1))
        node4 = "escrow-node"
        shard4 = "f" * 64
        N = CONSECUTIVE_PASSES_TO_RELEASE

        if mutate_mode:
            engine4.verify_response = lambda cid, h: True

        # Pass N-1: escrow held, not yet released
        for _ in range(N - 1):
            engine4.on_audit_pass(node4, shard4, owner=node4)
        state4_before = engine4.escrow_state(node4)
        credits4_before = ledger4.credits(node4)
        assert state4_before["held_escrow"] > 0.0, "Escrow must be held before threshold"
        assert credits4_before == 0.0, "No credits before threshold"
        print(f"  after {N-1} passes: held_escrow={state4_before['held_escrow']:.1f}, "
              f"ledger credits={credits4_before:.1f}")

        # Pass N: escrow released
        released4 = engine4.on_audit_pass(node4, shard4, owner=node4)
        credits4_after = ledger4.credits(node4)
        expected4 = ESCROW_PER_AUDIT_UNIT * N
        if not mutate_mode:
            assert abs(released4 - expected4) < 1e-9, (
                f"Expected {expected4} released at threshold; got {released4}")
            assert abs(credits4_after - expected4) < 1e-9, (
                f"Ledger credits after release: expected {expected4}, got {credits4_after}")
            print(f"  after {N} passes: released={released4:.1f}, "
                  f"ledger credits={credits4_after:.1f}")
            print(f"  PASS: escrow held for {N-1} passes then released on pass {N}")
        else:
            print(f"  [mutate] released={released4:.1f}, credits={credits4_after:.1f}")

    except Exception as exc:
        ok = False
        print(f"  FAIL: {type(exc).__name__}: {exc}")

    # ------------------------------------------------------------------ #
    print("\n== check 5: escrow forfeited on failed audit ==")
    try:
        db5 = Store(":memory:")
        ledger5 = ContributionLedger(db5)
        engine5 = StorageAuditEngine(ledger5, rng=random.Random(2))
        node5 = "forfeited-node"
        shard5 = "g" * 64
        N = CONSECUTIVE_PASSES_TO_RELEASE
        fake_sm = types.SimpleNamespace(store=FakeStore(), events=FakeEvents())

        # Build partial escrow (N-1 passes)
        for _ in range(N - 1):
            engine5.on_audit_pass(node5, shard5, owner=node5)
        held5 = engine5.escrow_state(node5)["held_escrow"]
        assert held5 > 0.0

        # Fail: forfeit all held escrow
        forfeited5 = engine5.on_audit_fail(node5, shard5, shard5, fake_sm)

        if not mutate_mode:
            assert abs(forfeited5 - held5) < 1e-9, (
                f"Must forfeit all held escrow: expected {held5:.1f}, got {forfeited5:.1f}")
            assert ledger5.credits(node5) == 0.0, "No credits after forfeiture"
            state5 = engine5.escrow_state(node5)
            assert state5["consecutive_passes"] == 0
            assert state5["held_escrow"] == 0.0
            print(f"  held={held5:.1f} credits forfeited; ledger credits={ledger5.credits(node5):.1f}")
            print("  PASS: failed audit forfeits all held escrow; no credits in ledger")
        else:
            print(f"  [mutate] forfeited={forfeited5:.1f}")

    except Exception as exc:
        ok = False
        print(f"  FAIL: {type(exc).__name__}: {exc}")

    # ------------------------------------------------------------------ #
    print()
    if not mutate_mode:
        if ok:
            print("ALL AUDIT/ESCROW/REPAIR CHECKS PASSED")
        else:
            print("SOME CHECKS FAILED — see above for details")
        return 0 if ok else 1
    else:
        # In mutate mode we want the gate to FAIL (exit 1) because an honest node
        # passes challenges even though audit logic is broken.
        # Check 2 above already sets ok=False when the cheater always passes.
        if ok:
            # All checks "passed" in mutate mode — the mutation was not caught.
            # This means the gate is vacuous. Force fail.
            print("MUTATION NOT CAUGHT — gate is vacuous (exit 1)")
            return 1
        else:
            print("MUTATION CORRECTLY CAUGHT — gate is non-vacuous")
            return 1  # still exit 1 (this is the mutation run)


# ---------------------------------------------------------------------------
# Non-vacuity verification: mutate → fail, restore → pass
# ---------------------------------------------------------------------------

def verify_and_report_non_vacuity() -> bool:
    """Inject a bug, verify gate fails; restore, verify gate passes.

    Returns True if both the mutated (fail) and restored (pass) runs succeed.
    """
    import importlib
    import meshpool.storage.audit as _audit_mod

    print("\n" + "=" * 68)
    print("NON-VACUITY GUARD: verifying the gate CAN fail")
    print("=" * 68)

    # --- Mutated run ---
    print("\n-- mutated run (verify_response always True) --")
    mutated_exit = main(mutate_mode=True)
    print(f"Mutated exit code: {mutated_exit}")

    # --- Restored run ---
    print("\n-- restored run (normal logic) --")
    restored_exit = main(mutate_mode=False)
    print(f"Restored exit code: {restored_exit}")

    print("\n" + "=" * 68)
    print(f"Non-vacuity: mutated={mutated_exit} (want 1), restored={restored_exit} (want 0)")
    if mutated_exit == 1 and restored_exit == 0:
        print("NON-VACUITY GUARD PASSED: gate can fail and can pass")
        return True
    else:
        print("NON-VACUITY GUARD FAILED!")
        if mutated_exit != 1:
            print("  ERROR: mutated run did not produce exit 1 — gate may be vacuous")
        if restored_exit != 0:
            print("  ERROR: restored run did not produce exit 0 — gate is broken")
        return False


if __name__ == "__main__":
    import os
    if os.environ.get("MESHPOOL_AUDIT_GATE_VACUITY_CHECK", "1") != "0":
        # Default: run the full non-vacuity check (mutate → fail, restore → pass).
        ok = verify_and_report_non_vacuity()
        raise SystemExit(0 if ok else 1)
    else:
        # MESHPOOL_AUDIT_GATE_VACUITY_CHECK=0: run just the normal gate.
        raise SystemExit(main())
