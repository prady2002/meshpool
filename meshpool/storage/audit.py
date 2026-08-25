"""Proof-of-storage audit engine — coordinator side (spec §P3 trust layer).

WHY THIS EXISTS
---------------
Donors are paid to store other people's data.  Nothing currently proves they
still HOLD it.  A rational cheater can delete the bytes, keep collecting rent,
and only be caught when a user tries to read the data — by which time the bytes
may be unrecoverable.  This module implements a challenge/response protocol that
forces node to prove possession of specific byte ranges on demand.

PROTOCOL
--------
1. **Write-time**: when a shard is placed on a node, the coordinator precomputes
   per-chunk hashes (SHA-256 of every AUDIT_CHUNK_SIZE block) and stores them
   in ``_chunk_hashes``.  These are the ground truth the coordinator uses to
   check responses without holding the full shard bytes.

2. **Challenge**: the coordinator picks a RANDOM chunk index and sends
   ``{shard_id, chunk_index, chunk_size, challenge_id}`` to the holding node.

3. **Response**: the node reads ``shard_bytes[chunk_index*chunk_size:...]`` from
   its local BlobStore and returns ``sha256(chunk_bytes).hex()``.

4. **Verification**: the coordinator checks the response against its precomputed
   chunk hash.  A node that still holds the bytes passes; one that deleted them
   cannot compute the right hash.

5. **Escrow**: storage earnings are held in escrow.  After
   ``CONSECUTIVE_PASSES_TO_RELEASE`` consecutive passing audits the escrow is
   released as "donated" credits in the existing ContributionLedger.  A failed
   audit forfeits all held escrow (no credit for that period).

TRADEOFFS vs. A FULL MERKLE TREE
---------------------------------
This implementation stores **one hash per chunk** (``O(n / AUDIT_CHUNK_SIZE)``
entries per shard).  A Merkle tree would let the coordinator verify any
sub-range of bytes with an ``O(log n)`` proof — the coordinator only needs the
root hash, and the node sends a sibling path.  That is stronger (a node cannot
defeat it by keeping *all* chunk hashes without the bytes), but significantly
more complex to implement correctly.

Limitation: a sufficiently motivated cheater CAN defeat per-chunk hashes by
keeping a hash table of all chunk hashes instead of the raw bytes.  For a 1 MB
shard with 4 KB chunks the hash table costs 256 × 32 = 8 KB vs. 1 MB of data.
The economic defence is that the **escrow forfeit must exceed the storage cost
savings**: if forfeiting escrow is more expensive than keeping the bytes, cheating
is negative-EV even for a rational actor.  Frequent audits reduce the time window
a cheater can exploit before detection.

Future: upgrade to PoRep (Proof of Replication) or a Merkle-tree based PoSt
for cryptographically strong guarantees that the bytes themselves must be held.
"""
from __future__ import annotations

import hashlib
import random
import time
from typing import Optional

from ..common.logging import get_logger
from ..common.models import new_id, now_ms
from ..common.trust import ContributionLedger

log = get_logger("storage.audit")

# Size of each auditable chunk inside a shard (bytes).
# Smaller → more granular challenges (harder to defeat with a hash-cache).
# Larger  → fewer hashes stored per shard (lower coordinator memory).
# 4096 is a good balance: O(256) hashes per MB of shard data.
AUDIT_CHUNK_SIZE: int = 4096

# How many consecutive passing audits before held escrow is released.
CONSECUTIVE_PASSES_TO_RELEASE: int = 3

# Storage credits accrued per audit cycle (held in escrow until release).
# The forfeit-on-fail must exceed the storage cost savings for cheating to be
# negative-EV (see module docstring tradeoffs section).
ESCROW_PER_AUDIT_UNIT: float = 1.0

# Anti-thrash: minimum seconds between repairs for the same blob.
REPAIR_COOLDOWN_S: float = 60.0

# Maximum seconds we'll wait for a node to respond to a challenge.
CHALLENGE_TIMEOUT_S: float = 30.0


class StorageAuditEngine:
    """Coordinator-side proof-of-storage engine.

    Thread safety: all public methods are designed to be called from a single
    asyncio event loop (the control-plane coroutine).  They do not take locks
    internally; concurrent async callers should await one at a time.
    """

    def __init__(
        self,
        ledger: ContributionLedger,
        rng: Optional[random.Random] = None,
    ) -> None:
        # per-shard verification material: shard_id -> [sha256_bytes, ...]
        # One entry per AUDIT_CHUNK_SIZE block of the shard's raw bytes.
        self._chunk_hashes: dict[str, list[bytes]] = {}

        # pending challenges: challenge_id -> {shard_id, node_id, chunk_index, issued_ms}
        self._pending: dict[str, dict] = {}

        # per-node audit state
        # node_id -> {"consecutive_passes": int, "held_escrow": float}
        self._node_state: dict[str, dict] = {}

        # anti-thrash: blob_id -> last repair timestamp (seconds)
        self._last_repair: dict[str, float] = {}

        self.ledger = ledger

        self._rng = rng or random.Random()

    # ------------------------------------------------------------------ #
    # Write-time: record verification material
    # ------------------------------------------------------------------ #

    def record_shard_hashes(self, shard_id: str, data: bytes) -> None:
        """Precompute per-chunk hashes for ``shard_id`` from its raw bytes.

        Must be called at shard placement time (``_store_ec_leaf_locked``).
        The chunk hashes are the coordinator's ground truth; it does not need
        the full shard bytes to verify future challenge responses.
        """
        chunks: list[bytes] = []
        for off in range(0, max(len(data), 1), AUDIT_CHUNK_SIZE):
            chunk = data[off:off + AUDIT_CHUNK_SIZE]
            chunks.append(hashlib.sha256(chunk).digest())
        self._chunk_hashes[shard_id] = chunks

    def has_shard_info(self, shard_id: str) -> bool:
        return shard_id in self._chunk_hashes

    def forget_shard(self, shard_id: str) -> None:
        """Drop verification material for a shard that has been deleted."""
        self._chunk_hashes.pop(shard_id, None)

    async def ensure_shard_hashes(
        self,
        shard_id: str,
        node_ids: list[str],
        storage_manager,
    ) -> bool:
        """Re-derive missing verification material for ``shard_id``, if possible.

        WHY THIS EXISTS
        ---------------
        ``_chunk_hashes`` lives only in memory.  A control-plane restart therefore
        wipes every shard's verification material, and without this method the
        auditor would silently become a no-op for the entire pre-existing dataset
        forever — the single worst failure mode this module can have, because it
        looks exactly like success.

        Rebuilding is **cryptographically safe** even though the bytes come from
        the very nodes we are policing: a shard id IS the sha256 of its bytes
        (``StorageManager.digest``), and ``StorageManager._fetch`` refuses any
        payload whose digest does not match the id it asked for.  A cheater can
        therefore only supply the genuine bytes or nothing at all — and supplying
        the genuine bytes is precisely the thing we wanted to prove.

        Returns True if verification material exists after the call.
        """
        if shard_id in self._chunk_hashes:
            return True
        if not node_ids:
            return False
        try:
            raw = await storage_manager._fetch(node_ids, shard_id)
        except Exception as exc:  # noqa: BLE001
            log.warning("audit rebaseline fetch failed",
                        shard=shard_id[:12], err=str(exc)[:120])
            return False
        if raw is None:
            return False
        self.record_shard_hashes(shard_id, raw)
        log.info("audit material rebaselined from digest-verified shard",
                 shard=shard_id[:12], chunks=len(self._chunk_hashes[shard_id]))
        return True

    # ------------------------------------------------------------------ #
    # Challenge issuance
    # ------------------------------------------------------------------ #

    def issue_challenge(self, node_id: str, shard_id: str) -> Optional[dict]:
        """Create a proof-of-storage challenge for ``node_id`` holding ``shard_id``.

        Returns a dict the caller should send to the node, or None if no
        verification material exists for this shard (it was placed before the
        audit engine was attached, or already deleted).

        The random ``chunk_index`` ensures a node that memorised the answer to
        one specific challenge fails when a different chunk is asked next time.
        """
        hashes = self._chunk_hashes.get(shard_id)
        if not hashes:
            return None
        chunk_index = self._rng.randrange(len(hashes))
        challenge_id = new_id("aud")
        self._pending[challenge_id] = {
            "shard_id": shard_id,
            "node_id": node_id,
            "chunk_index": chunk_index,
            "issued_ms": now_ms(),
        }
        return {
            "challenge_id": challenge_id,
            "shard_id": shard_id,
            "chunk_index": chunk_index,
            "chunk_size": AUDIT_CHUNK_SIZE,
        }

    # ------------------------------------------------------------------ #
    # Challenge verification
    # ------------------------------------------------------------------ #

    def verify_response(self, challenge_id: str, response_hex: str) -> bool:
        """Check a node's response against the precomputed chunk hash.

        ``response_hex`` must equal ``sha256(chunk_bytes).hex()`` for the
        challenged chunk.  Returns True on match, False otherwise (including
        expired or unknown challenge ids).

        The challenge is consumed on first call; a second call with the same id
        always returns False (prevents replay).
        """
        ch = self._pending.pop(challenge_id, None)
        if ch is None:
            return False
        elapsed_s = (now_ms() - ch["issued_ms"]) / 1000.0
        if elapsed_s > CHALLENGE_TIMEOUT_S:
            log.info("audit challenge expired", node=ch["node_id"],
                     shard=ch["shard_id"][:12], elapsed_s=round(elapsed_s, 1))
            return False
        hashes = self._chunk_hashes.get(ch["shard_id"])
        if not hashes or ch["chunk_index"] >= len(hashes):
            return False
        expected = hashes[ch["chunk_index"]].hex()
        return response_hex == expected

    # ------------------------------------------------------------------ #
    # Audit outcome handlers
    # ------------------------------------------------------------------ #

    def on_audit_pass(self, node_id: str, shard_id: str, owner: str = "default") -> float:
        """Record a passing audit.  Returns the amount of escrow released (may be 0)."""
        state = self._node_state.setdefault(
            node_id, {"consecutive_passes": 0, "held_escrow": 0.0})
        state["consecutive_passes"] += 1
        state["held_escrow"] += ESCROW_PER_AUDIT_UNIT
        released = 0.0
        if state["consecutive_passes"] >= CONSECUTIVE_PASSES_TO_RELEASE:
            released = state["held_escrow"]
            # Recorded as a *grant*, not a donation. An escrow release is a bonus
            # the platform issues for sustained good behaviour -- no consumer paid
            # for it. Booking it as "donated" inflates one half of a two-sided
            # transfer, which quietly breaks the donated == consumed invariant that
            # verify_ledger_conservation.py relies on to detect credit forgery.
            # The node's spendable balance is unchanged: credits_by_owner() counts
            # donated and grant identically.
            self.ledger.record_grant(
                owner, released,
                detail=f"storage escrow released: {state['consecutive_passes']} "
                       f"consecutive audit passes for shard {shard_id[:12]}",
                node_id=node_id)
            log.info("escrow released", node=node_id, amount=released,
                     passes=state["consecutive_passes"])
            state["consecutive_passes"] = 0
            state["held_escrow"] = 0.0
        return released

    def on_audit_fail(
        self,
        node_id: str,
        shard_id: str,
        blob_id: str,
        storage_manager,
    ) -> float:
        """Record a failing audit.

        Forfeits all held escrow for ``node_id`` and drops the shard placement for
        ``node_id`` so reads stop targeting it.

        Repair is **not** performed here.  Removing the node from the shard's
        replica set is what causes ``StorageManager._reconcile`` to observe the
        blob as degraded and call ``_repair_ec_locked`` on its next sweep; that is
        deliberately the single repair path, so an audit failure and a node going
        offline heal through identical, already-tested code.  The event published
        below is an observability signal for the UI/SSE stream, rate-limited by
        ``REPAIR_COOLDOWN_S`` so a mass failure cannot flood subscribers.

        Returns the amount of escrow forfeited.
        """
        state = self._node_state.setdefault(
            node_id, {"consecutive_passes": 0, "held_escrow": 0.0})
        forfeited = state["held_escrow"]
        state["consecutive_passes"] = 0
        state["held_escrow"] = 0.0
        log.warning("audit FAIL: escrow forfeited",
                    node=node_id, shard=shard_id[:12],
                    blob=blob_id[:12], forfeited=forfeited)
        # Drop the failing node from the shard's placement set so it stops being
        # targeted for reads and reconcile re-places the shard elsewhere.
        try:
            sm = storage_manager.store.get_blob(shard_id)
            if sm is not None and node_id in sm.replicas:
                sm.replicas = [r for r in sm.replicas if r != node_id]
                if not sm.replicas:
                    sm.status = "lost"
                storage_manager.store.put_blob(sm)
        except Exception as exc:  # noqa: BLE001
            log.warning("could not update shard meta after audit fail",
                        err=str(exc)[:120])
        # Anti-thrash: do not re-trigger repair for the same blob too frequently.
        now = time.monotonic()
        last = self._last_repair.get(blob_id, 0.0)
        if now - last >= REPAIR_COOLDOWN_S:
            self._last_repair[blob_id] = now
            storage_manager.events.publish(
                "audit_fail_repair",
                {"blob_id": blob_id, "shard_id": shard_id, "node_id": node_id})
        return forfeited

    # ------------------------------------------------------------------ #
    # Full audit cycle (coordinator calls agent, verifies, updates state)
    # ------------------------------------------------------------------ #

    async def audit_shard(
        self,
        node_id: str,
        shard_id: str,
        blob_id: str,
        storage_manager,
        transport,
        owner: str = "default",
    ) -> bool:
        """Issue one challenge for ``shard_id`` on ``node_id`` and verify the response.

        Returns True if the node proved possession, False otherwise.
        Calls ``on_audit_pass`` or ``on_audit_fail`` accordingly.
        """
        challenge = self.issue_challenge(node_id, shard_id)
        if challenge is None:
            log.debug("no audit material for shard; skipping", shard=shard_id[:12])
            return True   # unknown shard — cannot judge; skip (not a fail)
        try:
            from ..common.transport import TransportError
            resp = await transport.call(
                node_id, "blob_audit", challenge, timeout=CHALLENGE_TIMEOUT_S)
        except Exception as exc:  # noqa: BLE001
            log.warning("audit transport error", node=node_id, shard=shard_id[:12],
                        err=str(exc)[:120])
            # Treat an unreachable node as a failure.
            self._pending.pop(challenge["challenge_id"], None)
            self.on_audit_fail(node_id, shard_id, blob_id, storage_manager)
            return False
        response_hex = (resp or {}).get("hash", "")
        passed = self.verify_response(challenge["challenge_id"], response_hex)
        if passed:
            self.on_audit_pass(node_id, shard_id, owner=owner)
        else:
            self.on_audit_fail(node_id, shard_id, blob_id, storage_manager)
        return passed

    async def run_audit_pass(self, storage_manager, transport) -> dict:
        """Audit a random shard for every online shard-holding node.

        Returns a summary dict ``{passed, failed, skipped}``.
        Safe to call repeatedly from a background loop; the anti-thrash guard
        prevents repair storms.

        A shard with no verification material is **never counted as a pass** — it
        is rebaselined if possible and otherwise reported as ``skipped``, so a
        blinded auditor can never masquerade as a healthy one.
        """
        passed = failed = skipped = 0
        live = {n.id for n in storage_manager.registry.online()}

        for meta in storage_manager.store.list_blobs():
            if not meta.chunks:
                continue   # replicated blob, not EC shards
            if not storage_manager._is_ec(meta):
                continue
            # Walk each shard and audit the node that holds it
            for shard_id in meta.chunks:
                shard_meta = storage_manager.store.get_blob(shard_id)
                if shard_meta is None or not shard_meta.replicas:
                    skipped += 1
                    continue
                node_id = shard_meta.replicas[0]
                if node_id not in live:
                    skipped += 1
                    continue
                live_holders = [r for r in shard_meta.replicas if r in live]
                if not await self.ensure_shard_hashes(
                        shard_id, live_holders, storage_manager):
                    # Cannot judge this shard: do NOT inflate the pass count.
                    skipped += 1
                    continue
                owner = meta.owner
                result = await self.audit_shard(
                    node_id, shard_id, meta.id, storage_manager, transport, owner=owner)
                if result:
                    passed += 1
                else:
                    failed += 1

        return {"passed": passed, "failed": failed, "skipped": skipped}

    async def run_audit_loop(
        self,
        storage_manager,
        transport,
        interval: float = 30.0,
    ) -> None:
        """Background loop: periodically audit all shard holders.

        Designed to run as an asyncio task in the control plane lifespan.
        """
        import asyncio
        while True:
            try:
                summary = await self.run_audit_pass(storage_manager, transport)
                if summary["passed"] + summary["failed"] > 0:
                    log.info("audit round complete", **summary)
            except Exception as exc:  # noqa: BLE001
                log.error("audit loop error", err=str(exc)[:160])
            await asyncio.sleep(interval)

    # ------------------------------------------------------------------ #
    # Escrow introspection (for tests and dashboards)
    # ------------------------------------------------------------------ #

    def escrow_state(self, node_id: str) -> dict:
        """Return the current escrow state for a node."""
        state = self._node_state.get(node_id, {"consecutive_passes": 0, "held_escrow": 0.0})
        return dict(state)

    def total_held_escrow(self) -> float:
        """Sum of all held escrow across all nodes."""
        return sum(s["held_escrow"] for s in self._node_state.values())
