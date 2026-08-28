"""Pooled object-store control-plane manager (spec §P3).

Turns spare disk across many nodes into one durable content-addressed store.  The
default placement strategy is Reed–Solomon (k=6, m=3) erasure coding: data is split
into k data shards plus m parity shards, stored on k+m distinct nodes, and survives
the simultaneous loss of ANY m nodes.  This gives 1.5× storage overhead and tolerates
3 failures, versus 2× overhead and only 1 tolerable failure with plain replication.

Very small objects (below the crossover threshold) fall back to replication since the
per-shard header overhead makes EC uneconomical there.

The public API of StorageManager is fully backwards-compatible: existing callers pass
a ``replication`` keyword (interpreted as the old replica count) and get replicated
storage; the new ``erasure`` keyword triggers the EC path.  The default changes from
replication to EC.

Encoding parameters (k, m) are recorded in ``replication_desired`` as the packed
integer ``k * 256 + m`` (≤ 65535), distinguished from plain replication by the
``_ec_params`` registry on the manager instance and by the EC shard blobs stored in
the ``chunks`` field of the manifest.
"""
from __future__ import annotations

import asyncio
import base64
import binascii
import hashlib
import json
import os
import secrets
import time
from typing import Optional

from ..common.logging import get_logger
from ..common.models import BlobMeta
from ..common.transport import AgentTransport, TransportError
from .erasure import RSCodec, erasure_crossover_bytes

log = get_logger("storage")

DEFAULT_REPLICATION = 2

# Default erasure-coding parameters:
#   k=6 data shards, m=3 parity shards → 1.5× overhead, tolerates 3 failures
# See docs/storage.md for the durability calculation.
DEFAULT_EC_K = 6
DEFAULT_EC_M = 3

# ---------------------------------------------------------------------------
# Storj-inspired tiered erasure-coding profiles (§5.1, §5.3, §5.4).
#
# Notation: RS(k, total) where total = k + m (parity shards m = total - k).
# Overhead = total/k.  "Repair threshold" = minimum live pieces before the
# repair loop fires -- below this count pieces are regenerated onto new nodes.
#
# Source labels (§5.2 honesty requirement):
#   cold 35/80 repair threshold -- CITED: Storj production (Storj blog, 2018)
#   hot/warm repair thresholds  -- CALIBRATED: same 43.75% ratio as Storj cold
#   all other numbers           -- MEASURED / computed from the RS math above
#
# HONESTY: RS(29,80) cold requires 80 nodes.  A 3-node cluster cannot use it;
# the engine falls back to 2× replication automatically (see `put()`).  The
# unit / sim tests for this tier use in-process virtual clusters with 80
# simulated node objects.  Real multi-datacenter durability numbers require an
# actual deployed fleet -- do NOT cite the single-host emulated numbers as if
# they represent a real deployment.
# ---------------------------------------------------------------------------
EC_PROFILES: dict[str, dict] = {
    "hot":  {"k": 10, "m": 20, "repair_threshold": 13},  # RS(10,30)  13/30 ≈ 43.3%  [calibrated]
    "warm": {"k": 20, "m": 40, "repair_threshold": 26},  # RS(20,60)  26/60 ≈ 43.3%  [calibrated]
    "cold": {"k": 29, "m": 51, "repair_threshold": 35},  # RS(29,80)  35/80 = 43.75% [cited: Storj]
}

# Objects larger than this are split into content-addressed chunks, each replicated
# independently across the pool, so one object can exceed any single machine's free
# disk. Read at call time so tests and end-to-end runs can shrink it via the env.
DEFAULT_CHUNK_SIZE = 8 * 1024 * 1024   # 8 MB


def _chunk_size() -> int:
    try:
        n = int(os.environ.get("MESHPOOL_CHUNK_SIZE", str(DEFAULT_CHUNK_SIZE)))
    except ValueError:
        return DEFAULT_CHUNK_SIZE
    return n if n > 0 else DEFAULT_CHUNK_SIZE


def _ec_crossover(k: int, m: int) -> int:
    """Return the minimum object size at which EC is cheaper than 2× replication."""
    return erasure_crossover_bytes(k, m, replication=DEFAULT_REPLICATION)


def _pack_ec_params(k: int, m: int) -> int:
    """Encode k,m into ``replication_desired`` as k*256+m."""
    return k * 256 + m


def _unpack_ec_params(packed: int) -> tuple[int, int]:
    """Decode k,m from a packed ``replication_desired``."""
    return packed >> 8, packed & 0xFF


class StorageError(Exception):
    """No schedulable node could accept a blob (the pool has no free disk right now)."""


# Sentinel: a blob_id suffix that marks a blob as holding one EC shard.
# Real shard blob IDs are SHA-256 hashes of shard bytes so they are just hex strings.
_EC_SHARD_MARK = "\x00EC\x00"   # never appears in a hex digest; used only in-memory
_BUCKET_META_PREFIX = "__meshpool_bucket__"
_VERSION_META_PREFIX = "__meshpool_version__"


class StorageManager:
    def __init__(self, registry, store, ledger, events, transport: AgentTransport,
                 audit=None) -> None:
        self.registry = registry
        self.store = store
        self.ledger = ledger
        self.events = events
        self.transport = transport
        # Optional proof-of-storage audit engine (StorageAuditEngine).  When set,
        # verification material is recorded at shard placement time and the audit
        # loop can challenge nodes on demand.  None disables auditing (backward compat).
        self.audit = audit
        self._lock = asyncio.Lock()   # serialize put/delete/reconcile metadata mutations
        # Maps blob_id -> (k, m) for objects stored via erasure coding.
        # Persisted in-band via replication_desired (k*256+m) and chunks presence;
        # this dict caches the decode to avoid recomputing on every access.
        self._ec_params: dict[str, tuple[int, int]] = {}
        # Per-blob repair threshold (min live pieces before triggering repair).
        # Keyed by blob_id; populated at put time from the EC profile.
        # Missing key → default (total = k+m): repair whenever any shard is absent.
        # Lost on restart; reverts to the conservative (immediate-repair) default.
        self._repair_thresholds: dict[str, int] = {}
        self._bucket_owners: dict[str, str] = {}
        self._bucket_acls: dict[tuple[str, str], set[str]] = {}
        self._object_versions: dict[tuple[str, str], list[dict]] = {}
        self._load_bucket_indexes()

    # --- bucket ACLs and object versions --------------------------------- #
    def create_bucket(self, bucket: str, owner: str) -> dict:
        """Create (or return) a bucket owned by ``owner``.

        ACLs are per-bucket and checked before versioned object operations.  The
        storage layer remains byte-opaque: client-side encryption packages are
        already ciphertext before they reach these methods.
        """
        self._validate_bucket_key(bucket, "bucket")
        if bucket not in self._bucket_owners:
            self._bucket_owners[bucket] = owner
            self._bucket_acls[(bucket, owner)] = {"read", "write", "admin"}
            self._persist_bucket(bucket)
        return {"bucket": bucket, "owner": self._bucket_owners[bucket],
                "acl": self.bucket_acl(bucket)}

    def bucket_acl(self, bucket: str) -> dict[str, list[str]]:
        return {subject: sorted(perms)
                for (b, subject), perms in self._bucket_acls.items() if b == bucket}

    def grant_bucket(self, bucket: str, actor: str, grantee: str,
                     permissions: list[str]) -> dict:
        self._require_bucket(bucket, actor, "admin")
        valid = {"read", "write", "admin"}
        perms = {p for p in permissions if p in valid}
        if not perms:
            raise StorageError("ACL grant must include at least one of read, write, admin")
        self._bucket_acls[(bucket, grantee)] = set(perms)
        self._persist_bucket(bucket)
        return {"bucket": bucket, "grantee": grantee, "permissions": sorted(perms)}

    async def put_object_version(self, data: bytes, *, actor: str, bucket: str, key: str,
                                 name: str = "", replication: int = DEFAULT_REPLICATION,
                                 erasure: bool = True) -> dict:
        self._validate_bucket_key(bucket, "bucket")
        self._validate_bucket_key(key, "key")
        if bucket not in self._bucket_owners:
            self.create_bucket(bucket, actor)
        self._require_bucket(bucket, actor, "write")
        version_id = secrets.token_hex(16)
        meta = await self.put(data, owner=actor, name=name or f"{bucket}/{key}",
                              replication=replication, erasure=erasure)
        rec = {
            "bucket": bucket,
            "key": key,
            "version_id": version_id,
            "blob_id": meta.id,
            "size": len(data),
            "created_ms": int(time.time() * 1000),
            "owner": actor,
        }
        self._object_versions.setdefault((bucket, key), []).append(rec)
        self._persist_version(rec)
        return dict(rec)

    async def get_object_version(self, *, actor: str, bucket: str, key: str,
                                 version_id: str | None = None) -> bytes | None:
        self._require_bucket(bucket, actor, "read")
        versions = self._object_versions.get((bucket, key), [])
        if not versions:
            return None
        rec = versions[-1] if version_id is None else next(
            (v for v in versions if v["version_id"] == version_id), None)
        if rec is None:
            return None
        return await self.get(rec["blob_id"])

    def list_object_versions(self, *, actor: str, bucket: str, key: str) -> list[dict]:
        self._require_bucket(bucket, actor, "read")
        return [dict(v) for v in self._object_versions.get((bucket, key), [])]

    def list_all_buckets(self, *, actor: str) -> list[dict]:
        """Return all buckets the actor can read."""
        result = []
        for bucket, owner in self._bucket_owners.items():
            if actor == owner:
                result.append({"bucket": bucket, "owner": owner, "created_ms": 0})
            else:
                perms = self._bucket_acls.get((bucket, actor), set())
                if "read" in perms or "admin" in perms:
                    result.append({"bucket": bucket, "owner": owner, "created_ms": 0})
        return result

    def list_bucket_objects(self, *, actor: str, bucket: str) -> list[dict]:
        """Return the latest version record for every key in a bucket."""
        self._require_bucket(bucket, actor, "read")
        result = []
        for (b, key), versions in self._object_versions.items():
            if b == bucket and versions:
                result.append(dict(versions[-1]))
        return result

    async def delete_object_version(self, *, actor: str, bucket: str,
                                     key: str) -> bool:
        """Delete the latest object version and its blob.  Returns False if not found."""
        self._require_bucket(bucket, actor, "write")
        versions = self._object_versions.get((bucket, key), [])
        if not versions:
            return False
        latest = versions[-1]
        blob_id = latest.get("blob_id", "")
        self._object_versions[(bucket, key)] = []
        if blob_id:
            try:
                await self.delete(blob_id)
            except Exception:
                pass
        return True

    def delete_bucket(self, *, bucket: str, actor: str) -> bool:
        """Delete an empty bucket.  Returns False if it does not exist.
        Raises StorageError if the bucket still has objects."""
        if bucket not in self._bucket_owners:
            return False
        self._require_bucket(bucket, actor, "admin")
        for (b, _key), versions in self._object_versions.items():
            if b == bucket and versions:
                raise StorageError("bucket is not empty")
        del self._bucket_owners[bucket]
        for k in [k for k in list(self._bucket_acls.keys()) if k[0] == bucket]:
            del self._bucket_acls[k]
        meta_id = _BUCKET_META_PREFIX + hashlib.sha256(bucket.encode("utf-8")).hexdigest()
        try:
            self.store.delete_blob(meta_id)
        except Exception:
            pass
        return True

    def _require_bucket(self, bucket: str, actor: str, permission: str) -> None:
        owner = self._bucket_owners.get(bucket)
        if owner is None:
            raise StorageError("unknown bucket")
        if actor == owner:
            return
        perms = self._bucket_acls.get((bucket, actor), set())
        if permission in perms or "admin" in perms:
            return
        raise PermissionError(f"{actor} lacks {permission} access to bucket {bucket}")

    @staticmethod
    def _validate_bucket_key(value: str, field: str) -> None:
        if not value or ".." in value or value.startswith("/") or value.startswith("\\"):
            raise StorageError(f"invalid {field}")

    def _persist_bucket(self, bucket: str) -> None:
        body = {"kind": "bucket", "bucket": bucket,
                "owner": self._bucket_owners[bucket],
                "acl": self.bucket_acl(bucket)}
        self.store.put_blob(BlobMeta(
            id=_BUCKET_META_PREFIX + hashlib.sha256(bucket.encode("utf-8")).hexdigest(),
            owner=self._bucket_owners[bucket],
            size=0, name=json.dumps(body, sort_keys=True),
            replicas=[], replication_desired=0, status="stored", internal=True))

    def _persist_version(self, rec: dict) -> None:
        raw = json.dumps(rec, sort_keys=True)
        self.store.put_blob(BlobMeta(
            id=_VERSION_META_PREFIX + hashlib.sha256(raw.encode("utf-8")).hexdigest(),
            owner=rec["owner"], size=rec["size"], name=raw,
            replicas=[], replication_desired=0, status="stored", internal=True,
            created_ms=rec["created_ms"]))

    def _load_bucket_indexes(self) -> None:
        try:
            rows = self.store.list_blobs()
        except Exception:
            return
        for meta in rows:
            if not meta.internal or not meta.id.startswith(_BUCKET_META_PREFIX):
                continue
            try:
                body = json.loads(meta.name)
                bucket = str(body["bucket"])
                self._bucket_owners[bucket] = str(body["owner"])
                for subject, perms in dict(body.get("acl") or {}).items():
                    self._bucket_acls[(bucket, str(subject))] = set(perms)
            except Exception:
                continue
        for meta in rows:
            if not meta.internal or not meta.id.startswith(_VERSION_META_PREFIX):
                continue
            try:
                rec = json.loads(meta.name)
                self._object_versions.setdefault(
                    (str(rec["bucket"]), str(rec["key"])), []).append(rec)
            except Exception:
                continue
        for versions in self._object_versions.values():
            versions.sort(key=lambda v: int(v.get("created_ms", 0)))

    # --- placement ------------------------------------------------------- #
    def _pick_nodes(self, count: int, exclude: set) -> list:
        """Up to ``count`` schedulable storage nodes not in ``exclude``.

        Nodes are sorted by available disk space (roomiest first) and then
        reordered by pick_diverse_nodes to maximise failure-domain spread.
        Adding more nodes to the pool never reduces the returned set -- the
        diversity layer is a soft tie-breaker, not a hard filter.
        """
        from ..controlplane.topology import pick_diverse_nodes
        nodes = [n for n in self.registry.schedulable_nodes()
                 if n.id not in exclude and n.provides("storage")]
        # Pre-sort by disk so that within a domain we prefer the roomiest node.
        nodes.sort(key=lambda n: n.capability.disk_mb, reverse=True)
        return pick_diverse_nodes(nodes, max(0, count))

    @staticmethod
    def digest(data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()

    # --- erasure-coding ops ---------------------------------------------- #
    async def _store_ec_leaf_locked(self, blob_id: str, data: bytes,
                                    k: int, m: int, *,
                                    owner: str, name: str,
                                    internal: bool,
                                    repair_threshold: int = 0) -> BlobMeta:
        """Erasure-code one blob and store k+m shards on distinct nodes.

        ``repair_threshold`` is the minimum live shard count before the repair
        loop fires.  0 (default) means ``k+m`` -- repair whenever any shard is
        missing (conservative / legacy behaviour).  Named tiers set this from
        ``EC_PROFILES``.
        """
        codec = RSCodec(k, m)
        shards = codec.encode(data)  # list of k+m bytes objects
        total = k + m

        existing = self.store.get_blob(blob_id)
        existing_shard_ids: list[str] = (
            list(existing.chunks) if existing and existing.chunks else [])

        # Derive a deterministic shard blob ID for each position.
        # The blob ID is sha256 of the full shard bytes (including the header),
        # which is exactly what we PUT on the node — the transport verifies this.
        shard_ids: list[str] = [self.digest(raw) for raw in shards]

        # Ensure we have enough nodes — one per shard
        live = {n.id for n in self.registry.online()}
        already_placed: dict[int, str] = {}  # shard_index -> node_id
        if existing and len(existing.chunks) == total:
            # reuse shards already on live nodes
            for idx, sid in enumerate(existing.chunks):
                cm = self.store.get_blob(sid)
                if cm and cm.replicas:
                    nid = cm.replicas[0]
                    if nid in live:
                        already_placed[idx] = nid

        placed_nodes = set(already_placed.values())
        shard_meta: list[Optional[str]] = [None] * total

        for idx in sorted(already_placed):
            shard_meta[idx] = already_placed[idx]

        # Place remaining shards
        need_nodes = [i for i in range(total) if shard_meta[i] is None]
        candidates = self._pick_nodes(len(need_nodes) + 5, exclude=placed_nodes)
        candidate_iter = iter(candidates)

        for idx in need_nodes:
            node = next(candidate_iter, None)
            if node is None:
                break
            payload = base64.b64encode(shards[idx]).decode("ascii")
            try:
                await self.transport.call(
                    node.id, "blob_put",
                    {"blob_id": shard_ids[idx], "data": payload}, timeout=30.0)
                shard_meta[idx] = node.id
                placed_nodes.add(node.id)
            except TransportError as exc:
                log.warning("ec shard_put failed", node=node.id, shard=idx,
                            err=str(exc)[:120])

        placed_count = sum(1 for n in shard_meta if n is not None)
        if placed_count < k:
            raise StorageError(
                f"only {placed_count}/{total} shards placed; need at least {k}")

        # Store each shard blob's metadata so the store knows where shards live
        for idx, nid in enumerate(shard_meta):
            if nid is None:
                continue
            shard_blob = BlobMeta(
                id=shard_ids[idx], owner=owner, size=len(shards[idx]),
                name="", replicas=[nid], replication_desired=1,
                status="stored", internal=True)
            self.store.put_blob(shard_blob)
            # Record per-chunk hashes for the audit engine so the coordinator can
            # challenge shard holders without keeping the full shard bytes.
            if self.audit is not None:
                self.audit.record_shard_hashes(shard_ids[idx], shards[idx])

        # Record EC params so we can decode without re-reading the manifest header
        self._ec_params[blob_id] = (k, m)
        # Record repair threshold (0 → total, i.e. repair immediately on any loss)
        self._repair_thresholds[blob_id] = repair_threshold or (k + m)

        status = "stored" if placed_count == total else "degraded"
        meta = BlobMeta(
            id=blob_id, owner=owner, size=len(data),
            name=name or (existing.name if existing else ""),
            replicas=[],   # shards tracked in chunks
            replication_desired=_pack_ec_params(k, m),
            status=status, chunks=shard_ids, internal=internal)
        self.store.put_blob(meta)
        if not internal:
            self.events.publish("blob_stored",
                                {"id": blob_id, "ec_k": k, "ec_m": m,
                                 "shards_placed": placed_count,
                                 "shards_total": total, "size": len(data)})
        return meta

    async def _store_ec_chunked_locked(self, blob_id: str, data: bytes,
                                       k: int, m: int, *,
                                       owner: str, name: str,
                                       internal: bool = False,
                                       repair_threshold: int = 0) -> BlobMeta:
        """Split a large object into chunk-sized pieces, EC-protect each."""
        size = len(data)
        step = _chunk_size()
        existing = self.store.get_blob(blob_id)
        chunk_ids: list[str] = []
        for off in range(0, size, step):
            chunk = data[off:off + step]
            cid = self.digest(chunk)
            await self._store_ec_leaf_locked(
                cid, chunk, k, m, owner=owner, name="", internal=True,
                repair_threshold=repair_threshold)
            chunk_ids.append(cid)
        manifest = BlobMeta(
            id=blob_id, owner=owner, size=size,
            name=name or (existing.name if existing else ""),
            replicas=[], replication_desired=_pack_ec_params(k, m),
            status=self._ec_chunk_status(chunk_ids, k),
            chunks=chunk_ids, internal=internal)
        self.store.put_blob(manifest)
        if not internal:
            self.events.publish("blob_stored",
                                {"id": blob_id, "ec_k": k, "ec_m": m,
                                 "chunks": len(chunk_ids), "size": size})
        return manifest

    def _ec_chunk_status(self, chunk_ids: list[str], k: int) -> str:
        """Aggregate health of an EC-protected chunked manifest."""
        worst = "stored"
        for cid in chunk_ids:
            cm = self.store.get_blob(cid)
            if cm is None:
                return "lost"
            placed = sum(1 for sid in cm.chunks
                         if self.store.get_blob(sid) is not None
                         and self.store.get_blob(sid).replicas)
            if placed == 0:
                return "lost"
            if placed < k:
                worst = "degraded"
        return worst

    async def _fetch_ec(self, meta: BlobMeta) -> Optional[bytes]:
        """Reconstruct an EC-protected blob from its shards."""
        k, m = self._ec_params.get(meta.id) or _unpack_ec_params(meta.replication_desired)
        codec = RSCodec(k, m)
        total = k + m
        shard_ids = meta.chunks
        if len(shard_ids) != total:
            log.warning("EC manifest has wrong shard count", blob=meta.id[:12],
                        expected=total, got=len(shard_ids))
            return None

        shards: list[Optional[bytes]] = []
        for sid in shard_ids:
            sm = self.store.get_blob(sid)
            if sm is None or not sm.replicas:
                shards.append(None)
                continue
            raw = await self._fetch(sm.replicas, sid)
            shards.append(raw)   # None if all replicas failed

        try:
            return codec.decode(shards)
        except (ValueError, Exception) as exc:
            log.warning("EC decode failed", blob=meta.id[:12], err=str(exc)[:120])
            return None

    async def _repair_ec_locked(self, meta: BlobMeta) -> None:
        """Regenerate missing EC shards onto new nodes."""
        k, m = self._ec_params.get(meta.id) or _unpack_ec_params(meta.replication_desired)
        codec = RSCodec(k, m)
        total = k + m
        shard_ids = list(meta.chunks)
        live = {n.id for n in self.registry.online()}

        # Fetch what we can
        shards: list[Optional[bytes]] = []
        for sid in shard_ids:
            sm = self.store.get_blob(sid)
            if sm is None or not sm.replicas or not any(r in live for r in sm.replicas):
                shards.append(None)
            else:
                active = [r for r in sm.replicas if r in live]
                raw = await self._fetch(active, sid)
                shards.append(raw)

        valid_count = sum(1 for s in shards if s is not None)
        if valid_count < k:
            log.warning("EC repair: not enough shards to reconstruct", blob=meta.id[:12],
                        have=valid_count, need=k)
            return

        # Reconstruct all shards (so we can re-place missing ones)
        try:
            original = codec.decode(shards)
        except Exception as exc:  # noqa: BLE001
            log.error("EC repair decode failed", blob=meta.id[:12], err=str(exc)[:120])
            return

        all_shards = codec.encode(original)
        new_shard_ids = [self.digest(s) for s in all_shards]

        placed_nodes = set()
        for sid in shard_ids:
            sm = self.store.get_blob(sid)
            if sm and sm.replicas:
                placed_nodes.update(r for r in sm.replicas if r in live)

        for idx, (sid, raw_shard) in enumerate(zip(shard_ids, all_shards)):
            sm = self.store.get_blob(sid)
            if sm and sm.replicas and any(r in live for r in sm.replicas):
                continue   # shard still alive
            # need to place this shard on a new node
            nodes = self._pick_nodes(1, exclude=placed_nodes)
            if not nodes:
                log.warning("EC repair: no nodes to place shard", idx=idx)
                continue
            node = nodes[0]
            payload = base64.b64encode(raw_shard).decode("ascii")
            new_sid = new_shard_ids[idx]
            try:
                await self.transport.call(
                    node.id, "blob_put",
                    {"blob_id": new_sid, "data": payload}, timeout=30.0)
                shard_blob = BlobMeta(
                    id=new_sid, owner=meta.owner, size=len(raw_shard),
                    name="", replicas=[node.id], replication_desired=1,
                    status="stored", internal=True)
                self.store.put_blob(shard_blob)
                if self.audit is not None:
                    # A repaired shard must be auditable too. new_sid normally
                    # equals the old id (RS encoding is deterministic, ids are
                    # content addresses), but re-recording makes the audit path
                    # correct without depending on that coincidence.
                    self.audit.record_shard_hashes(new_sid, raw_shard)
                shard_ids[idx] = new_sid
                placed_nodes.add(node.id)
            except TransportError as exc:
                log.warning("EC repair shard_put failed", node=node.id, err=str(exc)[:120])

        # Update manifest with any new shard IDs
        placed_count = sum(
            1 for sid in shard_ids
            if self.store.get_blob(sid) is not None
            and self.store.get_blob(sid).replicas)
        status = "stored" if placed_count == total else ("degraded" if placed_count >= k else "lost")
        meta.chunks = shard_ids
        meta.replicas = []
        meta.status = status
        self.store.put_blob(meta)
        self.events.publish("blob_health",
                            {"id": meta.id, "ec_k": k, "ec_m": m,
                             "shards_placed": placed_count, "shards_total": total,
                             "status": status})

    # --- core ops -------------------------------------------------------- #
    async def put(self, data: bytes, owner: str = "default", name: str = "",
                  replication: int = DEFAULT_REPLICATION,
                  erasure: bool = True,
                  ec_k: int = DEFAULT_EC_K,
                  ec_m: int = DEFAULT_EC_M,
                  tier: str = "",
                  internal: bool = False) -> BlobMeta:
        """Store ``data`` in the pool.

        By default uses Reed–Solomon (ec_k, ec_m) erasure coding, which gives
        lower storage overhead and higher durability than plain replication.
        Set ``erasure=False`` to fall back to ``replication``-way replication.

        ``tier`` selects a named erasure profile (``"hot"``, ``"warm"``,
        ``"cold"``).  When set it overrides ``ec_k`` and ``ec_m`` and also
        sets the Storj-inspired repair threshold for the object.  An empty
        string (the default) keeps the explicit ``ec_k``/``ec_m`` values and
        uses the conservative "repair immediately" threshold.

        RS(10,30) hot requires 30 nodes; RS(20,60) warm requires 60; RS(29,80)
        cold requires 80.  On a smaller cluster the engine falls back
        automatically to 2× replication and logs the reason.  Real durability
        numbers require a deployed multi-datacentre fleet -- the emulated sim
        tests on a single host are labelled accordingly.

        Very small objects (below the per-shard-header crossover) automatically
        fall back to replication even when ``erasure=True``.

        Idempotent: re-putting identical bytes tops the existing object back up
        to the desired durability rather than duplicating it.

        An object larger than the chunk size is split into content-addressed
        chunks, each protected independently across the pool and recorded in a
        manifest, so an object bigger than any single machine's free disk fits.

        ``internal=True`` marks the object as machinery so it is hidden from
        user-facing listings.
        """
        # Resolve named tier -> ec parameters + repair threshold
        if tier in EC_PROFILES:
            prof = EC_PROFILES[tier]
            ec_k = prof["k"]
            ec_m = prof["m"]
            repair_threshold = prof["repair_threshold"]
        else:
            repair_threshold = ec_k + ec_m   # repair immediately on any loss (legacy)

        blob_id = self.digest(data)
        async with self._lock:
            size = len(data)
            available_nodes = len(self.registry.schedulable_nodes())
            # Decide whether EC is actually worthwhile for this size and cluster
            use_ec = (erasure and size >= _ec_crossover(ec_k, ec_m)
                      and ec_k + ec_m <= 255
                      and available_nodes >= ec_k + ec_m)
            if not use_ec and erasure and tier:
                log.info("ec tier fallback: cluster too small for requested profile",
                         tier=tier, need=ec_k + ec_m, have=available_nodes)
            if use_ec:
                if size <= _chunk_size():
                    return await self._store_ec_leaf_locked(
                        blob_id, data, ec_k, ec_m,
                        owner=owner, name=name, internal=internal,
                        repair_threshold=repair_threshold)
                return await self._store_ec_chunked_locked(
                    blob_id, data, ec_k, ec_m,
                    owner=owner, name=name, internal=internal,
                    repair_threshold=repair_threshold)
            # Replication path (small objects or explicit request)
            if size <= _chunk_size():
                return await self._store_leaf_locked(
                    blob_id, data, replication, owner=owner, name=name, internal=internal)
            return await self._store_chunked_locked(
                blob_id, data, replication, owner=owner, name=name, internal=internal)

    async def _store_leaf_locked(self, blob_id: str, data: bytes, replication: int, *,
                                 owner: str, name: str, internal: bool) -> BlobMeta:
        """Replicate one content-addressed blob (a plain object, or a single chunk of a
        larger one) to ``replication`` nodes, reusing any live copies. Idempotent. The
        caller must hold ``self._lock``."""
        payload = base64.b64encode(data).decode("ascii")
        live = {n.id for n in self.registry.online()}
        existing = self.store.get_blob(blob_id)
        placed = [nid for nid in existing.replicas if nid in live] if existing else []
        need = max(0, replication - len(placed))
        for node in self._pick_nodes(need, exclude=set(placed)):
            try:
                await self.transport.call(
                    node.id, "blob_put",
                    {"blob_id": blob_id, "data": payload}, timeout=30.0)
                placed.append(node.id)
            except TransportError as exc:
                log.warning("blob_put failed", node=node.id, err=str(exc)[:120])
        if not placed:
            raise StorageError("no schedulable node accepted the object")
        meta = BlobMeta(
            id=blob_id, owner=owner, size=len(data),
            name=name or (existing.name if existing else ""),
            replicas=placed, replication_desired=replication,
            status=self._status(len(placed), replication), internal=internal)
        self.store.put_blob(meta)
        if not internal:
            self.events.publish("blob_stored",
                                {"id": blob_id, "replicas": len(placed),
                                 "desired": replication, "size": len(data)})
        return meta

    async def _store_chunked_locked(self, blob_id: str, data: bytes, replication: int, *,
                                    owner: str, name: str, internal: bool = False) -> BlobMeta:
        """Split ``data`` into chunk-sized pieces, replicate each across the pool as an
        ordinary blob, and record a manifest listing them. Caller must hold ``self._lock``."""
        size = len(data)
        step = _chunk_size()
        existing = self.store.get_blob(blob_id)
        chunk_ids: list[str] = []
        for off in range(0, size, step):
            chunk = data[off:off + step]
            cid = self.digest(chunk)
            await self._store_leaf_locked(
                cid, chunk, replication, owner=owner, name="", internal=True)
            chunk_ids.append(cid)
        manifest = BlobMeta(
            id=blob_id, owner=owner, size=size,
            name=name or (existing.name if existing else ""),
            replicas=[], replication_desired=replication,
            status=self._chunk_status(chunk_ids), chunks=chunk_ids, internal=internal)
        self.store.put_blob(manifest)
        if not internal:
            self.events.publish("blob_stored",
                                {"id": blob_id, "chunks": len(chunk_ids),
                                 "desired": replication, "size": size})
        return manifest

    def _chunk_status(self, chunk_ids) -> str:
        """A manifest's health is the worst health across its chunks."""
        worst = "stored"
        for cid in chunk_ids:
            cm = self.store.get_blob(cid)
            s = cm.status if cm else "lost"
            if s == "lost":
                return "lost"
            if s == "degraded":
                worst = "degraded"
        return worst

    def _is_ec(self, meta: BlobMeta) -> bool:
        """Return True if this blob is erasure-coded (not plain-replicated)."""
        return (meta.id in self._ec_params
                or (meta.chunks and meta.replication_desired >= 256))

    async def get(self, blob_id: str) -> Optional[bytes]:
        meta = self.store.get_blob(blob_id)
        if meta is None:
            return None
        if meta.chunks:
            if self._is_ec(meta):
                # EC leaf object: shards are in meta.chunks
                return await self._fetch_ec(meta)
            # Regular chunked object: each chunk is a replicated leaf blob
            parts: list[bytes] = []
            for cid in meta.chunks:
                cm = self.store.get_blob(cid)
                if cm is None:
                    return None
                # EC-protected chunk?
                if self._is_ec(cm):
                    raw = await self._fetch_ec(cm)
                else:
                    raw = await self._fetch(cm.replicas, cid)
                if raw is None:
                    return None
                parts.append(raw)
            whole = b"".join(parts)
            if self.digest(whole) != blob_id:
                log.warning("chunked object integrity check failed", blob=blob_id[:12])
                return None
            return whole
        return await self._fetch(meta.replicas, blob_id)

    async def delete(self, blob_id: str) -> bool:
        async with self._lock:
            meta = self.store.get_blob(blob_id)
            if meta is None:
                return False
            if meta.chunks:
                for cid in meta.chunks:
                    if self._chunk_referenced_elsewhere(cid, blob_id):
                        continue   # another object dedupes onto this chunk; keep it
                    cm = self.store.get_blob(cid)
                    if cm is None:
                        continue
                    for nid in cm.replicas:
                        try:
                            await self.transport.call(nid, "blob_delete",
                                                      {"blob_id": cid}, timeout=4.0)
                        except TransportError:
                            pass
                    self.store.delete_blob(cid)
                    if self.audit is not None:
                        # Release the shard's verification material, otherwise
                        # _chunk_hashes grows without bound across delete cycles.
                        self.audit.forget_shard(cid)
                self.store.delete_blob(blob_id)
                self.events.publish("blob_removed",
                                    {"id": blob_id, "chunks": len(meta.chunks)})
                return True
            for nid in meta.replicas:
                try:
                    await self.transport.call(nid, "blob_delete",
                                              {"blob_id": blob_id}, timeout=4.0)
                except TransportError:
                    pass   # best-effort; a dead node's copy is orphaned and reaped later
            self.store.delete_blob(blob_id)
            self.events.publish("blob_removed", {"id": blob_id})
            return True

    def _chunk_referenced_elsewhere(self, cid: str, exclude_id: str) -> bool:
        """True if any OTHER manifest still lists ``cid`` (content-addressed dedup means
        two large objects can share an identical chunk — don't delete it out from under
        the survivor)."""
        for other in self.store.list_blobs():
            if other.id != exclude_id and other.chunks and cid in other.chunks:
                return True
        return False

    def list(self, owner: Optional[str] = None) -> list:
        blobs = self.store.list_blobs()
        return [b for b in blobs
                if not b.internal and (owner is None or b.owner == owner)]

    def get_meta(self, blob_id: str) -> Optional[BlobMeta]:
        return self.store.get_blob(blob_id)

    def note_replica(self, blob_id: str, node_id: str) -> bool:
        """Record that ``node_id`` now holds a full copy of ``blob_id``.

        Distribution engines call this when a node acquires an object outside the
        normal replication path -- an inference stage pulling the weight groups for
        its layers, say. Without it the pool's picture of who holds what is frozen at
        whatever the original ``put`` decided, so every later reader is routed to the
        same original holders no matter how many nodes already have the bytes sitting
        on disk. Recording the acquisition turns a fan-out from a fixed origin into a
        swarm, and it is also what lets the repair loop count copies that already
        exist instead of manufacturing more.

        Returns True if this was new information.
        """
        meta = self.store.get_blob(blob_id)
        if meta is None or node_id in meta.replicas:
            return False
        meta.replicas = [*meta.replicas, node_id]
        # Only ever *improves* health, and a chunked object's status is derived from
        # its chunks, so leave that one alone rather than overwriting it here.
        if not meta.chunks:
            meta.status = self._status(len(meta.replicas), meta.replication_desired)
        self.store.put_blob(meta)
        return True

    # --- durability ------------------------------------------------------ #
    def on_node_offline(self, node_id: str) -> None:
        """A node dropped: forget its copies so reads stop targeting it and the repair
        loop re-replicates anything now under its desired replica count."""
        for meta in self.store.list_blobs():
            if node_id in meta.replicas:
                meta.replicas = [n for n in meta.replicas if n != node_id]
                meta.status = self._status(len(meta.replicas), meta.replication_desired)
                self.store.put_blob(meta)
                self.events.publish("blob_replica_lost",
                                    {"id": meta.id, "node": node_id,
                                     "replicas": len(meta.replicas)})

    async def _reconcile(self) -> None:
        live = {n.id for n in self.registry.online()}
        for meta in self.store.list_blobs():
            if meta.chunks:
                if self._is_ec(meta):
                    # EC leaf: check shard health and trigger repair at threshold
                    k, m = (self._ec_params.get(meta.id)
                            or _unpack_ec_params(meta.replication_desired))
                    placed = sum(
                        1 for sid in meta.chunks
                        if (sm := self.store.get_blob(sid)) and sm.replicas
                        and any(r in live for r in sm.replicas))
                    total = k + m
                    # Storj-style threshold: repair when placed drops below
                    # repair_threshold (default = total → repair immediately).
                    # Threshold is set from the EC profile at put() time.
                    repair_threshold = self._repair_thresholds.get(meta.id, total)
                    new_status = ("stored" if placed == total
                                  else ("degraded" if placed >= k else "lost"))
                    if placed < repair_threshold and placed >= k:
                        # Below repair threshold AND enough shards to reconstruct
                        await self._repair_ec_locked(meta)
                        continue
                    if new_status != meta.status:
                        meta.status = new_status
                        self.store.put_blob(meta)
                        self.events.publish("blob_health",
                                            {"id": meta.id, "status": new_status,
                                             "shards": placed, "total": total,
                                             "repair_threshold": repair_threshold})
                    continue
                # Regular chunked manifest: refresh aggregate health
                agg = self._chunk_status(meta.chunks)
                if agg != meta.status:
                    meta.status = agg
                    self.store.put_blob(meta)
                    self.events.publish("blob_health",
                                        {"id": meta.id, "chunks": len(meta.chunks),
                                         "status": agg})
                continue
            placed = [nid for nid in meta.replicas if nid in live]
            changed = len(placed) != len(meta.replicas)
            # re-replicate an under-replicated object (as long as a live copy survives)
            if 0 < len(placed) < meta.replication_desired:
                data = await self._fetch(placed, meta.id)
                if data is not None:
                    payload = base64.b64encode(data).decode("ascii")
                    for node in self._pick_nodes(
                            meta.replication_desired - len(placed), exclude=set(placed)):
                        try:
                            await self.transport.call(
                                node.id, "blob_put",
                                {"blob_id": meta.id, "data": payload}, timeout=30.0)
                            placed.append(node.id)
                            changed = True
                        except TransportError:
                            continue
            if changed:
                meta.replicas = placed
                meta.status = self._status(len(placed), meta.replication_desired)
                self.store.put_blob(meta)
                self.events.publish("blob_health",
                                    {"id": meta.id, "replicas": len(placed),
                                     "desired": meta.replication_desired})

    async def run_loop(self, interval: float = 3.0) -> None:
        while True:
            try:
                async with self._lock:
                    await self._reconcile()
            except Exception as exc:  # noqa: BLE001
                log.error("storage reconcile error", err=str(exc)[:160])
            await asyncio.sleep(interval)

    # --- helpers --------------------------------------------------------- #
    async def _fetch(self, node_ids, blob_id: str) -> Optional[bytes]:
        """Pull a blob from the first replica that returns bytes matching its id. The
        digest check means a faulty or dishonest node can never hand back wrong data."""
        for nid in node_ids:
            try:
                body = await self.transport.call(
                    nid, "blob_get", {"blob_id": blob_id}, timeout=30.0)
            except TransportError:
                continue
            if not body.get("found"):
                continue
            try:
                raw = base64.b64decode(body.get("data", ""), validate=True)
            except (binascii.Error, ValueError):
                continue
            if self.digest(raw) == blob_id:
                return raw
            log.warning("blob integrity check failed", node=nid, blob=blob_id[:12])
        return None

    @staticmethod
    def _status(n: int, desired: int) -> str:
        if n >= desired:
            return "stored"
        return "degraded" if n > 0 else "lost"
