"""SQLite datastore with a tiny migration runner (spec §6: SQLite -> Postgres path).

Durable records (nodes, jobs, deployments, models, contribution ledger, trust
scores) are persisted here. Complex objects are stored as JSON in a ``data``
column with a few promoted, indexed columns for querying. The schema deliberately
includes Tier 1/2 trust columns now (inert at Tier 0) so no migration is needed
to enable them later (spec §13.5).

Durability contract
-------------------
WAL + synchronous=FULL ensures that every committed transaction survives an abrupt
OS/process kill (power loss, OOM kill).  See docs/durability.md for the full
locking and durability model.

Locking model (this file)
--------------------------
``Store._lock`` (threading.RLock) serialises **all** SQLite access — both reads
and writes — on a single shared connection.  The lock must be held any time
``self._conn`` is touched.  This is safe because the control plane is a
single-process service; read contention is negligible at its scale.

Atomic multi-step operations
-----------------------------
``add_ledger_pair()`` writes a donation entry and a consumption entry in a single
``BEGIN … COMMIT`` block.  Use it instead of two separate ``add_ledger()`` calls
so a crash between them cannot leave the ledger half-charged.

Idempotency
-----------
``add_ledger_idempotent()`` uses INSERT OR IGNORE keyed on the entry's ``id``
field.  Callers that derive the id deterministically (e.g. from unit_id + kind)
can call it safely on retries — the second call is a no-op.

Retention / compaction
-----------------------
``prune_ledger()`` and ``prune_completed_jobs()`` bound unbounded table growth.
Call them periodically (e.g. from a background task).
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import sqlite3
import threading
from typing import Any, Iterable, Optional

from .models import (
    BatchJob, BlobMeta, Deployment, LedgerEntry, ModelDeployment, Node, TrustScore,
)

SCHEMA_VERSION = 5

_MIGRATIONS: list[str] = [
    # v1 — initial schema
    """
    CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);

    CREATE TABLE IF NOT EXISTS nodes (
        id           TEXT PRIMARY KEY,
        owner        TEXT NOT NULL,
        status       TEXT NOT NULL,
        trust_tier   INTEGER NOT NULL DEFAULT 0,
        trust_score  REAL NOT NULL DEFAULT 1.0,
        updated_ms   INTEGER NOT NULL,
        data         TEXT NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_nodes_owner ON nodes(owner);

    CREATE TABLE IF NOT EXISTS jobs (
        id         TEXT PRIMARY KEY,
        owner      TEXT NOT NULL,
        status     TEXT NOT NULL,
        created_ms INTEGER NOT NULL,
        data       TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS deployments (
        id         TEXT PRIMARY KEY,
        owner      TEXT NOT NULL,
        status     TEXT NOT NULL,
        created_ms INTEGER NOT NULL,
        data       TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS models (
        id         TEXT PRIMARY KEY,
        owner      TEXT NOT NULL,
        status     TEXT NOT NULL,
        created_ms INTEGER NOT NULL,
        data       TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS ledger (
        id            TEXT PRIMARY KEY,
        node_id       TEXT NOT NULL,
        owner         TEXT NOT NULL,
        kind          TEXT NOT NULL,
        compute_units REAL NOT NULL,
        ts_ms         INTEGER NOT NULL,
        detail        TEXT NOT NULL DEFAULT ''
    );
    CREATE INDEX IF NOT EXISTS idx_ledger_node ON ledger(node_id);
    CREATE INDEX IF NOT EXISTS idx_ledger_owner ON ledger(owner);

    CREATE TABLE IF NOT EXISTS trust_scores (
        node_id TEXT PRIMARY KEY,
        data    TEXT NOT NULL
    );
    """,
    # v2 — optional accounts + device linking + one-time starter grants (Tier 2
    # global mesh). Anonymous devices never need a row here; these tables only hold
    # the *optional* account overlay and the anti-sybil "granted already" markers.
    """
    CREATE TABLE IF NOT EXISTS accounts (
        id         TEXT PRIMARY KEY,
        handle     TEXT NOT NULL,
        provider   TEXT NOT NULL,
        created_ms INTEGER NOT NULL
    );

    CREATE TABLE IF NOT EXISTS device_links (
        device_id  TEXT PRIMARY KEY,
        account_id TEXT NOT NULL,
        linked_ms  INTEGER NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_device_links_account ON device_links(account_id);

    CREATE TABLE IF NOT EXISTS starter_grants (
        subject TEXT PRIMARY KEY,
        ts_ms   INTEGER NOT NULL
    );
    """,
    # v3 — durable per-node session tokens. These are server-side bearer-token
    # mappings used by reconnecting agents after a coordinator restart.
    """
    CREATE TABLE IF NOT EXISTS node_tokens (
        node_id    TEXT PRIMARY KEY,
        token      TEXT NOT NULL,
        updated_ms INTEGER
    );
    CREATE INDEX IF NOT EXISTS idx_node_tokens_token ON node_tokens(token);
    """,
    # v4 — pooled content-addressed object store. Metadata only (which nodes hold
    # each blob); the bytes live on the agents' donated disk, keyed by SHA-256 digest.
    """
    CREATE TABLE IF NOT EXISTS blobs (
        id         TEXT PRIMARY KEY,
        owner      TEXT NOT NULL,
        status     TEXT NOT NULL,
        created_ms INTEGER NOT NULL,
        data       TEXT NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_blobs_owner ON blobs(owner);
    """,
    # v5 — user-minted capability tokens. Only a SHA-256 hash of the bearer secret
    # is stored; listing tokens can never disclose material usable for auth.
    """
    CREATE TABLE IF NOT EXISTS capability_tokens (
        id             TEXT PRIMARY KEY,
        owner          TEXT NOT NULL,
        token_hash     TEXT NOT NULL UNIQUE,
        label          TEXT NOT NULL,
        scopes_json    TEXT NOT NULL,
        expires_at_ms  INTEGER NOT NULL,
        created_ms     INTEGER NOT NULL,
        last_used_ms   INTEGER,
        revoked_ms     INTEGER,
        spend_cap      REAL,
        spent          REAL NOT NULL DEFAULT 0.0
    );
    CREATE INDEX IF NOT EXISTS idx_capability_tokens_owner ON capability_tokens(owner);
    CREATE INDEX IF NOT EXISTS idx_capability_tokens_hash ON capability_tokens(token_hash);
    """,
]


class Store:
    """Thread-safe SQLite wrapper. Safe to share across the async control plane.

    ``_lock`` (RLock) is held for **every** access to ``_conn`` — reads and
    writes alike — so callers never need to reason about SQLite's thread-safety
    quirks.  See the module docstring for the full durability and locking model.
    """

    def __init__(self, path: str) -> None:
        self.path = path
        # Single reentrant lock serialises all SQLite access on this connection.
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        # WAL: readers never block writers; writers never block readers.
        self._conn.execute("PRAGMA journal_mode=WAL;")
        # FULL: the WAL file is fsynced before each checkpoint so every committed
        # transaction survives a power loss or OOM kill.  (WAL's default NORMAL
        # skips the fsync, which is fast but not crash-safe.)
        self._conn.execute("PRAGMA synchronous=FULL;")
        self._conn.execute("PRAGMA foreign_keys=ON;")
        self._migrate()

    # --- migrations ------------------------------------------------------ #
    def _migrate(self) -> None:
        with self._lock:
            current = self._current_version()
            for version in range(current, len(_MIGRATIONS)):
                self._conn.executescript(_MIGRATIONS[version])
                current = version + 1
            self._conn.execute(
                "INSERT INTO meta(key,value) VALUES('schema_version',?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (str(current),),
            )
            self._conn.commit()

    def _table_exists(self, name: str) -> bool:
        row = self._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (name,)
        ).fetchone()
        return row is not None

    def _current_version(self) -> int:
        if not self._table_exists("meta"):
            return 0
        row = self._conn.execute(
            "SELECT value FROM meta WHERE key='schema_version'"
        ).fetchone()
        return int(row[0]) if row else 0

    # --- nodes ----------------------------------------------------------- #
    def put_node(self, node: Node) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO nodes(id,owner,status,trust_tier,trust_score,updated_ms,data) "
                "VALUES(?,?,?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET "
                "owner=excluded.owner,status=excluded.status,trust_tier=excluded.trust_tier,"
                "trust_score=excluded.trust_score,updated_ms=excluded.updated_ms,data=excluded.data",
                (node.id, node.owner, node.status.value, int(node.trust_tier),
                 node.trust_score, node.last_heartbeat_ms, node.model_dump_json()),
            )
            self._conn.commit()

    def get_node(self, node_id: str) -> Optional[Node]:
        with self._lock:
            row = self._conn.execute("SELECT data FROM nodes WHERE id=?", (node_id,)).fetchone()
        return Node.model_validate_json(row["data"]) if row else None

    def list_nodes(self) -> list[Node]:
        with self._lock:
            rows = self._conn.execute("SELECT data FROM nodes").fetchall()
        return [Node.model_validate_json(r["data"]) for r in rows]

    def delete_node(self, node_id: str) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM nodes WHERE id=?", (node_id,))
            self._conn.commit()

    # --- node session tokens --------------------------------------------- #
    def put_node_token(self, node_id: str, token: str, ts_ms: int) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO node_tokens(node_id,token,updated_ms) VALUES(?,?,?) "
                "ON CONFLICT(node_id) DO UPDATE SET token=excluded.token,"
                "updated_ms=excluded.updated_ms",
                (node_id, token, ts_ms),
            )
            self._conn.commit()

    def load_node_tokens(self) -> dict[str, str]:
        with self._lock:
            rows = self._conn.execute("SELECT node_id,token FROM node_tokens").fetchall()
        return {r["node_id"]: r["token"] for r in rows}

    def delete_node_token(self, node_id: str) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM node_tokens WHERE node_id=?", (node_id,))
            self._conn.commit()

    # --- user capability tokens ------------------------------------------ #
    @staticmethod
    def capability_token_hash(secret: str) -> str:
        return hashlib.sha256(secret.encode("utf-8")).hexdigest()

    @staticmethod
    def _capability_token_row_to_dict(row) -> dict:
        return {
            "id": row["id"],
            "owner": row["owner"],
            "token_hash": row["token_hash"],
            "label": row["label"],
            "scopes": json.loads(row["scopes_json"]),
            "expires_at_ms": row["expires_at_ms"],
            "created_ms": row["created_ms"],
            "last_used_ms": row["last_used_ms"],
            "revoked_ms": row["revoked_ms"],
            "spend_cap": row["spend_cap"],
            "spent": row["spent"],
        }

    def put_capability_token(self, token: dict) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO capability_tokens("
                "id,owner,token_hash,label,scopes_json,expires_at_ms,created_ms,"
                "last_used_ms,revoked_ms,spend_cap,spent"
                ") VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                (
                    token["id"], token["owner"], token["token_hash"], token["label"],
                    json.dumps(sorted(token["scopes"]), separators=(",", ":")),
                    int(token["expires_at_ms"]), int(token["created_ms"]),
                    token.get("last_used_ms"), token.get("revoked_ms"),
                    token.get("spend_cap"), float(token.get("spent", 0.0)),
                ),
            )
            self._conn.commit()

    def capability_token_by_secret(self, secret: str) -> Optional[dict]:
        token_hash = self.capability_token_hash(secret)
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM capability_tokens WHERE token_hash=?", (token_hash,)
            ).fetchone()
        return self._capability_token_row_to_dict(row) if row else None

    def list_capability_tokens(self, owner: str) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM capability_tokens WHERE owner=? "
                "ORDER BY created_ms DESC,id DESC", (owner,)
            ).fetchall()
        return [self._capability_token_row_to_dict(r) for r in rows]

    def get_capability_token(self, token_id: str) -> Optional[dict]:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM capability_tokens WHERE id=?", (token_id,)
            ).fetchone()
        return self._capability_token_row_to_dict(row) if row else None

    def mark_capability_token_used(self, token_id: str, ts_ms: int) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE capability_tokens SET last_used_ms=? "
                "WHERE id=? AND revoked_ms IS NULL", (ts_ms, token_id),
            )
            self._conn.commit()

    def revoke_capability_token(self, token_id: str, owner: str, ts_ms: int) -> bool:
        with self._lock:
            cur = self._conn.execute(
                "UPDATE capability_tokens SET revoked_ms=? "
                "WHERE id=? AND owner=? AND revoked_ms IS NULL",
                (ts_ms, token_id, owner),
            )
            self._conn.commit()
            return cur.rowcount > 0

    def add_capability_token_spend(self, token_id: str, amount: float) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE capability_tokens SET spent=spent+? "
                "WHERE id=? AND revoked_ms IS NULL", (max(0.0, amount), token_id),
            )
            self._conn.commit()

    # --- generic doc tables (jobs/deployments/models) -------------------- #
    def _put_doc(self, table: str, id_: str, owner: str, status: str,
                 created_ms: int, json_str: str) -> None:
        with self._lock:
            self._conn.execute(
                f"INSERT INTO {table}(id,owner,status,created_ms,data) VALUES(?,?,?,?,?) "
                f"ON CONFLICT(id) DO UPDATE SET status=excluded.status,data=excluded.data",
                (id_, owner, status, created_ms, json_str),
            )
            self._conn.commit()

    def put_job(self, job: BatchJob) -> None:
        self._put_doc("jobs", job.id, job.owner, job.status, job.created_ms,
                      job.model_dump_json())

    def get_job(self, job_id: str) -> Optional[BatchJob]:
        with self._lock:
            row = self._conn.execute("SELECT data FROM jobs WHERE id=?", (job_id,)).fetchone()
        return BatchJob.model_validate_json(row["data"]) if row else None

    def list_jobs(self) -> list[BatchJob]:
        with self._lock:
            rows = self._conn.execute("SELECT data FROM jobs ORDER BY created_ms DESC").fetchall()
        return [BatchJob.model_validate_json(r["data"]) for r in rows]

    def put_deployment(self, dep: Deployment) -> None:
        self._put_doc("deployments", dep.id, dep.owner, dep.status, dep.created_ms,
                      dep.model_dump_json())

    def get_deployment(self, dep_id: str) -> Optional[Deployment]:
        with self._lock:
            row = self._conn.execute("SELECT data FROM deployments WHERE id=?", (dep_id,)).fetchone()
        return Deployment.model_validate_json(row["data"]) if row else None

    def list_deployments(self) -> list[Deployment]:
        with self._lock:
            rows = self._conn.execute("SELECT data FROM deployments ORDER BY created_ms DESC").fetchall()
        return [Deployment.model_validate_json(r["data"]) for r in rows]

    def delete_deployment(self, dep_id: str) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM deployments WHERE id=?", (dep_id,))
            self._conn.commit()

    def put_model(self, model: ModelDeployment) -> None:
        self._put_doc("models", model.id, model.owner, model.status, model.created_ms,
                      model.model_dump_json())

    def get_model(self, model_id: str) -> Optional[ModelDeployment]:
        with self._lock:
            row = self._conn.execute("SELECT data FROM models WHERE id=?", (model_id,)).fetchone()
        return ModelDeployment.model_validate_json(row["data"]) if row else None

    def list_models(self) -> list[ModelDeployment]:
        with self._lock:
            rows = self._conn.execute("SELECT data FROM models ORDER BY created_ms DESC").fetchall()
        return [ModelDeployment.model_validate_json(r["data"]) for r in rows]

    def delete_model(self, model_id: str) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM models WHERE id=?", (model_id,))
            self._conn.commit()

    # --- pooled object store (blobs) ------------------------------------- #
    def put_blob(self, blob: BlobMeta) -> None:
        self._put_doc("blobs", blob.id, blob.owner, blob.status, blob.created_ms,
                      blob.model_dump_json())

    def get_blob(self, blob_id: str) -> Optional[BlobMeta]:
        with self._lock:
            row = self._conn.execute("SELECT data FROM blobs WHERE id=?", (blob_id,)).fetchone()
        return BlobMeta.model_validate_json(row["data"]) if row else None

    def list_blobs(self) -> list[BlobMeta]:
        with self._lock:
            rows = self._conn.execute("SELECT data FROM blobs ORDER BY created_ms DESC").fetchall()
        return [BlobMeta.model_validate_json(r["data"]) for r in rows]

    def delete_blob(self, blob_id: str) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM blobs WHERE id=?", (blob_id,))
            self._conn.commit()

    # --- ledger ---------------------------------------------------------- #
    def _insert_ledger_entry(self, entry: LedgerEntry) -> None:
        """INSERT OR IGNORE one ledger entry (caller holds _lock, no commit)."""
        self._conn.execute(
            "INSERT OR IGNORE INTO ledger(id,node_id,owner,kind,compute_units,ts_ms,detail) "
            "VALUES(?,?,?,?,?,?,?)",
            (entry.id, entry.node_id, entry.owner, entry.kind,
             entry.compute_units, entry.ts_ms, entry.detail),
        )

    def add_ledger(self, entry: LedgerEntry) -> None:
        with self._lock:
            self._insert_ledger_entry(entry)
            self._conn.commit()

    def add_ledger_idempotent(self, entry: LedgerEntry) -> bool:
        """INSERT OR IGNORE — returns True if the entry was newly inserted.

        Callers that derive the entry's ``id`` deterministically (e.g. from
        unit_id + kind) can call this on retries without double-charging.
        """
        with self._lock:
            cur = self._conn.execute(
                "INSERT OR IGNORE INTO ledger(id,node_id,owner,kind,compute_units,ts_ms,detail) "
                "VALUES(?,?,?,?,?,?,?)",
                (entry.id, entry.node_id, entry.owner, entry.kind,
                 entry.compute_units, entry.ts_ms, entry.detail),
            )
            self._conn.commit()
            return cur.rowcount > 0

    def add_ledger_pair(self, donation: LedgerEntry, consumption: LedgerEntry) -> None:
        """Write a donation *and* a consumption entry in a single transaction.

        Use this instead of two separate ``add_ledger()`` calls so a crash
        between them cannot leave the ledger half-charged.  Both entries use
        INSERT OR IGNORE, so a retry with the same ids is a safe no-op.
        """
        with self._lock:
            self._insert_ledger_entry(donation)
            self._insert_ledger_entry(consumption)
            self._conn.commit()

    def prune_ledger(self, keep_count: int = 50_000) -> int:
        """Delete the oldest ledger rows beyond *keep_count*, returning deleted count.

        Bounded-growth guarantee: call periodically (e.g. nightly) from a
        background task.  Audit / accounting queries must be satisfied by an
        external aggregation store for very long-running deployments.
        """
        with self._lock:
            row = self._conn.execute("SELECT COUNT(*) FROM ledger").fetchone()
            total = row[0]
            if total <= keep_count:
                return 0
            excess = total - keep_count
            self._conn.execute(
                "DELETE FROM ledger WHERE id IN "
                "(SELECT id FROM ledger ORDER BY ts_ms ASC LIMIT ?)", (excess,)
            )
            self._conn.commit()
            return excess

    def ledger_by_node(self) -> dict[str, dict[str, float]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT node_id, kind, SUM(compute_units) AS units "
                "FROM ledger GROUP BY node_id, kind"
            ).fetchall()
        out: dict[str, dict[str, float]] = {}
        for r in rows:
            out.setdefault(r["node_id"], {"donated": 0.0, "consumed": 0.0})
            out[r["node_id"]][r["kind"]] = float(r["units"])
        return out

    def credits_by_owner(self) -> dict[str, float]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT owner, kind, SUM(compute_units) AS units FROM ledger GROUP BY owner, kind"
            ).fetchall()
        acc: dict[str, float] = {}
        for r in rows:
            # donated compute and starter grants add credits; consumed compute spends
            # them. Net balance = (donated + grant) - consumed.
            sign = 1.0 if r["kind"] in ("donated", "grant") else -1.0
            acc[r["owner"]] = acc.get(r["owner"], 0.0) + sign * float(r["units"])
        return acc

    def ledger_totals_by_owner(self) -> dict[str, dict[str, float]]:
        """Per-owner gross totals split by kind (donated / consumed / grant)."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT owner, kind, SUM(compute_units) AS units FROM ledger GROUP BY owner, kind"
            ).fetchall()
        out: dict[str, dict[str, float]] = {}
        for r in rows:
            out.setdefault(r["owner"], {"donated": 0.0, "consumed": 0.0, "grant": 0.0})
            out[r["owner"]][r["kind"]] = float(r["units"])
        return out

    def recent_ledger(self, limit: int = 50) -> list[LedgerEntry]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT id,node_id,owner,kind,compute_units,ts_ms,detail "
                "FROM ledger ORDER BY ts_ms DESC LIMIT ?", (limit,)
            ).fetchall()
        return [LedgerEntry(id=r["id"], node_id=r["node_id"], owner=r["owner"],
                            kind=r["kind"], compute_units=r["compute_units"],
                            ts_ms=r["ts_ms"], detail=r["detail"]) for r in rows]

    # --- trust ----------------------------------------------------------- #
    def put_trust(self, score: TrustScore) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO trust_scores(node_id,data) VALUES(?,?) "
                "ON CONFLICT(node_id) DO UPDATE SET data=excluded.data",
                (score.node_id, score.model_dump_json()),
            )
            self._conn.commit()

    def get_trust(self, node_id: str) -> Optional[TrustScore]:
        with self._lock:
            row = self._conn.execute(
                "SELECT data FROM trust_scores WHERE node_id=?", (node_id,)
            ).fetchone()
        return TrustScore.model_validate_json(row["data"]) if row else None

    # --- accounts + device links + starter grants (Tier 2) --------------- #
    def put_account(self, account_id: str, handle: str, provider: str,
                    created_ms: int) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO accounts(id,handle,provider,created_ms) VALUES(?,?,?,?) "
                "ON CONFLICT(id) DO UPDATE SET handle=excluded.handle,"
                "provider=excluded.provider",
                (account_id, handle, provider, created_ms),
            )
            self._conn.commit()

    def get_account(self, account_id: str) -> Optional[dict]:
        row = self._conn.execute(
            "SELECT id,handle,provider,created_ms FROM accounts WHERE id=?",
            (account_id,)).fetchone()
        return dict(row) if row else None

    def link_device(self, device_id: str, account_id: str, linked_ms: int) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO device_links(device_id,account_id,linked_ms) VALUES(?,?,?) "
                "ON CONFLICT(device_id) DO UPDATE SET account_id=excluded.account_id,"
                "linked_ms=excluded.linked_ms",
                (device_id, account_id, linked_ms),
            )
            self._conn.commit()

    def account_for_device(self, device_id: str) -> Optional[str]:
        row = self._conn.execute(
            "SELECT account_id FROM device_links WHERE device_id=?", (device_id,)
        ).fetchone()
        return row["account_id"] if row else None

    def devices_for_account(self, account_id: str) -> list[str]:
        rows = self._conn.execute(
            "SELECT device_id FROM device_links WHERE account_id=?", (account_id,)
        ).fetchall()
        return [r["device_id"] for r in rows]

    def has_grant(self, subject: str) -> bool:
        with self._lock:
            row = self._conn.execute(
                "SELECT 1 FROM starter_grants WHERE subject=?", (subject,)).fetchone()
        return row is not None

    def mark_grant(self, subject: str, ts_ms: int) -> bool:
        """Record that `subject` received its one-time starter grant. Returns True
        if this call created the marker (i.e. no prior grant), False if it already
        existed — used to make the grant atomic and exactly-once per subject."""
        with self._lock:
            cur = self._conn.execute(
                "INSERT OR IGNORE INTO starter_grants(subject,ts_ms) VALUES(?,?)",
                (subject, ts_ms))
            self._conn.commit()
            return cur.rowcount > 0

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # --- retention / compaction ------------------------------------------ #
    def prune_completed_jobs(self, keep_count: int = 10_000) -> int:
        """Delete the oldest completed/failed jobs beyond *keep_count*.

        Only jobs in a terminal state are pruned; running/pending jobs are
        always kept.  Returns the number of rows deleted.
        """
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*) FROM jobs WHERE status NOT IN ('running','pending')"
            ).fetchone()
            total = row[0]
            if total <= keep_count:
                return 0
            excess = total - keep_count
            self._conn.execute(
                "DELETE FROM jobs WHERE id IN ("
                "  SELECT id FROM jobs WHERE status NOT IN ('running','pending') "
                "  ORDER BY created_ms ASC LIMIT ?)", (excess,)
            )
            self._conn.commit()
            return excess

    def recover_inflight_units(self) -> list[str]:
        """Requeue work units left in ASSIGNED/RUNNING state after a crash.

        On restart, in-flight units that the batch manager never completed must
        be moved back to PENDING so they are re-dispatched.  Returns a list of
        job ids that were touched.
        """
        from .models import UnitStatus
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, data FROM jobs WHERE status IN ('running','pending')"
            ).fetchall()
        changed = []
        for row in rows:
            job = BatchJob.model_validate_json(row["data"])
            dirty = False
            for u in job.units:
                if u.status in (UnitStatus.ASSIGNED, UnitStatus.RUNNING):
                    u.status = UnitStatus.PENDING
                    u.assigned_node = None
                    dirty = True
            if dirty:
                self._put_doc("jobs", job.id, job.owner, job.status,
                              job.created_ms, job.model_dump_json())
                changed.append(job.id)
        return changed
