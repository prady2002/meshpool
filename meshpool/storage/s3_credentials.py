"""Durable storage for S3-compatible access keys.

Why this is its own module rather than a dict on the ControlPlane:

SigV4 is a *shared-secret* scheme. The server has to recompute the client's
signature, so it must hold the secret key in recoverable form -- there is no
hash-and-compare option the way there is for a password. That makes where and
how these keys live a real design decision rather than an implementation
detail, and it makes an in-memory dict actively dangerous:

  * every access key silently disappears on control-plane restart, so every
    boto3 client, rclone mount and ``aws s3`` script breaks at once with
    ``SignatureDoesNotMatch`` -- an error that points at the client, not at the
    server that actually lost the credential;
  * nothing can ever be revoked, because there is nothing to revoke *from*.

Keys are stored alongside the rest of the control-plane state, which is the
same trust boundary that already holds session tokens and the ledger.
"""
from __future__ import annotations

import secrets
import time
from typing import Optional

from ..common.logging import get_logger

log = get_logger("storage.s3creds")

_DDL = """
CREATE TABLE IF NOT EXISTS s3_credentials (
    access_key  TEXT PRIMARY KEY,
    secret_key  TEXT NOT NULL,
    owner       TEXT NOT NULL,
    created_ms  INTEGER NOT NULL
)
"""


class S3CredentialStore:
    """Access-key -> (secret_key, owner) with durable backing.

    Presents a ``Mapping``-ish surface (``__getitem__``/``get``/``items``/
    ``__contains__``) so it is a drop-in for the plain dict this replaced.
    """

    def __init__(self, store) -> None:
        self.store = store
        self._ensure_table()

    def _ensure_table(self) -> None:
        # Idempotent DDL for the same reason spend_caps uses it: the table is
        # additive and has no foreign keys, so it is safe to introduce into an
        # already-populated database without a numbered migration.
        with self.store._lock:
            self.store._conn.execute(_DDL)
            self.store._conn.commit()

    # --- mapping surface -------------------------------------------------- #
    def get(self, access_key: str, default=None):
        if not access_key:
            return default
        with self.store._lock:
            row = self.store._conn.execute(
                "SELECT secret_key, owner FROM s3_credentials WHERE access_key = ?",
                (access_key,)).fetchone()
        if row is None:
            return default
        return (row["secret_key"], row["owner"])

    def __getitem__(self, access_key: str):
        v = self.get(access_key)
        if v is None:
            raise KeyError(access_key)
        return v

    def __contains__(self, access_key: str) -> bool:
        return self.get(access_key) is not None

    def __setitem__(self, access_key: str, value) -> None:
        secret_key, owner = value
        self.put(access_key, secret_key, owner)

    def items(self):
        with self.store._lock:
            rows = self.store._conn.execute(
                "SELECT access_key, secret_key, owner FROM s3_credentials "
                "ORDER BY created_ms").fetchall()
        return [(r["access_key"], (r["secret_key"], r["owner"])) for r in rows]

    def __len__(self) -> int:
        with self.store._lock:
            row = self.store._conn.execute(
                "SELECT COUNT(*) AS n FROM s3_credentials").fetchone()
        return int(row["n"]) if row else 0

    # --- management ------------------------------------------------------- #
    def put(self, access_key: str, secret_key: str, owner: str) -> None:
        with self.store._lock:
            self.store._conn.execute(
                "INSERT OR REPLACE INTO s3_credentials "
                "(access_key, secret_key, owner, created_ms) VALUES (?, ?, ?, ?)",
                (access_key, secret_key, owner, int(time.time() * 1000)))
            self.store._conn.commit()
        log.info("s3 credential created", access_key=access_key, owner=owner)

    def create(self, owner: str) -> tuple[str, str]:
        """Mint and persist a new access-key pair; returns ``(access, secret)``."""
        access_key = "MESHPOOL" + secrets.token_hex(6).upper()
        secret_key = secrets.token_urlsafe(32)
        self.put(access_key, secret_key, owner)
        return access_key, secret_key

    def revoke(self, access_key: str) -> bool:
        """Delete a key. Returns True if it existed."""
        with self.store._lock:
            cur = self.store._conn.execute(
                "DELETE FROM s3_credentials WHERE access_key = ?", (access_key,))
            self.store._conn.commit()
            gone = cur.rowcount > 0
        if gone:
            log.info("s3 credential revoked", access_key=access_key)
        return gone

    def owner_of(self, access_key: str) -> Optional[str]:
        v = self.get(access_key)
        return v[1] if v else None
