"""Disk-backed, content-addressed blob store used by the node agent.

Each blob is keyed by the SHA-256 hex digest of its bytes and written under a
two-char shard directory (``<root>/<ab>/<abc...>``). Content-addressing buys three
properties for free: identical uploads dedupe, a corrupted or adversarial copy is
caught on read (the digest won't match), and a write is idempotent (re-storing the
same id is a no-op). A byte cap bounds how much disk a node donates so a large
upload can't fill a donor's disk.
"""
from __future__ import annotations

import hashlib
import os
import threading

# Per-leaf ceiling: the largest a single content-addressed blob (a plain small object
# or one chunk of a large object) may be. Kept well under a typical node's donated-disk
# cap so a single leaf always fits on the donors chosen to hold it.
MAX_BLOB_BYTES = int(os.environ.get("MESHPOOL_MAX_BLOB_BYTES", "50000000"))

# Per-object ceiling enforced at the control-plane boundary. Objects above the chunk
# size are split into MAX_BLOB_BYTES-bounded chunks and spread across the pool, so this
# ceiling can far exceed any single machine's free disk (that is the whole point of
# pooled storage). The upload body is buffered in memory, so this also bounds peak RAM.
MAX_OBJECT_BYTES = int(os.environ.get("MESHPOOL_MAX_OBJECT_BYTES", "524288000"))  # 500 MB


class StorageFull(Exception):
    """Raised when accepting a blob would exceed the node's donated-disk cap."""


class BlobStore:
    """Thread-safe, content-addressed file store rooted at ``root`` with a byte cap."""

    def __init__(self, root: str, cap_bytes: int) -> None:
        self.root = root
        self.cap_bytes = max(0, int(cap_bytes))
        self._lock = threading.Lock()
        os.makedirs(root, exist_ok=True)
        self._used = self._scan_used()

    def _scan_used(self) -> int:
        total = 0
        for dirpath, _dirs, files in os.walk(self.root):
            for name in files:
                try:
                    total += os.path.getsize(os.path.join(dirpath, name))
                except OSError:
                    pass
        return total

    def _path(self, blob_id: str) -> str:
        return os.path.join(self.root, blob_id[:2], blob_id)

    def has(self, blob_id: str) -> bool:
        return os.path.isfile(self._path(blob_id))

    def used_bytes(self) -> int:
        return self._used

    def free_bytes(self) -> int:
        return max(0, self.cap_bytes - self._used)

    def put(self, blob_id: str, data: bytes) -> None:
        """Store ``data`` under ``blob_id``. Verifies the digest (raising ValueError on
        mismatch), is idempotent, and raises StorageFull if the cap would be exceeded."""
        digest = hashlib.sha256(data).hexdigest()
        if digest != blob_id:
            raise ValueError(f"content hash mismatch: expected {blob_id}, got {digest}")
        with self._lock:
            path = self._path(blob_id)
            if os.path.isfile(path):
                return  # already stored — content-addressed, so identical bytes
            if self._used + len(data) > self.cap_bytes:
                raise StorageFull(
                    f"donated disk cap reached ({self._used}+{len(data)} > {self.cap_bytes})")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            tmp = f"{path}.tmp-{os.getpid()}"
            with open(tmp, "wb") as fh:
                fh.write(data)
            os.replace(tmp, path)   # atomic publish
            self._used += len(data)

    def get(self, blob_id: str) -> "bytes | None":
        try:
            with open(self._path(blob_id), "rb") as fh:
                return fh.read()
        except OSError:
            return None

    def delete(self, blob_id: str) -> None:
        with self._lock:
            path = self._path(blob_id)
            try:
                size = os.path.getsize(path)
            except OSError:
                return
            try:
                os.remove(path)
                self._used = max(0, self._used - size)
            except OSError:
                pass
