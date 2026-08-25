"""Client-side encryption helpers for Mesh Store objects.

The control plane and storage nodes must see only ciphertext.  This module is
therefore intentionally usable by clients (CLI/tests/gates) before bytes are POSTed
to `/api/v1/blobs`; the storage manager stores the returned package as opaque bytes.

Key hierarchy:
  user root key (local only) -> wraps per-bucket key -> wraps per-object data key.

All encryption and key wrapping uses AES-256-GCM from ``cryptography``.  The random
data key and random nonce mean identical plaintext uploads intentionally do NOT
dedupe at the ciphertext layer.  That trades deduplication for confidentiality:
convergent encryption would preserve dedup but would leak equality of objects to the
control plane and node operators, which is the wrong default for strangers' disks.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import secrets
from pathlib import Path
from typing import Any

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

MAGIC = b"MPENC1\0"
ALG = "AES-256-GCM"
KEY_LEN = 32
NONCE_LEN = 12


class EncryptionError(Exception):
    """Raised when an encrypted object package is malformed or fails authentication."""


def _b64e(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("ascii")


def _b64d(text: str) -> bytes:
    return base64.urlsafe_b64decode(text.encode("ascii"))


def _default_key_path() -> Path:
    env = os.environ.get("MESHPOOL_CLIENT_KEY_PATH")
    if env:
        return Path(env)
    data_dir = os.environ.get("MESHPOOL_DATA_DIR")
    base = Path(data_dir) if data_dir else Path.home() / ".meshpool"
    return base / "client-root-key.json"


class ClientKeyring:
    """Local keyring containing the user root key and wrapped bucket keys."""

    def __init__(self, path: str | os.PathLike | None = None) -> None:
        self.path = Path(path) if path is not None else _default_key_path()
        self.root_key: bytes
        self.buckets: dict[str, dict[str, str]]
        self._load_or_create()

    def _load_or_create(self) -> None:
        if self.path.is_file():
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            self.root_key = _b64d(raw["root_key"])
            self.buckets = dict(raw.get("buckets") or {})
            if len(self.root_key) != KEY_LEN:
                raise EncryptionError("stored root key has the wrong length")
            return
        self.root_key = secrets.token_bytes(KEY_LEN)
        self.buckets = {}
        self._save()

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        body = {
            "version": 1,
            "root_key": _b64e(self.root_key),
            "buckets": self.buckets,
        }
        tmp = self.path.with_name(f"{self.path.name}.tmp-{os.getpid()}")
        tmp.write_text(json.dumps(body, sort_keys=True), encoding="utf-8")
        os.replace(tmp, self.path)
        try:
            os.chmod(self.path, 0o600)
        except OSError:
            pass

    @property
    def root_key_id(self) -> str:
        return hashlib.sha256(self.root_key).hexdigest()[:16]

    def bucket_key(self, bucket: str) -> bytes:
        rec = self.buckets.get(bucket)
        if rec:
            nonce = _b64d(rec["nonce"])
            wrapped = _b64d(rec["wrapped_key"])
            return AESGCM(self.root_key).decrypt(nonce, wrapped, _bucket_aad(bucket))
        key = secrets.token_bytes(KEY_LEN)
        nonce = secrets.token_bytes(NONCE_LEN)
        wrapped = AESGCM(self.root_key).encrypt(nonce, key, _bucket_aad(bucket))
        self.buckets[bucket] = {"alg": ALG, "nonce": _b64e(nonce), "wrapped_key": _b64e(wrapped)}
        self._save()
        return key

    def wrapped_bucket_record(self, bucket: str) -> dict[str, str]:
        self.bucket_key(bucket)
        return dict(self.buckets[bucket])


def _bucket_aad(bucket: str) -> bytes:
    return f"meshpool.bucket.v1:{bucket}".encode("utf-8")


def _object_aad(bucket: str, key: str, version_id: str) -> bytes:
    return f"meshpool.object.v1:{bucket}:{key}:{version_id}".encode("utf-8")


def is_encrypted_package(data: bytes) -> bool:
    return data.startswith(MAGIC)


def encrypt_object(plaintext: bytes, *, bucket: str, key: str,
                   keyring: ClientKeyring | None = None,
                   version_id: str | None = None) -> tuple[bytes, str]:
    """Return a self-describing encrypted package and its object version id."""
    kr = keyring or ClientKeyring()
    version = version_id or secrets.token_hex(16)
    bucket_key = kr.bucket_key(bucket)
    data_key = secrets.token_bytes(KEY_LEN)
    wrap_nonce = secrets.token_bytes(NONCE_LEN)
    data_nonce = secrets.token_bytes(NONCE_LEN)
    aad = _object_aad(bucket, key, version)
    wrapped_data_key = AESGCM(bucket_key).encrypt(wrap_nonce, data_key, aad)
    ciphertext = AESGCM(data_key).encrypt(data_nonce, plaintext, aad)
    header: dict[str, Any] = {
        "v": 1,
        "alg": ALG,
        "bucket": bucket,
        "key": key,
        "version_id": version,
        "root_key_id": kr.root_key_id,
        "bucket_key": kr.wrapped_bucket_record(bucket),
        "data_key_nonce": _b64e(wrap_nonce),
        "wrapped_data_key": _b64e(wrapped_data_key),
        "data_nonce": _b64e(data_nonce),
    }
    header_bytes = json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return MAGIC + len(header_bytes).to_bytes(4, "big") + header_bytes + ciphertext, version


def decrypt_object(package: bytes, *, keyring: ClientKeyring | None = None) -> bytes:
    """Decrypt a package produced by :func:`encrypt_object`.

    Raises ``EncryptionError`` on malformed input or AEAD authentication failure.
    """
    if not is_encrypted_package(package):
        return package
    kr = keyring or ClientKeyring()
    if len(package) < len(MAGIC) + 4:
        raise EncryptionError("encrypted package is truncated")
    header_len = int.from_bytes(package[len(MAGIC):len(MAGIC) + 4], "big")
    start = len(MAGIC) + 4
    end = start + header_len
    if header_len <= 0 or end > len(package):
        raise EncryptionError("encrypted package header length is invalid")
    try:
        header = json.loads(package[start:end].decode("utf-8"))
        if header.get("v") != 1 or header.get("alg") != ALG:
            raise EncryptionError("unsupported encrypted package")
        bucket = str(header["bucket"])
        key = str(header["key"])
        version = str(header["version_id"])
        brec = header["bucket_key"]
        bucket_key = AESGCM(kr.root_key).decrypt(
            _b64d(brec["nonce"]), _b64d(brec["wrapped_key"]), _bucket_aad(bucket))
        aad = _object_aad(bucket, key, version)
        data_key = AESGCM(bucket_key).decrypt(
            _b64d(header["data_key_nonce"]), _b64d(header["wrapped_data_key"]), aad)
        return AESGCM(data_key).decrypt(_b64d(header["data_nonce"]), package[end:], aad)
    except (KeyError, TypeError, ValueError, InvalidTag) as exc:
        raise EncryptionError("encrypted object authentication failed") from exc

