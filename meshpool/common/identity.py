"""Device identity — a self-generated cryptographic identity per machine.

The global network needs a stable, unforgeable way to name a participating device
without any signup step (spec: "auto device identity + optional lightweight
account"). On first run each machine generates an **Ed25519 keypair**, persisted at
``<data_dir>/identity.json``. The public key hashes to a stable ``device_id`` that:

* stays constant across restarts and IP/network changes (so a laptop that moves
  Wi-Fi networks is still the same node), and
* lets the coordinator verify — via a signature — that whoever presents a
  ``device_id`` actually holds the matching private key (proof of possession),
  instead of trusting a caller-supplied id.

Only vetted, standard crypto (``cryptography``'s Ed25519) is used — no hand-rolled
primitives. An optional user account (Phase 2) links one or more ``device_id``s to
an owner; that layer builds on this one and does not replace it.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
from pathlib import Path
from typing import Optional

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey, Ed25519PublicKey,
)


def _b64e(raw: bytes) -> str:
    return base64.b64encode(raw).decode("ascii")


def _b64d(txt: str) -> bytes:
    return base64.b64decode(txt.encode("ascii"))


def device_id_for_public_key(public_key_b64: str) -> str:
    """Derive the stable, human-shareable device id from a public key.

    ``dev_`` + lowercase base32 of the first 15 bytes of SHA-256(pubkey) → a
    24-char id with no padding, safe in URLs, headers and QR codes.
    """
    raw = _b64d(public_key_b64)
    digest = hashlib.sha256(raw).digest()[:15]
    b32 = base64.b32encode(digest).decode("ascii").rstrip("=").lower()
    return f"dev_{b32}"


def verify_signature(public_key_b64: str, data: bytes, signature_b64: str) -> bool:
    """True iff ``signature_b64`` is a valid Ed25519 signature of ``data``."""
    try:
        pub = Ed25519PublicKey.from_public_bytes(_b64d(public_key_b64))
        pub.verify(_b64d(signature_b64), data)
        return True
    except (InvalidSignature, ValueError, Exception):  # noqa: BLE001
        return False


class DeviceIdentity:
    """A machine's persistent Ed25519 identity (generated once, then loaded)."""

    def __init__(self, private_key: Ed25519PrivateKey) -> None:
        self._priv = private_key
        raw_pub = private_key.public_key().public_bytes(
            serialization.Encoding.Raw, serialization.PublicFormat.Raw)
        self.public_key_b64 = _b64e(raw_pub)
        self.device_id = device_id_for_public_key(self.public_key_b64)

    # --- persistence ----------------------------------------------------- #
    @classmethod
    def load_or_create(cls, data_dir: str) -> "DeviceIdentity":
        path = Path(data_dir) / "identity.json"
        if path.exists():
            try:
                return cls._from_file(path)
            except Exception:  # noqa: BLE001 — corrupt file → regenerate
                pass
        ident = cls(Ed25519PrivateKey.generate())
        ident._save(path)
        return ident

    @classmethod
    def _from_file(cls, path: Path) -> "DeviceIdentity":
        data = json.loads(path.read_text(encoding="utf-8"))
        priv = serialization.load_pem_private_key(
            data["private_key_pem"].encode("utf-8"), password=None)
        if not isinstance(priv, Ed25519PrivateKey):
            raise ValueError("not an Ed25519 private key")
        return cls(priv)

    def _save(self, path: Path) -> None:
        pem = self._priv.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption()).decode("utf-8")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "device_id": self.device_id,
            "public_key": self.public_key_b64,
            "private_key_pem": pem,
        }, indent=2), encoding="utf-8")
        try:  # best-effort: restrict to the owner on POSIX
            os.chmod(path, 0o600)
        except OSError:
            pass

    # --- signing --------------------------------------------------------- #
    def sign(self, data: bytes) -> str:
        return _b64e(self._priv.sign(data))

    def sign_text(self, text: str) -> str:
        return self.sign(text.encode("utf-8"))


def enrollment_challenge(device_id: str, ts_ms: int, nonce: str = "") -> bytes:
    """Canonical bytes a device signs at registration to prove key possession."""
    return f"meshpool-enroll:{device_id}:{ts_ms}:{nonce}".encode("utf-8")


# Acceptance window for a signed enrollment/auth challenge. The signed ``ts_ms`` must
# be within this many milliseconds of the coordinator's clock, so a captured signature
# can only be replayed briefly (bounded by clock skew) instead of forever. ±5 min is the
# usual leeway for signed-timestamp schemes (JWT ``iat``/``exp``, Kerberos) and is far
# larger than realistic NTP skew, so it never rejects an honest, roughly-synced device.
ENROLL_FRESHNESS_MS = 300_000


def challenge_is_fresh(ts_ms: int, now_ms: int,
                       window_ms: int = ENROLL_FRESHNESS_MS) -> bool:
    """True iff a signed challenge timestamp is recent enough to accept.

    Rejects both stale (replayed) and implausibly future-dated timestamps. Purely a
    function of the two clocks, so it is trivially unit-testable.
    """
    try:
        return abs(int(now_ms) - int(ts_ms)) <= window_ms
    except (TypeError, ValueError):
        return False
