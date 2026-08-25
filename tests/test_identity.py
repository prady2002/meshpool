"""Unit tests for device identity (Ed25519 auto-identity, zero-signup)."""
from __future__ import annotations

from meshpool.common.identity import (
    DeviceIdentity, challenge_is_fresh, device_id_for_public_key,
    enrollment_challenge, verify_signature,
)


def test_identity_is_stable_across_reload(tmp_path):
    d = str(tmp_path)
    a = DeviceIdentity.load_or_create(d)
    b = DeviceIdentity.load_or_create(d)          # second run reloads, doesn't regen
    assert a.device_id == b.device_id
    assert a.public_key_b64 == b.public_key_b64
    assert a.device_id.startswith("dev_") and len(a.device_id) == 28


def test_device_id_derives_deterministically_from_pubkey(tmp_path):
    idn = DeviceIdentity.load_or_create(str(tmp_path))
    assert device_id_for_public_key(idn.public_key_b64) == idn.device_id


def test_distinct_data_dirs_yield_distinct_devices(tmp_path):
    a = DeviceIdentity.load_or_create(str(tmp_path / "a"))
    b = DeviceIdentity.load_or_create(str(tmp_path / "b"))
    assert a.device_id != b.device_id             # co-located sim nodes stay distinct


def test_signature_roundtrip_and_tamper_detection(tmp_path):
    idn = DeviceIdentity.load_or_create(str(tmp_path))
    challenge = enrollment_challenge(idn.device_id, 1234567890)
    sig = idn.sign(challenge)
    assert verify_signature(idn.public_key_b64, challenge, sig)
    assert not verify_signature(idn.public_key_b64, challenge + b"x", sig)
    assert not verify_signature(idn.public_key_b64, challenge, sig[:-2] + "AA")


def test_other_key_cannot_forge_signature(tmp_path):
    a = DeviceIdentity.load_or_create(str(tmp_path / "a"))
    b = DeviceIdentity.load_or_create(str(tmp_path / "b"))
    challenge = enrollment_challenge(a.device_id, 42)
    sig_by_b = b.sign(challenge)
    # b's signature must not validate against a's public key
    assert not verify_signature(a.public_key_b64, challenge, sig_by_b)


def test_corrupt_identity_file_regenerates(tmp_path):
    p = tmp_path / "identity.json"
    p.write_text("not json", encoding="utf-8")
    idn = DeviceIdentity.load_or_create(str(tmp_path))   # must not raise
    assert idn.device_id.startswith("dev_")


def test_challenge_freshness_window_bounds_replay():
    now = 1_000_000_000_000
    assert challenge_is_fresh(now, now)                       # exact match
    assert challenge_is_fresh(now - 250_000, now)             # 250s stale: inside ±5min
    assert challenge_is_fresh(now + 250_000, now)             # 250s ahead: inside window
    assert not challenge_is_fresh(now - 3_600_000, now)       # 1h stale: replay rejected
    assert not challenge_is_fresh(now + 3_600_000, now)       # 1h future: rejected
    assert not challenge_is_fresh("bogus", now)               # malformed ts: rejected
