from __future__ import annotations

from meshpool.common.identity import (
    DeviceIdentity,
    hardware_timing_challenge,
    solve_hardware_timing_challenge,
    verify_hardware_timing_response,
)


def test_hardware_timing_challenge_is_bound_to_device_identity(tmp_path):
    honest = DeviceIdentity.load_or_create(str(tmp_path / "honest"))
    other = DeviceIdentity.load_or_create(str(tmp_path / "other"))
    now = 1_000_000
    challenge = hardware_timing_challenge(
        honest.device_id, now, nonce="n", iterations=1200, memory_kib=1)
    response = solve_hardware_timing_challenge(honest, challenge)

    assert verify_hardware_timing_response(
        honest.public_key_b64, challenge, response, now_ms_=now,
        min_iterations=1000)
    assert not verify_hardware_timing_response(
        other.public_key_b64, challenge, response, now_ms_=now,
        min_iterations=1000)


def test_hardware_timing_challenge_rejects_too_cheap_or_stale(tmp_path):
    ident = DeviceIdentity.load_or_create(str(tmp_path / "dev"))
    now = 1_000_000
    cheap = hardware_timing_challenge(
        ident.device_id, now, nonce="cheap", iterations=10, memory_kib=1)
    cheap_response = solve_hardware_timing_challenge(ident, cheap)

    assert not verify_hardware_timing_response(
        ident.public_key_b64, cheap, cheap_response, now_ms_=now,
        min_iterations=1000)
    assert not verify_hardware_timing_response(
        ident.public_key_b64, cheap, cheap_response,
        now_ms_=now + 600_000, min_iterations=1)


def test_sybil_resistance_is_a_cost_raise_not_an_impossibility(tmp_path):
    ident = DeviceIdentity.load_or_create(str(tmp_path / "bot"))
    now = 1_000_000
    challenge = hardware_timing_challenge(
        ident.device_id, now, nonce="loose", iterations=10, memory_kib=1)
    response = solve_hardware_timing_challenge(ident, challenge)

    assert verify_hardware_timing_response(
        ident.public_key_b64, challenge, response, now_ms_=now,
        min_iterations=1)
