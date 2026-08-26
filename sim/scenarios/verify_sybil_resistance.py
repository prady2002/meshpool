"""Acceptance gate: Sybil resistance uses device-bound hardware timing.

Scope: identity primitive only.  This proves registration requires both Ed25519
key possession and a serial timing challenge.  It also states the honest limit:
the timing toll raises the cost of mass fake identities, but it cannot prove a
unique human or make Sybils impossible without KYC.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from meshpool.common.identity import (  # noqa: E402
    DeviceIdentity,
    hardware_timing_challenge,
    solve_hardware_timing_challenge,
    verify_hardware_timing_response,
)

DATA = Path(__file__).resolve().parent / "_data_sybil"


def main() -> int:
    ok = True
    try:
        shutil.rmtree(DATA, ignore_errors=True)
        DATA.mkdir(parents=True, exist_ok=True)
        now = 1_000_000

        print("== test 1: honest device solves a timing challenge bound to its Ed25519 key ==")
        honest = DeviceIdentity.load_or_create(str(DATA / "honest"))
        challenge = hardware_timing_challenge(
            honest.device_id, now, nonce="gate", iterations=1500, memory_kib=1)
        response = solve_hardware_timing_challenge(honest, challenge)
        assert verify_hardware_timing_response(
            honest.public_key_b64, challenge, response, now_ms_=now,
            min_iterations=1000), "honest device was rejected"
        print(f"  PASS: honest device accepted after {response.elapsed_ms} ms timing toll")

        print("\n== test 2: NON-VACUITY a different key cannot reuse the response ==")
        other = DeviceIdentity.load_or_create(str(DATA / "other"))
        assert not verify_hardware_timing_response(
            other.public_key_b64, challenge, response, now_ms_=now,
            min_iterations=1000), "response was not bound to the original device key"
        print("  PASS: response is bound to the device public key")

        print("\n== test 3: stale replay and too-cheap work are rejected ==")
        cheap = hardware_timing_challenge(
            honest.device_id, now, nonce="cheap", iterations=10, memory_kib=1)
        cheap_response = solve_hardware_timing_challenge(honest, cheap)
        assert not verify_hardware_timing_response(
            honest.public_key_b64, cheap, cheap_response, now_ms_=now,
            min_iterations=1000), "cheap challenge was accepted"
        assert not verify_hardware_timing_response(
            honest.public_key_b64, challenge, response, now_ms_=now + 600_000,
            min_iterations=1000), "stale replay was accepted"
        print("  PASS: minimum-work and freshness checks reject weak registrations")

        print("\n== test 4: NON-VACUITY detector can fail; timing is not KYC ==")
        assert verify_hardware_timing_response(
            honest.public_key_b64, cheap, cheap_response, now_ms_=now,
            min_iterations=1), "loose policy should accept the cheap solved challenge"
        print("  PASS: loose policy accepts a cheap identity; this raises Sybil cost only")

        print("\nALL SYBIL RESISTANCE CHECKS PASSED")
    except Exception as exc:  # noqa: BLE001
        ok = False
        print(f"\nFAILED: {type(exc).__name__}: {exc}")
        import traceback
        traceback.print_exc()
    finally:
        shutil.rmtree(DATA, ignore_errors=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
