"""Locked-down curated-workload sandbox (spec §5, public Tier-2 mesh).

These prove the enforcement a donor relies on when running a *stranger's* unit:
correct results still come back, but an over-long, over-large, or abusive unit is
denied rather than allowed to monopolise the machine.
"""
from __future__ import annotations

import socket
import time

import pytest

from meshpool.agent.sandbox import (
    SELFTEST_ALLOC, SELFTEST_CONNECT, SandboxError, SandboxLimits, run_sandboxed,
)
from meshpool.batch.functions import run_function, validate_payload


def test_curated_unit_runs_and_returns_correct_result():
    assert run_sandboxed("square", {"x": 9}) == 81
    assert run_sandboxed("is_prime", {"n": 97}) == {"n": 97, "prime": True}
    # identical to the un-sandboxed computation (the sandbox is transparent on success)
    assert run_sandboxed("embed", {"text": "meshpool", "dim": 8}) == \
        run_function("embed", {"text": "meshpool", "dim": 8})


def test_wall_clock_timeout_is_enforced():
    t0 = time.time()
    with pytest.raises(SandboxError, match="timeout"):
        run_sandboxed("sleep_echo", {"seconds": 5}, SandboxLimits(wall_clock_s=1.0))
    assert time.time() - t0 < 4.5, "sandbox did not kill the overrunning unit promptly"


def test_memory_cap_is_enforced():
    # Allocating far past the cap must be denied (Job Object on Windows / RLIMIT_AS
    # on POSIX), while a small allocation under the cap succeeds.
    with pytest.raises(SandboxError):
        run_sandboxed(SELFTEST_ALLOC, {"mb": 1024},
                      SandboxLimits(memory_mb=128, wall_clock_s=20), allow_selftest=True)
    ok = run_sandboxed(SELFTEST_ALLOC, {"mb": 8}, SandboxLimits(memory_mb=256),
                       allow_selftest=True)
    assert ok["allocated_mb"] == 8


def test_abusive_input_is_rejected_before_running():
    with pytest.raises(SandboxError, match="exceeds the maximum"):
        run_sandboxed("sha_pow", {"rounds": 10 ** 12})
    with pytest.raises(SandboxError, match="exceeds the maximum"):
        run_sandboxed("sleep_echo", {"seconds": 10 ** 9})


def test_unknown_function_is_rejected():
    with pytest.raises(SandboxError):
        run_sandboxed("rm_rf_slash", {})


def test_output_size_cap():
    # A 1-byte output cap makes even a tiny legitimate result too big -> denied,
    # proving the donor is protected from a unit returning a huge blob.
    with pytest.raises(SandboxError, match="output cap"):
        run_sandboxed("square", {"x": 3}, SandboxLimits(max_output_bytes=1))


def test_sandbox_egress_positive_control_reaches_allowed_destination():
    result = run_sandboxed(SELFTEST_CONNECT, {
        "host": "93.184.216.34",
        "port": 443,
        "whitelist": [["93.184.216.34", 443]],
        "fake_original_connect": True,
    }, allow_selftest=True)
    assert result["attempted"] is True
    assert result["connected"] is True
    assert result["original_connect_calls"] == 1


def test_sandbox_egress_blocks_private_destination_before_connecting():
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    port = listener.getsockname()[1]
    try:
        result = run_sandboxed(SELFTEST_CONNECT, {
            "host": "127.0.0.1",
            "port": port,
            "allow_all": True,
        }, allow_selftest=True)
        assert result["attempted"] is True
        assert result["connected"] is False
        assert result["denied"] is True
    finally:
        listener.close()


def test_sandbox_egress_blocks_hard_port_before_connecting():
    result = run_sandboxed(SELFTEST_CONNECT, {
        "host": "93.184.216.34",
        "port": 25,
        "allow_all": True,
        "fake_original_connect": True,
    }, allow_selftest=True)
    assert result["attempted"] is True
    assert result["connected"] is False
    assert result["denied"] is True
    assert result["original_connect_calls"] == 0


def test_selftest_functions_are_unreachable_from_a_submitted_unit():
    """A work unit must not be able to name a selftest function.

    `func` and `payload` arrive from the unit and are passed straight into
    run_sandboxed() by the runtime, while the selftest branches deliberately skip
    validate_payload() and can steer the child's own egress policy. If a unit
    could name them, `allow_all` would switch off the node's egress filter -- the
    one thing standing between a donated PC and a botnet. The switch that unlocks
    them is therefore a Python keyword argument set at the call site, which a
    submitted unit has no way to spell.
    """
    for name in (SELFTEST_ALLOC, SELFTEST_CONNECT):
        with pytest.raises(SandboxError, match="unknown function"):
            run_sandboxed(name, {"mb": 1, "host": "93.184.216.34",
                                 "port": 443, "allow_all": True})


def test_validate_payload_allows_legitimate_jobs():
    # The values used across the existing suite/scenarios must pass untouched.
    for name, payload in [
        ("square", {"x": 11}), ("sha_pow", {"seed": "s", "rounds": 60000}),
        ("sleep_echo", {"seconds": 1.5}), ("embed", {"text": "a b c", "dim": 16}),
        ("is_prime", {"n": 7}),
    ]:
        validate_payload(name, payload)  # must not raise


def test_validate_payload_rejects_abuse():
    with pytest.raises(ValueError):
        validate_payload("sha_pow", {"rounds": 99_999_999_999})
    with pytest.raises(ValueError):
        validate_payload("embed", {"text": "x" * 300_000})
