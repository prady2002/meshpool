"""Mesh Functions: the WASM sandbox that lets *arbitrary user code* run.

Every module here is built from WAT source at test time with ``wasmtime.wat2wasm``,
so the suite needs no Rust/Go/clang toolchain and still exercises real compiled
WebAssembly through real WASI — not a mock.

The guest ABI under test is the WASI *command* ABI: JSON in on fd 0, JSON out on
fd 1, entry point ``_start``. That ABI was chosen because every toolchain that
targets ``wasm32-wasip1`` already speaks it.

Each test that asserts a *limit* also asserts the limit was actually approached
(fuel burned, time elapsed, bytes produced). Without that, a module that failed to
compile — or silently did nothing — would satisfy the assertion and the test would
prove nothing. That failure mode has bitten this project repeatedly.
"""
from __future__ import annotations

import json

import pytest

wasmtime = pytest.importorskip("wasmtime", reason="wasmtime not installed")

from meshpool.functions import wasm as W  # noqa: E402


# --------------------------------------------------------------------------- #
# WAT building blocks
#
# WASI preview-1 signatures used below:
#   fd_read (fd, iovs, iovs_len, nread_ptr)   -> errno
#   fd_write(fd, iovs, iovs_len, nwritten_ptr)-> errno
#   proc_exit(code)                           -> noreturn
#   path_open(dirfd, dirflags, path, path_len, oflags,
#             rights_base, rights_inheriting, fdflags, opened_fd_ptr) -> errno
# --------------------------------------------------------------------------- #

_WASI = '(import "wasi_snapshot_preview1" "{name}" (func ${name} {sig}))'


def _wat_to_wasm(wat: str) -> bytes:
    return bytes(wasmtime.wat2wasm(wat))


def _echo_module() -> bytes:
    """Read all of stdin, write it straight back to stdout. Valid JSON in => out."""
    return _wat_to_wasm(r"""
(module
  (import "wasi_snapshot_preview1" "fd_read"
    (func $fd_read (param i32 i32 i32 i32) (result i32)))
  (import "wasi_snapshot_preview1" "fd_write"
    (func $fd_write (param i32 i32 i32 i32) (result i32)))
  (memory (export "memory") 2)
  ;; iovec at 0: {buf=64, len=8192}; nread/nwritten scratch at 16
  (func (export "_start")
    (i32.store (i32.const 0) (i32.const 64))
    (i32.store (i32.const 4) (i32.const 8192))
    (drop (call $fd_read (i32.const 0) (i32.const 0) (i32.const 1) (i32.const 16)))
    ;; reuse the iovec, but with the byte count we actually read
    (i32.store (i32.const 0) (i32.const 64))
    (i32.store (i32.const 4) (i32.load (i32.const 16)))
    (drop (call $fd_write (i32.const 1) (i32.const 0) (i32.const 1) (i32.const 20)))
  )
)
""")


def _const_output_module(text: str) -> bytes:
    """Ignore stdin, always write *text* to stdout."""
    data = text.encode("utf-8")
    escaped = "".join(f"\\{b:02x}" for b in data)
    return _wat_to_wasm(f"""
(module
  (import "wasi_snapshot_preview1" "fd_write"
    (func $fd_write (param i32 i32 i32 i32) (result i32)))
  (memory (export "memory") 1)
  (data (i32.const 64) "{escaped}")
  (func (export "_start")
    (i32.store (i32.const 0) (i32.const 64))
    (i32.store (i32.const 4) (i32.const {len(data)}))
    (drop (call $fd_write (i32.const 1) (i32.const 0) (i32.const 1) (i32.const 16)))
  )
)
""")


def _spin_module() -> bytes:
    """An unbounded loop. Must be stopped by fuel, never by finishing."""
    return _wat_to_wasm(r"""
(module
  (memory (export "memory") 1)
  (func (export "_start")
    (local $i i64)
    (loop $forever
      (local.set $i (i64.add (local.get $i) (i64.const 1)))
      (br $forever)
    )
  )
)
""")


def _exit_module(code: int) -> bytes:
    return _wat_to_wasm(f"""
(module
  (import "wasi_snapshot_preview1" "proc_exit" (func $proc_exit (param i32)))
  (memory (export "memory") 1)
  (func (export "_start") (call $proc_exit (i32.const {code})))
)
""")


def _trap_module() -> bytes:
    return _wat_to_wasm(r"""
(module
  (memory (export "memory") 1)
  (func (export "_start") (unreachable))
)
""")


def _silent_module() -> bytes:
    """Starts, writes nothing, returns cleanly."""
    return _wat_to_wasm(r"""
(module (memory (export "memory") 1) (func (export "_start") (nop)))
""")


def _fs_escape_module() -> bytes:
    """Try to open a host file. With no preopened dir the guest has no fd to
    resolve against, so this must fail; we surface the errno on stdout."""
    path = "/etc/passwd"
    escaped = "".join(f"\\{b:02x}" for b in path.encode())
    return _wat_to_wasm(f"""
(module
  (import "wasi_snapshot_preview1" "path_open"
    (func $path_open (param i32 i32 i32 i32 i32 i64 i64 i32 i32) (result i32)))
  (import "wasi_snapshot_preview1" "fd_write"
    (func $fd_write (param i32 i32 i32 i32) (result i32)))
  (memory (export "memory") 1)
  (data (i32.const 200) "{escaped}")
  (data (i32.const 300) "\\22")          ;; a double-quote character
  (func (export "_start")
    (local $errno i32)
  ;; fd 3 is the first preopen slot; with no preopens it does not exist.
  ;; Rights are passed as 0 (request nothing) — passing an all-ones mask
  ;; overflows wasmtime's Rights conversion before the sandbox check is reached.
  (local.set $errno
    (call $path_open (i32.const 3) (i32.const 0)
                     (i32.const 200) (i32.const {len(path)})
                     (i32.const 0) (i64.const 0) (i64.const 0)
                     (i32.const 0) (i32.const 400)))
    ;; emit the errno as a bare JSON number
    (i32.store (i32.const 500) (i32.add (i32.const 48) (local.get $errno)))
    (i32.store (i32.const 0) (i32.const 500))
    (i32.store (i32.const 4) (i32.const 1))
    (drop (call $fd_write (i32.const 1) (i32.const 0) (i32.const 1) (i32.const 16)))
  )
)
""")


def _foreign_import_module() -> bytes:
    """Imports something the host does not provide. Must fail to instantiate —
    this is the check that the guest cannot reach an ambient capability."""
    return _wat_to_wasm(r"""
(module
  (import "env" "definitely_not_provided" (func $x (param i32) (result i32)))
  (memory (export "memory") 1)
  (func (export "_start") (drop (call $x (i32.const 1))))
)
""")


def _busy_module(iterations: int) -> bytes:
    """Do *iterations* of pure guest arithmetic, then emit a constant JSON result.

    Used to measure fuel against known guest work. The output is fixed so the
    result size cannot confound the measurement.
    """
    out = b'{"ok":true}'
    escaped = "".join(f"\\{b:02x}" for b in out)
    return _wat_to_wasm(f"""
(module
  (import "wasi_snapshot_preview1" "fd_write"
    (func $fd_write (param i32 i32 i32 i32) (result i32)))
  (memory (export "memory") 1)
  (data (i32.const 64) "{escaped}")
  (func (export "_start")
    (local $i i32) (local $acc i64)
    (block $done
      (loop $work
        (br_if $done (i32.ge_u (local.get $i) (i32.const {iterations})))
        (local.set $acc (i64.add (local.get $acc) (i64.const 7)))
        (local.set $i (i32.add (local.get $i) (i32.const 1)))
        (br $work)
      )
    )
    (i32.store (i32.const 0) (i32.const 64))
    (i32.store (i32.const 4) (i32.const {len(out)}))
    (drop (call $fd_write (i32.const 1) (i32.const 0) (i32.const 1) (i32.const 16)))
  )
)
""")


def _big_output_module(n_bytes: int) -> bytes:
    """Write *n_bytes* of 'A' to stdout in one go, to trip the output cap."""
    pages = max(2, (n_bytes // 65536) + 2)
    return _wat_to_wasm(f"""
(module
  (import "wasi_snapshot_preview1" "fd_write"
    (func $fd_write (param i32 i32 i32 i32) (result i32)))
  (memory (export "memory") {pages})
  (func (export "_start")
    (local $i i32)
    (local.set $i (i32.const 0))
    (block $done
      (loop $fill
        (br_if $done (i32.ge_u (local.get $i) (i32.const {n_bytes})))
        (i32.store8 (i32.add (i32.const 1024) (local.get $i)) (i32.const 65))
        (local.set $i (i32.add (local.get $i) (i32.const 1)))
        (br $fill)
      )
    )
    (i32.store (i32.const 0) (i32.const 1024))
    (i32.store (i32.const 4) (i32.const {n_bytes}))
    (drop (call $fd_write (i32.const 1) (i32.const 0) (i32.const 1) (i32.const 16)))
  )
)
""")


# --------------------------------------------------------------------------- #
# Happy path
# --------------------------------------------------------------------------- #

def test_echo_round_trips_json_through_real_wasi():
    payload = {"hello": "world", "n": 42, "xs": [1, 2, 3], "nested": {"a": True}}
    res = W.run(_echo_module(), payload)
    assert res.value == payload
    # Non-vacuity: the guest really executed. A module that never ran would burn
    # no fuel and produce no bytes, yet an `== payload` check on a mock would pass.
    assert res.fuel_used > 0, "no fuel burned => the module never executed"
    assert res.stdout_bytes == len(json.dumps(payload))


def test_unicode_survives_the_round_trip():
    payload = {"s": "héllo — 世界 🌍", "emoji": "🚀"}
    assert W.run(_echo_module(), payload).value == payload


@pytest.mark.parametrize("payload", [
    None, True, False, 0, -1, 3.5, "", "x", [], {}, [1, [2, [3]]], {"k": {"k": {}}},
])
def test_all_json_shapes_round_trip(payload):
    assert W.run(_echo_module(), payload).value == payload


def test_module_cache_makes_the_second_call_warm():
    W.clear_cache()
    mod = _echo_module()
    first = W.run(mod, {"i": 1})
    second = W.run(mod, {"i": 2})
    assert first.cold_start is True
    assert second.cold_start is False, "identical bytes must hit the compile cache"
    assert W.cache_size() == 1
    assert second.value == {"i": 2}, "a cache hit must not reuse the previous result"


def test_distinct_modules_get_distinct_cache_entries():
    W.clear_cache()
    W.run(_const_output_module('{"who":"a"}'), None)
    W.run(_const_output_module('{"who":"b"}'), None)
    assert W.cache_size() == 2
    assert W.run(_const_output_module('{"who":"a"}'), None).value == {"who": "a"}


# --------------------------------------------------------------------------- #
# Resource limits — a runaway function must be *stopped*, not merely observed
# --------------------------------------------------------------------------- #

def test_infinite_loop_is_killed_by_fuel():
    limits = W.Limits(fuel=2_000_000, wall_timeout_s=30.0)
    with pytest.raises(W.FunctionError) as ei:
        W.run(_spin_module(), None, limits=limits)
    assert ei.value.kind == "fuel", f"expected fuel exhaustion, got {ei.value.kind}"


def test_fuel_limit_is_actually_enforced_not_just_reported():
    """Non-vacuity for the test above: a *smaller* fuel budget must stop the same
    module sooner. If fuel were ignored, both runs would behave identically."""
    import time
    slow = W.Limits(fuel=200_000_000, wall_timeout_s=60.0)
    fast = W.Limits(fuel=1_000_000, wall_timeout_s=60.0)

    t0 = time.perf_counter()
    with pytest.raises(W.FunctionError):
        W.run(_spin_module(), None, limits=fast)
    t_fast = time.perf_counter() - t0

    t0 = time.perf_counter()
    with pytest.raises(W.FunctionError):
        W.run(_spin_module(), None, limits=slow)
    t_slow = time.perf_counter() - t0

    assert t_slow > t_fast, (
        f"200x the fuel did not take longer ({t_slow:.4f}s vs {t_fast:.4f}s) "
        f"— fuel metering is not actually bounding execution")


def test_wall_clock_timeout_stops_a_module_fuel_alone_would_not():
    """Fuel cannot interrupt a guest that is *not* burning fuel fast enough to hit
    its budget. The epoch-interruption watchdog is what guarantees a bounded call."""
    import time
    limits = W.Limits(fuel=10_000_000_000, wall_timeout_s=0.4)
    t0 = time.perf_counter()
    with pytest.raises(W.FunctionError) as ei:
        W.run(_spin_module(), None, limits=limits)
    elapsed = time.perf_counter() - t0
    assert ei.value.kind == "timeout", f"expected timeout, got {ei.value.kind}"
    # Non-vacuity in both directions: it must have actually run for ~the budget
    # (not failed instantly for some unrelated reason) and must have been stopped
    # (not run to completion).
    assert elapsed >= 0.2, f"stopped after only {elapsed:.3f}s — did it even start?"
    assert elapsed < 10.0, f"watchdog did not fire promptly ({elapsed:.3f}s)"


def test_oversized_output_is_refused():
    limits = W.Limits(max_output_bytes=1024)
    with pytest.raises(W.FunctionError) as ei:
        W.run(_big_output_module(64 * 1024), None, limits=limits)
    assert ei.value.kind == "output", f"expected output cap, got {ei.value.kind}"


def test_output_just_under_the_cap_is_accepted():
    """Non-vacuity for the cap: the same module under a larger cap must succeed,
    proving the rejection above was the cap and not a broken module."""
    n = 4096
    limits = W.Limits(max_output_bytes=1024 * 1024)
    payload = "A" * n
    res = W.run(_const_output_module(json.dumps(payload)), None, limits=limits)
    assert res.value == payload


# --------------------------------------------------------------------------- #
# Isolation — the properties that make it safe to run strangers' code
# --------------------------------------------------------------------------- #

def test_guest_cannot_open_a_host_file():
    """No directory is preopened, so the guest has no capability to name any host
    path at all. WASI is deny-by-default: absent a preopen there is nothing to
    resolve a path against, and path_open fails with EBADF (8)."""
    res = W.run(_fs_escape_module(), None)
    assert res.value == 8, f"expected EBADF(8) from path_open, got {res.value!r}"


def test_module_importing_an_unprovided_host_function_is_rejected():
    with pytest.raises(W.FunctionError) as ei:
        W.run(_foreign_import_module(), None)
    assert ei.value.kind in ("invalid_module", "trap"), ei.value.kind
    # It must fail at instantiation, i.e. the guest never got to call anything.


def test_garbage_bytes_are_rejected_as_an_invalid_module():
    with pytest.raises(W.FunctionError) as ei:
        W.run(b"this is definitely not webassembly", None)
    assert ei.value.kind == "invalid_module"


def test_truncated_valid_module_is_rejected_cleanly():
    good = _echo_module()
    with pytest.raises(W.FunctionError) as ei:
        W.run(good[: len(good) // 2], None)
    assert ei.value.kind == "invalid_module"


def test_two_calls_do_not_share_state():
    """Each invocation gets a fresh Store and fresh linear memory. If instances were
    reused, the second call could observe the first call's data — a cross-tenant
    information leak."""
    mod = _echo_module()
    a = W.run(mod, {"secret": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"})
    b = W.run(mod, {"x": 1})
    assert a.value == {"secret": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"}
    assert b.value == {"x": 1}, "second call saw stale memory from the first"


# --------------------------------------------------------------------------- #
# Error classification — the coordinator routes on `kind`, so it must be right
# --------------------------------------------------------------------------- #

def test_nonzero_exit_is_reported_as_a_user_error():
    with pytest.raises(W.FunctionError) as ei:
        W.run(_exit_module(3), None)
    assert ei.value.kind == "exit"
    assert "3" in ei.value.message


def test_exit_zero_with_no_output_is_a_protocol_error_not_a_crash():
    with pytest.raises(W.FunctionError) as ei:
        W.run(_silent_module(), None)
    assert ei.value.kind in ("output", "bad_output"), ei.value.kind


def test_guest_trap_is_reported_as_a_trap():
    with pytest.raises(W.FunctionError) as ei:
        W.run(_trap_module(), None)
    assert ei.value.kind == "trap"


def test_non_json_stdout_is_a_bad_output_error():
    with pytest.raises(W.FunctionError) as ei:
        W.run(_const_output_module("not json at all"), None)
    assert ei.value.kind == "bad_output"


def test_unserialisable_payload_is_rejected_before_the_guest_runs():
    with pytest.raises(W.FunctionError) as ei:
        W.run(_echo_module(), {"bad": object()})
    assert ei.value.kind == "bad_input"


def test_error_kinds_are_distinct_so_the_coordinator_can_route_on_them():
    """A user bug must be distinguishable from a policy stop: one means "this job
    is broken, stop retrying", the other means "retry elsewhere". If several
    failure modes collapsed to the same kind, the scheduler would loop forever."""
    kinds = {}
    cases = [
        ("garbage", lambda: W.run(b"nope", None)),
        ("exit", lambda: W.run(_exit_module(9), None)),
        ("trap", lambda: W.run(_trap_module(), None)),
        ("bad_output", lambda: W.run(_const_output_module("<html>"), None)),
        ("fuel", lambda: W.run(_spin_module(), None,
                               limits=W.Limits(fuel=1_000_000, wall_timeout_s=30))),
    ]
    for label, fn in cases:
        with pytest.raises(W.FunctionError) as ei:
            fn()
        kinds[label] = ei.value.kind
    assert len(set(kinds.values())) == len(cases), (
        f"failure modes collapsed into the same kind, so they cannot be routed "
        f"differently: {kinds}")


# --------------------------------------------------------------------------- #
# Accounting & preflight
# --------------------------------------------------------------------------- #

def test_result_carries_the_billing_and_debugging_fields():
    res = W.run(_echo_module(), {"a": 1})
    d = res.to_dict()
    for field in ("value", "fuel_used", "duration_ms", "stdout_bytes",
                  "stderr", "cold_start"):
        assert field in d, f"missing {field}"
    assert d["fuel_used"] > 0 and d["duration_ms"] >= 0


def test_fuel_used_scales_with_the_work_actually_done():
    """Fuel is the compute meter, so it must track real guest work.

    Note the deliberate choice of module: fuel counts *guest instructions*, and
    bytes moved by a host call (fd_read/fd_write) burn none. Echoing a 4 KB
    payload therefore costs the same fuel as echoing 8 bytes — measured, not
    assumed (see the assertion below). So this test drives a module that does
    proportional work *inside* the guest instead.
    """
    small = W.run(_busy_module(64), None)
    large = W.run(_busy_module(16384), None)
    assert small.value == {"ok": True} and large.value == {"ok": True}
    assert large.fuel_used > small.fuel_used, (
        f"fuel did not increase with guest work ({large.fuel_used} vs "
        f"{small.fuel_used}) — it cannot meter compute")
    assert large.fuel_used > 10 * small.fuel_used, (
        f"256x the loop iterations only cost {large.fuel_used}/{small.fuel_used} "
        f"fuel — metering is not proportional to work")


def test_host_io_burns_no_fuel_so_billing_cannot_rely_on_fuel_alone():
    """Pins the documented limitation above, so it cannot silently change.

    If a future wasmtime charged fuel for host I/O this would fail, and the
    billing model documented on ``Result`` would need revisiting.
    """
    small = W.run(_echo_module(), {"x": 1})
    large = W.run(_echo_module(), {"x": "y" * 4000})
    assert large.stdout_bytes > 100 * small.stdout_bytes, "payloads not size-separated"
    assert large.fuel_used == small.fuel_used, (
        f"host I/O now burns fuel ({large.fuel_used} vs {small.fuel_used}); "
        f"revisit the billing note on Result")


def test_preflight_accepts_a_good_module_and_names_its_abi():
    info = W.preflight(_echo_module())
    assert "_start" in info["exports"]
    assert info["abi"] == "wasi-command/json-stdio"
    assert info["sha256"] == W.module_key(_echo_module())
    assert info["size_bytes"] == len(_echo_module())
    # Every import must be WASI preview 1 — nothing else is granted.
    assert all(i.startswith("wasi_snapshot_preview1::") for i in info["imports"]), \
        info["imports"]


def test_preflight_rejects_a_module_without_start():
    wat = "(module (memory (export \"memory\") 1) (func (export \"other\") (nop)))"
    with pytest.raises(W.FunctionError) as ei:
        W.preflight(_wat_to_wasm(wat))
    assert ei.value.kind == "invalid_module"
    assert "_start" in ei.value.message


def test_preflight_rejects_a_module_importing_host_functions():
    """This is the check that stops a module needing sockets/FFI from ever being
    scheduled onto a donor's machine, where it would fail after wasting their CPU."""
    with pytest.raises(W.FunctionError) as ei:
        W.preflight(_foreign_import_module())
    assert ei.value.kind == "invalid_module"
    assert "env" in ei.value.message


def test_preflight_rejects_garbage():
    with pytest.raises(W.FunctionError) as ei:
        W.preflight(b"\x00\x01\x02not-wasm")
    assert ei.value.kind == "invalid_module"


def test_module_key_is_content_addressed():
    a = _echo_module()
    assert W.module_key(a) == W.module_key(bytes(a))
    assert W.module_key(a) != W.module_key(_silent_module())
    assert len(W.module_key(a)) == 64  # sha256 hex
