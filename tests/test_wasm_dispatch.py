"""User WASM is dispatched through the isolated child process, not the shared pool.

This pins a real availability defect. A four-line infinite-loop module submitted as
a normal batch unit used to run inside the agent's ``ProcessPoolExecutor``. When it
was stopped, the worker process died, the executor raised ``BrokenProcessPool``, and
the node was **out of service for every other job** until it restarted — one user's
bad code could take down a volunteer's machine for everyone.
"""
from __future__ import annotations

import asyncio
import base64
import hashlib

import pytest

wasmtime = pytest.importorskip("wasmtime", reason="wasmtime not installed")

from meshpool.agent.runtime import ProcessRuntime  # noqa: E402
from meshpool.batch.functions import is_wasm_function  # noqa: E402
from meshpool.common.models import DonationPolicy  # noqa: E402

SPIN_WAT = """
(module (memory (export "memory") 1)
  (func (export "_start") (local $i i64)
    (loop $f (local.set $i (i64.add (local.get $i) (i64.const 1))) (br $f))))
"""


def _spin_job() -> tuple[str, dict]:
    blob = bytes(wasmtime.wat2wasm(SPIN_WAT))
    name = "wasm:" + hashlib.sha256(blob).hexdigest()
    return name, {"_wasm_module": base64.b64encode(blob).decode("ascii")}


def test_wasm_function_names_are_recognised():
    assert is_wasm_function("wasm:" + "a" * 64)
    assert not is_wasm_function("square")
    assert not is_wasm_function("")


def test_runaway_user_module_does_not_take_the_node_out_of_service():
    name, payload = _spin_job()

    async def scenario():
        # sandbox=False is the Tier 0/1 "fast shared pool" configuration — the exact
        # setting under which this used to break the node.
        rt = ProcessRuntime(2, DonationPolicy(cpu_pct=100, ram_mb=2048),
                            sandbox=False)
        try:
            with pytest.raises(Exception) as ei:
                await rt.run_unit(name, payload)
            # Non-vacuity: it must have been stopped for exceeding its budget, not
            # have failed for some unrelated reason (e.g. module never loaded).
            msg = str(ei.value).lower()
            assert ("fuel" in msg or "timeout" in msg or "budget" in msg), \
                f"runaway stopped for an unexpected reason: {ei.value}"

            # THE POINT: the node still works. Before the fix this raised
            # BrokenProcessPool("the process pool is not usable anymore").
            result, _ms = await rt.run_unit("square", {"x": 7})
            assert result == 49, f"node returned {result!r} after a runaway unit"

            # And it keeps working for more than one follow-up call.
            result2, _ = await rt.run_unit("is_prime", {"n": 97})
            assert result2 == {"n": 97, "prime": True}
        finally:
            rt.shutdown()

    asyncio.run(scenario())


def test_a_normal_wasm_unit_still_returns_its_result():
    """The isolation fix must not break the happy path."""
    echo = bytes(wasmtime.wat2wasm(r"""
(module
  (import "wasi_snapshot_preview1" "fd_read"
    (func $fd_read (param i32 i32 i32 i32) (result i32)))
  (import "wasi_snapshot_preview1" "fd_write"
    (func $fd_write (param i32 i32 i32 i32) (result i32)))
  (memory (export "memory") 2)
  (func (export "_start")
    (i32.store (i32.const 0) (i32.const 64))
    (i32.store (i32.const 4) (i32.const 8192))
    (drop (call $fd_read (i32.const 0) (i32.const 0) (i32.const 1) (i32.const 16)))
    (i32.store (i32.const 0) (i32.const 64))
    (i32.store (i32.const 4) (i32.load (i32.const 16)))
    (drop (call $fd_write (i32.const 1) (i32.const 0) (i32.const 1) (i32.const 20)))))
"""))
    name = "wasm:" + hashlib.sha256(echo).hexdigest()
    payload = {"value": 123, "_wasm_module": base64.b64encode(echo).decode("ascii")}

    async def scenario():
        rt = ProcessRuntime(2, DonationPolicy(cpu_pct=100, ram_mb=2048),
                            sandbox=False)
        try:
            result, compute_ms = await rt.run_unit(name, payload)
            # The guest must NOT see MeshPool's internal plumbing.
            assert result == {"value": 123}, result
            assert "_wasm_module" not in result
            assert compute_ms >= 0
        finally:
            rt.shutdown()

    asyncio.run(scenario())


def test_wrong_module_bytes_are_refused_by_content_address():
    """A node handed bytes that do not match the requested hash must refuse to run
    them — otherwise a malicious peer could substitute code."""
    from meshpool.batch import functions as F

    good = bytes(wasmtime.wat2wasm("(module (func (export \"_start\") (nop)))"))
    wrong_name = "wasm:" + "0" * 64
    with pytest.raises(ValueError, match="hash mismatch"):
        F.run_function(wrong_name,
                       {"_wasm_module": base64.b64encode(good).decode("ascii")})


def test_missing_module_is_a_clear_error_not_a_crash():
    from meshpool.batch import functions as F

    name = "wasm:" + "b" * 64
    with pytest.raises(ValueError, match="not delivered"):
        F.run_function(name, {})
