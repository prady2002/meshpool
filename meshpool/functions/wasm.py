"""Mesh Functions — a real sandbox that runs **arbitrary user code**.

Why this module exists
----------------------
Until now the batch engine could only dispatch six functions hardcoded into
``meshpool/batch/functions.py``. That made MeshPool a distributed *demo*, not a
compute platform: no user could ever run their own code. This module is the
substrate that fixes it.

The execution model
-------------------
A user function is a **WASI command module** — an ordinary program compiled to
``wasm32-wasip1``. It reads a JSON request on **stdin** and writes a JSON response on
**stdout**. That ABI was chosen deliberately over a custom one because every toolchain
already targets it:

===============  ================================================
language         how the user builds a function
===============  ================================================
Rust             ``cargo build --target wasm32-wasip1 --release``
Go / TinyGo      ``GOOS=wasip1 GOARCH=wasm go build``
C / C++ / Zig    wasi-sdk / ``zig build-exe -target wasm32-wasi``
JavaScript       ``javy build``
Python           ``componentize-py`` / ``py2wasm``
===============  ================================================

No SDK to import, no FFI, no memory-layout contract to get wrong: if it reads stdin
and writes stdout, it runs on MeshPool.

The isolation boundary (this is the part that matters)
------------------------------------------------------
The previous "sandbox" was a subprocess with an RLIMIT. That is not a boundary --
it shares the host's filesystem, network, and syscall surface, so it can never be
pointed at a stranger's machine. WASM is, because the guest can only reach what it is
explicitly handed:

* **No filesystem.** No directory is preopened, so every ``path_open`` fails with
  ``ENOTCAPABLE``. The guest cannot even *name* a host path.
* **No network.** WASI preview 1 has no socket API and ``wasi-http`` is not linked in.
  There is no reachable syscall that opens a connection -- egress is impossible by
  construction rather than by filtering.
* **No host functions.** Only the WASI preview-1 shim is defined on the linker; a
  module importing anything else fails to *instantiate* rather than trapping later.
* **Bounded CPU** via wasmtime fuel, which counts executed instructions. This is
  deterministic and machine-independent: the same function costs the same fuel on a
  laptop and on a server, so it can be metered and billed fairly.
* **Bounded wall-clock** via epoch interruption, because fuel alone does not stop a
  guest blocked in a host call (e.g. ``poll_oneoff`` sleeping).
* **Bounded memory** via store limits: growing past the cap fails the ``memory.grow``
  rather than the donor's machine.
* **Bounded output**, so a guest cannot fill a donor's disk by printing forever.
* **Fresh instance per call.** State cannot leak between two users' invocations.

Cold start is 1-5 ms once a module is compiled, and compiled modules are cached by
content hash, so the second call skips compilation entirely.
"""
from __future__ import annotations

import json
import os
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

try:  # optional dependency: nodes without it simply cannot host functions
    from wasmtime import (Config, Engine, ExitTrap, Linker, Module, Store, Trap,
                          TrapCode, WasiConfig, WasmtimeError)
    WASM_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only on installs without wasmtime
    WASM_AVAILABLE = False
    Config = Engine = ExitTrap = Linker = Module = Store = WasiConfig = None  # type: ignore
    Trap = TrapCode = None  # type: ignore
    WasmtimeError = Exception  # type: ignore


class FunctionError(Exception):
    """A user function failed. ``kind`` says whose fault it was and why."""

    def __init__(self, kind: str, message: str, stderr: str = "") -> None:
        super().__init__(message)
        self.kind = kind          # invalid_module | timeout | fuel | memory |
        self.message = message    # exit | output | bad_output | trap | unavailable
        self.stderr = stderr

    def to_dict(self) -> dict:
        out = {"error": self.message, "kind": self.kind}
        if self.stderr:
            out["stderr"] = self.stderr
        return out


@dataclass(frozen=True)
class Limits:
    """Resource ceilings for one invocation.

    Defaults are sized for a "function", not a batch job: a few hundred million
    instructions is ~0.1-1 s of real work, which comfortably covers request handlers,
    transforms and parsers while stopping a runaway loop almost immediately.
    """

    fuel: int = 500_000_000
    memory_bytes: int = 256 * 1024 * 1024
    wall_timeout_s: float = 10.0
    max_output_bytes: int = 8 * 1024 * 1024
    max_stderr_bytes: int = 16 * 1024

    def scaled(self, factor: float) -> "Limits":
        return Limits(
            fuel=int(self.fuel * factor),
            memory_bytes=self.memory_bytes,
            wall_timeout_s=self.wall_timeout_s * factor,
            max_output_bytes=self.max_output_bytes,
            max_stderr_bytes=self.max_stderr_bytes,
        )


@dataclass
class Result:
    """Everything the coordinator needs to bill, debug and verify one call.

    ``fuel_used`` counts **guest instructions only**. Work done inside a host call
    (``fd_read``/``fd_write`` moving bytes, WASI syscalls) burns no fuel, so an
    I/O-heavy function can move megabytes for a near-constant fuel cost. Fuel is
    therefore the right meter for *compute*, but billing must also account for
    ``stdout_bytes`` and ``duration_ms``, or a function that mostly shuffles bytes
    would be metered as free.
    """

    value: Any
    fuel_used: int
    duration_ms: float
    stdout_bytes: int
    stderr: str
    cold_start: bool

    def to_dict(self) -> dict:
        return {
            "value": self.value,
            "fuel_used": self.fuel_used,
            "duration_ms": round(self.duration_ms, 3),
            "stdout_bytes": self.stdout_bytes,
            "stderr": self.stderr,
            "cold_start": self.cold_start,
        }


# --- module compilation cache --------------------------------------------- #
# Compiling a multi-megabyte module takes tens of milliseconds to seconds; running it
# takes microseconds. Caching by content hash is what turns "cold start" into a
# once-per-module cost instead of a per-request one. Modules are immutable and
# content-addressed, so the cache can never be stale.

_ENGINE: Optional["Engine"] = None
_ENGINE_LOCK = threading.Lock()
_MODULES: "dict[str, Module]" = {}
_MODULE_LOCK = threading.Lock()


def _engine() -> "Engine":
    global _ENGINE
    with _ENGINE_LOCK:
        if _ENGINE is None:
            cfg = Config()
            # Fuel = deterministic instruction metering (billing + runaway protection).
            cfg.consume_fuel = True
            # Epochs = the only way to interrupt a guest blocked inside a host call.
            cfg.epoch_interruption = True
            # A guest that recurses forever must hit a stack limit, not the host's.
            cfg.max_wasm_stack = 2 * 1024 * 1024
            # Deny non-standard proposals: a smaller accepted language is a smaller
            # attack surface, and everything below is off by default in most toolchains.
            cfg.wasm_threads = False          # no shared memory, no data races
            cfg.wasm_reference_types = True
            cfg.wasm_bulk_memory = True
            cfg.wasm_multi_value = True
            cfg.wasm_simd = True
            _ENGINE = Engine(cfg)
        return _ENGINE


def module_key(wasm_bytes: bytes) -> str:
    import hashlib
    return hashlib.sha256(wasm_bytes).hexdigest()


def compile_module(wasm_bytes: bytes) -> "tuple[Module, bool]":
    """Compile (or fetch from cache) a user module. Returns ``(module, was_cold)``."""
    if not WASM_AVAILABLE:
        raise FunctionError("unavailable",
                            "wasmtime is not installed on this node; it cannot host "
                            "Mesh Functions (install with: pip install wasmtime)")
    key = module_key(wasm_bytes)
    with _MODULE_LOCK:
        cached = _MODULES.get(key)
    if cached is not None:
        return cached, False
    try:
        mod = Module(_engine(), wasm_bytes)
    except WasmtimeError as exc:
        raise FunctionError("invalid_module", f"not a valid WebAssembly module: {exc}")
    with _MODULE_LOCK:
        _MODULES.setdefault(key, mod)
    return mod, True


def cache_size() -> int:
    with _MODULE_LOCK:
        return len(_MODULES)


def clear_cache() -> None:
    with _MODULE_LOCK:
        _MODULES.clear()


# --- invocation ------------------------------------------------------------ #

def run(wasm_bytes: bytes, payload: Any, limits: Optional[Limits] = None,
        env: Optional[dict] = None, argv: Optional[list] = None) -> Result:
    """Run a user function once and return its JSON result.

    ``payload`` is serialised to JSON on the guest's stdin; the guest's stdout is
    parsed back as JSON. Any failure raises :class:`FunctionError` whose ``kind``
    distinguishes a user bug (``exit``, ``bad_output``, ``trap``) from a policy stop
    (``timeout``, ``fuel``, ``memory``, ``output``) -- the coordinator needs that
    distinction to decide between "retry elsewhere" and "this job is broken".
    """
    limits = limits or Limits()
    module, cold = compile_module(wasm_bytes)
    engine = _engine()

    try:
        request = json.dumps(payload).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise FunctionError("bad_input", f"payload is not JSON-serialisable: {exc}")

    # wasmtime's WASI plumbing works on real files, so a scratch dir holds the three
    # streams. It is created per call and removed immediately; the guest never sees
    # it, because it is not preopened -- only the host holds the descriptors.
    tmp = Path(tempfile.mkdtemp(prefix="meshfn-"))
    stdin_p, stdout_p, stderr_p = tmp / "in", tmp / "out", tmp / "err"
    stdin_p.write_bytes(request)
    stdout_p.touch()
    stderr_p.touch()

    timer: Optional[threading.Timer] = None
    timed_out = threading.Event()
    started = time.perf_counter()
    fuel_left = 0
    try:
        store = Store(engine)
        store.set_fuel(limits.fuel)
        store.set_limits(memory_size=limits.memory_bytes)
        # One tick past the deadline is enough to trap; the watchdog fires exactly once.
        store.set_epoch_deadline(1)

        wasi = WasiConfig()
        wasi.argv = list(argv) if argv else ["function"]
        wasi.env = [(str(k), str(v)) for k, v in (env or {}).items()]
        wasi.stdin_file = str(stdin_p)
        wasi.stdout_file = str(stdout_p)
        wasi.stderr_file = str(stderr_p)
        # NOTE: no preopen_dir call. That omission *is* the filesystem sandbox.
        store.set_wasi(wasi)

        linker = Linker(engine)
        linker.define_wasi()   # the only imports that exist for the guest

        def fire() -> None:
            timed_out.set()
            engine.increment_epoch()

        timer = threading.Timer(limits.wall_timeout_s, fire)
        timer.daemon = True
        timer.start()

        exit_code = 0
        try:
            try:
                instance = linker.instantiate(store, module)
            except (ExitTrap, Trap):
                raise
            except WasmtimeError as exc:
                # Genuinely could not be instantiated (missing import, bad signature).
                # That is a packaging error and gets its own kind so the user is told
                # to rebuild rather than to retry somewhere else.
                stderr = _read_capped(stderr_p, limits.max_stderr_bytes)
                raise FunctionError(
                    "invalid_module",
                    f"module could not be instantiated: {exc}", stderr)
            start = instance.exports(store).get("_start")
            if start is None:
                raise FunctionError(
                    "invalid_module",
                    "module has no _start export — build a WASI *command* "
                    "(a program with a main), not a library")
            start(store)
        except ExitTrap as exc:
            # WASI proc_exit: the normal way a command module finishes.
            exit_code = int(getattr(exc, "code", 0) or 0)
        except Trap as exc:
            stderr = _read_capped(stderr_p, limits.max_stderr_bytes)
            raise _classify_trap(exc, timed_out, stderr)
        except WasmtimeError as exc:
            # The module instantiated fine and then failed *while running* — e.g. it
            # passed an out-of-range value to a WASI call. That is a guest bug, not a
            # packaging bug, so it must NOT be reported as "invalid_module": the user
            # would be told to rebuild a module that is in fact well-formed.
            stderr = _read_capped(stderr_p, limits.max_stderr_bytes)
            if timed_out.is_set():
                raise FunctionError(
                    "timeout",
                    f"function exceeded its {limits.wall_timeout_s:.3g}s wall-clock "
                    "limit", stderr)
            raise FunctionError(
                "trap", f"function failed at runtime: {exc}", stderr)
        finally:
            fuel_left = _fuel_left(store)

        stderr = _read_capped(stderr_p, limits.max_stderr_bytes)
        if exit_code != 0:
            raise FunctionError("exit", f"function exited with status {exit_code}", stderr)

        size = stdout_p.stat().st_size
        if size > limits.max_output_bytes:
            raise FunctionError(
                "output",
                f"function wrote {size} bytes, over the {limits.max_output_bytes}-byte "
                "output limit", stderr)
        raw = stdout_p.read_bytes()
        if not raw.strip():
            raise FunctionError(
                "bad_output",
                "function produced no output — write the JSON result to stdout", stderr)
        try:
            value = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            preview = raw[:200].decode("utf-8", "replace")
            raise FunctionError(
                "bad_output",
                f"stdout is not valid JSON ({exc}); first bytes: {preview!r}", stderr)

        return Result(
            value=value,
            fuel_used=max(0, limits.fuel - fuel_left),
            duration_ms=(time.perf_counter() - started) * 1000.0,
            stdout_bytes=size,
            stderr=stderr,
            cold_start=cold,
        )
    finally:
        if timer is not None:
            timer.cancel()
        _rmtree(tmp)


def _fuel_left(store: "Store") -> int:
    try:
        return int(store.get_fuel())
    except Exception:  # noqa: BLE001 — fuel is unreadable after some traps
        return 0


def _classify_trap(exc: "Trap", timed_out: threading.Event, stderr: str) -> FunctionError:
    """Turn a wasmtime trap into an actionable reason.

    The interesting cases all arrive as one exception type but demand opposite
    responses: a timeout should be retried on a faster node, fuel exhaustion means the
    job needs a bigger budget, and a genuine trap means the user's code is broken.
    ``trap_code`` is used rather than message text so this stays correct across
    wasmtime releases.
    """
    code = getattr(exc, "trap_code", None)
    if code == TrapCode.INTERRUPT or timed_out.is_set():
        return FunctionError(
            "timeout", "function exceeded its wall-clock limit", stderr)
    if code == TrapCode.OUT_OF_FUEL:
        return FunctionError(
            "fuel", "function exceeded its CPU (fuel) budget", stderr)
    if code in (TrapCode.MEMORY_OUT_OF_BOUNDS, TrapCode.ALLOCATION_TOO_LARGE,
                TrapCode.HEAP_MISALIGNED, TrapCode.TABLE_OUT_OF_BOUNDS):
        return FunctionError("memory", f"function made an invalid memory access: {exc}",
                             stderr)
    if code == TrapCode.STACK_OVERFLOW:
        return FunctionError("memory", "function overflowed its stack "
                                       "(infinite recursion?)", stderr)
    label = {
        TrapCode.UNREACHABLE: "executed an unreachable instruction "
                              "(a panic/abort in most languages)",
        TrapCode.INTEGER_DIVISION_BY_ZERO: "divided by zero",
        TrapCode.INTEGER_OVERFLOW: "overflowed an integer conversion",
        TrapCode.BAD_CONVERSION_TO_INTEGER: "made an invalid float→int conversion",
        TrapCode.INDIRECT_CALL_TO_NULL: "called a null function pointer",
        TrapCode.BAD_SIGNATURE: "called a function with the wrong signature",
    }.get(code)
    if label:
        return FunctionError("trap", f"function {label}", stderr)
    return FunctionError("trap", f"function trapped: {exc}", stderr)


def _read_capped(path: Path, cap: int) -> str:
    try:
        raw = path.read_bytes()
    except OSError:
        return ""
    if len(raw) > cap:
        raw = raw[:cap] + b"\n...[truncated]"
    return raw.decode("utf-8", "replace")


def _rmtree(path: Path) -> None:
    import shutil
    try:
        shutil.rmtree(path, ignore_errors=True)
    except OSError:  # pragma: no cover
        pass


def preflight(wasm_bytes: bytes) -> dict:
    """Validate a module at upload time so users learn about problems immediately.

    Deploy-time failure is worth a lot: without this a broken module is only
    discovered when a job runs on somebody else's machine, wasting their CPU and the
    user's time. Returns a description of the module's imports and exports.
    """
    module, _ = compile_module(wasm_bytes)
    imports = []
    for imp in module.imports:
        imports.append(f"{imp.module}::{imp.name}")
    exports = [e.name for e in module.exports]
    if "_start" not in exports:
        raise FunctionError(
            "invalid_module",
            "module has no _start export — build a WASI command (a program with a "
            "main), not a library")
    # Anything outside the WASI preview-1 shim can never be satisfied here, so say so
    # now rather than letting instantiation fail on a donor's machine.
    foreign = sorted({i.split("::")[0] for i in imports
                      if not i.startswith("wasi_snapshot_preview1::")
                      and not i.startswith("wasi_unstable::")})
    if foreign:
        raise FunctionError(
            "invalid_module",
            "module imports host modules MeshPool does not provide: "
            + ", ".join(foreign)
            + " — MeshPool grants WASI preview 1 only (no sockets, no host FFI)")
    return {
        "sha256": module_key(wasm_bytes),
        "size_bytes": len(wasm_bytes),
        "imports": sorted(imports),
        "exports": sorted(exports),
        "abi": "wasi-command/json-stdio",
    }


__all__ = ["Limits", "Result", "FunctionError", "run", "preflight", "compile_module",
           "module_key", "cache_size", "clear_cache", "WASM_AVAILABLE"]
