"""Locked-down execution sandbox for curated workloads (spec §5, global Tier 2).

At Tier 0/1 (LAN/office, mutual trust) work runs in the in-process worker pool with
CPU-slot enforcement only — fast, and unchanged. On the **public** mesh we accept
work units from *strangers* onto our own machine, so each unit must run under hard,
enforced limits. This module runs a single curated function in a **dedicated child
process** with:

* **Memory cap** — Windows: a Job Object with ``JOB_OBJECT_LIMIT_PROCESS_MEMORY``
  (+ ``KILL_ON_JOB_CLOSE`` + ``DIE_ON_UNHANDLED_EXCEPTION`` + a single-process cap so
  a unit can't fork-bomb). POSIX: ``RLIMIT_AS``.
* **Wall-clock timeout** — the parent kills the child (and, on Windows, its whole job)
  if it overruns, so a hung/however-malicious unit can't pin a donor's core forever.
* **No filesystem writes** — POSIX: ``RLIMIT_FSIZE = 0``. Curated functions never
  write files, so this is defence-in-depth (Windows relies on that by-construction
  property until the container runtime lands).
* **Bounded output** — the child's result is size-capped so it can't exhaust the
  donor's memory by returning a huge blob.
* **Curated allow-list + input validation** — only the fixed ``batch.functions``
  registry runs, and each payload is validated (``validate_payload``) *inside* the
  sandbox so an abusive parameter (e.g. ``rounds = 10**12``) is rejected before it
  can burn a donor's CPU.

Network isolation for *arbitrary* code is delivered by the container runtime (a later
phase); curated functions are network-free by construction, so nothing here opens a
socket. Keeping this seam stable means turning on real container isolation later is a
runtime swap, not a rewrite.
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Any, Optional


class SandboxError(RuntimeError):
    """Raised when a unit breaches a sandbox limit or the worker fails to produce a
    valid result (timeout, out-of-memory, oversized output, crash, bad input)."""


@dataclass(frozen=True)
class SandboxLimits:
    wall_clock_s: float = 30.0
    memory_mb: int = 512
    max_output_bytes: int = 1_000_000

    @classmethod
    def from_env(cls) -> "SandboxLimits":
        def _f(name: str, default: float) -> float:
            try:
                return float(os.environ.get(name, "") or default)
            except ValueError:
                return default
        return cls(
            wall_clock_s=_f("MESHPOOL_SANDBOX_WALL_S", 30.0),
            memory_mb=int(_f("MESHPOOL_SANDBOX_MEM_MB", 512)),
            max_output_bytes=int(_f("MESHPOOL_SANDBOX_OUT_BYTES", 1_000_000)),
        )


# A self-test-only function name understood by the *worker* but deliberately NOT in
# the batch registry, so it can never be dispatched through the real engine. It lets
# the test-suite prove memory enforcement deterministically on every platform.
SELFTEST_ALLOC = "__sandbox_selftest_alloc__"

# Test-only: attempt exactly one outbound TCP connection and report the outcome as
# structured data. Like SELFTEST_ALLOC it is deliberately absent from the batch
# registry, so it can never be dispatched through the real engine. It exists because
# every other egress assertion in the suite exercises an EgressPolicy *object*, which
# proves the policy's logic but not that any sandboxed workload is actually subject
# to it. This is the only way to prove enforcement is live in the child, where it
# is the thing standing between a donated PC and a botnet.
SELFTEST_CONNECT = "__sandbox_selftest_connect__"


# --------------------------------------------------------------------------- #
# Parent side: spawn a limited child and enforce wall-clock + memory + output.
# --------------------------------------------------------------------------- #
def _posix_preexec(limits: SandboxLimits):  # pragma: no cover - exercised on POSIX
    import resource

    def _apply() -> None:
        mem = int(limits.memory_mb) * 1024 * 1024
        for res, soft_hard in (
            (resource.RLIMIT_AS, (mem, mem)),          # address space cap
            (resource.RLIMIT_FSIZE, (0, 0)),           # no filesystem writes
            (resource.RLIMIT_CPU, (int(math.ceil(limits.wall_clock_s)) + 1,) * 2),
        ):
            try:
                resource.setrlimit(res, soft_hard)
            except (ValueError, OSError):
                pass

    return _apply


def _win_assign_job(proc: subprocess.Popen, limits: SandboxLimits):  # pragma: no cover
    """Put the child in a memory-capped Job Object that dies when we close it.

    Returns an opaque handle to close later, or ``None`` if the platform refused
    (memory enforcement is then best-effort; the wall-clock timeout still applies).
    """
    import ctypes
    from ctypes import wintypes

    JOB_OBJECT_LIMIT_ACTIVE_PROCESS = 0x00000008
    JOB_OBJECT_LIMIT_PROCESS_MEMORY = 0x00000100
    JOB_OBJECT_LIMIT_DIE_ON_UNHANDLED_EXCEPTION = 0x00000400
    JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
    JobObjectExtendedLimitInformation = 9

    ULONG_PTR = ctypes.c_size_t

    class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_int64),
            ("PerJobUserTimeLimit", ctypes.c_int64),
            ("LimitFlags", wintypes.DWORD),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", wintypes.DWORD),
            ("Affinity", ULONG_PTR),
            ("PriorityClass", wintypes.DWORD),
            ("SchedulingClass", wintypes.DWORD),
        ]

    class IO_COUNTERS(ctypes.Structure):
        _fields_ = [(n, ctypes.c_uint64) for n in (
            "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
            "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]

    class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", JOBOBJECT_BASIC_LIMIT_INFORMATION),
            ("IoInfo", IO_COUNTERS),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]

    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.CreateJobObjectW.restype = wintypes.HANDLE
    k32.CreateJobObjectW.argtypes = [wintypes.LPVOID, wintypes.LPCWSTR]
    k32.SetInformationJobObject.restype = wintypes.BOOL
    k32.SetInformationJobObject.argtypes = [
        wintypes.HANDLE, ctypes.c_int, wintypes.LPVOID, wintypes.DWORD]
    k32.AssignProcessToJobObject.restype = wintypes.BOOL
    k32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]

    try:
        job = k32.CreateJobObjectW(None, None)
        if not job:
            return None
        info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
        info.BasicLimitInformation.LimitFlags = (
            JOB_OBJECT_LIMIT_PROCESS_MEMORY
            | JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
            | JOB_OBJECT_LIMIT_DIE_ON_UNHANDLED_EXCEPTION
            | JOB_OBJECT_LIMIT_ACTIVE_PROCESS)
        info.BasicLimitInformation.ActiveProcessLimit = 1
        info.ProcessMemoryLimit = int(limits.memory_mb) * 1024 * 1024
        if not k32.SetInformationJobObject(
                job, JobObjectExtendedLimitInformation,
                ctypes.byref(info), ctypes.sizeof(info)):
            k32.CloseHandle(job)
            return None
        if not k32.AssignProcessToJobObject(job, int(proc._handle)):
            k32.CloseHandle(job)
            return None
        return job
    except Exception:  # noqa: BLE001 — never let sandboxing crash the agent
        return None


def _close_job(job) -> None:  # pragma: no cover - Windows only
    if job is None:
        return
    try:
        import ctypes
        ctypes.WinDLL("kernel32").CloseHandle(job)
    except Exception:  # noqa: BLE001
        pass


def run_sandboxed(func: str, payload: Optional[dict] = None,
                  limits: Optional[SandboxLimits] = None,
                  return_timing: bool = False,
                  allow_selftest: bool = False) -> Any:
    """Run one curated function in a limited child process and return its result.

    Blocking. Raises :class:`SandboxError` on any limit breach or worker failure.
    When ``return_timing`` is set, returns ``(result, compute_ms)`` where
    ``compute_ms`` is the child's *actual* function-execution time (excluding the
    process-spawn / IPC overhead) so the coordinator can bill only real compute.
    """
    limits = limits or SandboxLimits()
    payload = payload or {}
    request = json.dumps({"func": func, "payload": payload,
                          "selftest": bool(allow_selftest)}).encode() + b"\n"

    is_win = sys.platform == "win32"
    # Ensure the child can import `meshpool` no matter what cwd it inherits.
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    child_env = dict(os.environ)
    child_env["PYTHONPATH"] = repo_root + os.pathsep + child_env.get("PYTHONPATH", "")
    popen_kwargs: dict = dict(
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        env=child_env)
    if is_win:
        # No console window; don't inherit the parent's std handles into the job.
        popen_kwargs["creationflags"] = 0x08000000  # CREATE_NO_WINDOW
    else:
        popen_kwargs["preexec_fn"] = _posix_preexec(limits)  # type: ignore[assignment]

    proc = subprocess.Popen(
        [sys.executable, "-m", "meshpool.agent.sandbox"], **popen_kwargs)

    job = _win_assign_job(proc, limits) if is_win else None
    try:
        # The worker blocks reading stdin until we send the request below, so on
        # Windows the job assignment above always precedes any heavy work.
        out, err = proc.communicate(input=request, timeout=limits.wall_clock_s)
    except subprocess.TimeoutExpired:
        _kill(proc)
        raise SandboxError(
            f"wall-clock timeout after {limits.wall_clock_s:g}s running {func!r}")
    finally:
        _close_job(job)  # KILL_ON_JOB_CLOSE reaps the child if it somehow survived

    if len(out) > limits.max_output_bytes:
        raise SandboxError(
            f"result exceeded output cap ({len(out)} > {limits.max_output_bytes} bytes)")
    if proc.returncode != 0 or not out.strip():
        detail = (err.decode("utf-8", "replace")[:200].strip()
                  or f"worker exited with code {proc.returncode}")
        raise SandboxError(f"sandboxed worker failed ({func!r}): {detail}")
    try:
        parsed = json.loads(out.decode("utf-8", "replace"))
    except ValueError:
        raise SandboxError(f"sandboxed worker returned non-JSON output for {func!r}")
    if not parsed.get("ok"):
        raise SandboxError(parsed.get("error") or f"unit {func!r} failed in sandbox")
    if return_timing:
        return parsed["result"], float(parsed.get("compute_ms") or 0.0)
    return parsed["result"]


def _kill(proc: subprocess.Popen) -> None:
    try:
        proc.kill()
    except Exception:  # noqa: BLE001
        pass
    try:
        proc.wait(timeout=5)
    except Exception:  # noqa: BLE001
        pass


# --------------------------------------------------------------------------- #
# Child side: read one request, run one curated function, print one JSON result.
# --------------------------------------------------------------------------- #
def _worker_main() -> int:
    # Install deny-by-default egress hook in this child process.
    # Curated functions are network-free by construction, so this is defence-in-depth.
    # For future code paths that attempt connections, this catches them before the OS call.
    #
    # This fails CLOSED. A sandbox whose egress filter did not install is not a sandbox,
    # and running a stranger's work inside it is precisely the botnet/DDoS/spam-relay risk
    # the filter exists to prevent. Silently continuing would leave the node unprotected
    # while every test and dashboard still reported "sandboxed", which is worse than an
    # outright failure because nobody would ever look.
    try:
        from ..common.egress import EgressPolicy, install_egress_hook
        install_egress_hook(EgressPolicy())  # deny-by-default
    except Exception as exc:  # noqa: BLE001
        sys.stdout.write(json.dumps({
            "ok": False,
            "error": f"EgressUnavailable: refusing to run work without egress enforcement: "
                     f"{type(exc).__name__}: {exc}",
        }))
        sys.stdout.flush()
        return 3
    raw = sys.stdin.readline()
    try:
        req = json.loads(raw)
        func = req["func"]
        payload = req.get("payload") or {}
        # The selftest branches below bypass validate_payload() and can steer the
        # child's own egress policy, which is exactly what a test needs and exactly
        # what a caller must never be able to ask for. `func` and `payload` come
        # from the work unit and are therefore attacker-controlled, so the switch
        # that unlocks them is a run_sandboxed() keyword argument instead: it is set
        # at the Python call site by test code, and a submitted unit has no way to
        # spell it. Without it a selftest name is just an unknown function.
        selftest_ok = bool(req.get("selftest"))
        if func in (SELFTEST_ALLOC, SELFTEST_CONNECT) and not selftest_ok:
            raise ValueError(f"unknown function: {func}")
        if func == SELFTEST_ALLOC:
            # Only reachable through run_sandboxed() — never via the batch engine.
            mb = int(payload.get("mb", 64))
            t0 = time.perf_counter()
            buf = bytearray(mb * 1024 * 1024)
            for i in range(0, len(buf), 4096):   # touch pages so the OS commits them
                buf[i] = 1
            compute_ms = (time.perf_counter() - t0) * 1000.0
            result = {"allocated_mb": mb, "checksum": buf[0]}
        elif func == SELFTEST_CONNECT:
            # Only reachable through run_sandboxed() — never via the batch engine.
            import socket as _socket
            host = str(payload.get("host", "127.0.0.1"))
            port = int(payload.get("port", 0))
            whitelist = [(str(h), int(p)) for h, p in payload.get("whitelist", [])]
            # Only override the policy the production path installed when a test
            # explicitly asks for a different one. Re-installing unconditionally made
            # this selftest prove its OWN hook worked rather than the one installed
            # above: deleting that install still left this branch installing a
            # deny-by-default policy, so the connection was still refused and the
            # gate still passed. That is precisely the blindness this selftest exists
            # to remove, so the default path must inherit whatever _worker_main set up.
            if "whitelist" in payload or "allow_all" in payload:
                install_egress_hook(EgressPolicy(
                    whitelist=whitelist, allow_all=bool(payload.get("allow_all", False))))
            original_calls: list[tuple[str, int]] = []
            if payload.get("fake_original_connect"):
                from ..common import egress as _egress

                def _fake_connect(_sock, address) -> None:
                    original_calls.append((str(address[0]), int(address[1])))
                    return None

                _egress._ORIGINAL_CONNECT = _fake_connect  # type: ignore[attr-defined]
            t0 = time.perf_counter()
            # "nonce" is echoed back so the caller can distinguish "the child ran and
            # was refused" from "the child never ran", which otherwise look identical.
            outcome = {"attempted": True, "nonce": payload.get("nonce"),
                       "connected": False, "denied": False, "error": None,
                       "original_connect_calls": 0}
            try:
                conn = _socket.create_connection((host, port), timeout=3.0)
            except Exception as exc:  # noqa: BLE001
                # The hook re-raises EgressDenied as ConnectionRefusedError so blocked
                # destinations are indistinguishable from ordinary network failures to
                # the workload, but it chains the original via `from exc`. Inspecting
                # __cause__ tells a *policy* refusal apart from a genuinely closed port
                # without depending on message text.
                cause = getattr(exc, "__cause__", None)
                outcome["error"] = f"{type(exc).__name__}: {exc}"
                outcome["denied"] = "EgressDenied" in (type(exc).__name__, type(cause).__name__)
            else:
                outcome["connected"] = True
                conn.close()
            outcome["original_connect_calls"] = len(original_calls)
            compute_ms = (time.perf_counter() - t0) * 1000.0
            result = outcome
        else:
            from ..batch.functions import run_function, validate_payload
            validate_payload(func, payload)
            # Time only the function itself, so the billed compute excludes this
            # child's one-off interpreter/import startup (a fresh process per unit).
            t0 = time.perf_counter()
            result = run_function(func, payload)
            compute_ms = (time.perf_counter() - t0) * 1000.0
        sys.stdout.write(json.dumps({"ok": True, "result": result, "compute_ms": compute_ms}))
    except MemoryError:
        sys.stdout.write(json.dumps({"ok": False, "error": "MemoryError: memory cap exceeded"}))
    except Exception as exc:  # noqa: BLE001
        sys.stdout.write(json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"}))
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(_worker_main())
