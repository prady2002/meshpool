"""Execution runtime abstraction (spec §4.1, §5, §6).

The scheduler's contract is "run this unit / start this replica with these limits";
the *runtime* decides how. This lets us swap Process -> Docker -> Firecracker
without touching the scheduler (spec §6: "swap runtime later without changing the
scheduler's contract").

* ``ProcessRuntime`` (default here): each CPU-bound work unit runs in a separate
  OS process via a pool whose size is derived from the donation CPU cap — that is
  how "donate up to N% CPU" is *enforced* (never trust the workload to self-limit).
  Instant reclaim hard-terminates the pool's child processes.
* ``DockerRuntime`` (stub): documented drop-in for when Docker is available; uses
  the same interface with cgroup/Job-Object quotas + network isolation.
"""
from __future__ import annotations

import asyncio
import collections
import functools
import math
import os
import shlex
import socket
import subprocess
import sys
import threading
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Optional, Protocol

from ..common.logging import get_logger
from ..common.models import DonationPolicy
from ..batch.functions import is_wasm_function, run_function
from .sandbox import SandboxLimits, run_sandboxed

log = get_logger("runtime")


class Runtime(Protocol):
    def cpu_slots(self) -> int: ...
    async def run_unit(self, func: str, payload: dict) -> tuple[Any, float]: ...
    def apply_policy(self, policy: DonationPolicy) -> None: ...
    def reclaim(self) -> None: ...
    def resume(self) -> None: ...
    def shutdown(self) -> None: ...


def _slots_for(cpu_pct: int, cores: int) -> int:
    return max(1, int(math.floor((cpu_pct / 100.0) * cores)))


class ProcessRuntime:
    def __init__(self, cores: int, policy: DonationPolicy, sandbox: bool = False,
                 sandbox_limits: Optional[SandboxLimits] = None) -> None:
        self.cores = max(1, cores)
        self.policy = policy
        self._draining = False
        # Public mesh (Tier 2): run each stranger's unit in a locked-down child
        # process (memory/wall-clock/output caps + input validation) instead of the
        # shared worker pool. LAN/office (Tier 0/1, mutual trust) keeps the fast pool.
        self._sandbox = sandbox
        self._limits = sandbox_limits or SandboxLimits.from_env()
        self._slots = _slots_for(policy.cpu_pct, self.cores)
        self._pool: Optional[ProcessPoolExecutor] = ProcessPoolExecutor(max_workers=self._slots)
        self._running = 0

    def cpu_slots(self) -> int:
        return 0 if self._draining else self._slots

    def apply_policy(self, policy: DonationPolicy) -> None:
        new_slots = _slots_for(policy.cpu_pct, self.cores)
        self.policy = policy
        if new_slots != self._slots and not self._draining:
            log.info("resizing worker pool", old=self._slots, new=new_slots)
            self._slots = new_slots
            self._rebuild_pool()

    def _rebuild_pool(self) -> None:
        self._kill_pool()
        if not self._draining:
            self._pool = ProcessPoolExecutor(max_workers=self._slots)

    def _kill_pool(self) -> None:
        pool, self._pool = self._pool, None
        if pool is None:
            return
        # Hard-terminate child processes so reclaim frees CPU *immediately*
        # (ProcessPoolExecutor.shutdown alone waits for running tasks).
        procs = list(getattr(pool, "_processes", {}).values())
        try:
            pool.shutdown(wait=False, cancel_futures=True)
        except TypeError:
            pool.shutdown(wait=False)
        for p in procs:
            try:
                p.terminate()
            except Exception:  # noqa: BLE001
                pass

    async def run_unit(self, func: str, payload: dict) -> tuple[Any, float]:
        """Run one unit and return ``(result, compute_ms)``.

        ``compute_ms`` is the *actual* function-execution time so the coordinator
        bills only real compute, not MeshPool's sandbox-spawn / scheduling overhead.
        """
        if self._draining or self._pool is None:
            raise RuntimeError("node is draining (resources reclaimed by owner)")
        loop = asyncio.get_running_loop()
        self._running += 1
        try:
            # User-supplied WebAssembly ALWAYS runs in the isolated child process,
            # at every tier. Two reasons, both learned the hard way:
            #  * Trust: `wasm:` code is written by the job's author, not by us. The
            #    "LAN/office nodes trust each other" argument that justifies the
            #    shared pool for curated functions does not extend to arbitrary code
            #    a colleague uploaded.
            #  * Availability: a runaway module aborts its OS process. In the shared
            #    ProcessPoolExecutor that raises BrokenProcessPool and takes the
            #    node OUT OF SERVICE for every other job until it restarts —
            #    reproducible with a 4-line infinite-loop module. A dedicated child
            #    per unit contains the blast radius to that one unit.
            if self._sandbox or is_wasm_function(func):
                # Blocking subprocess management runs off the event loop; the sandbox
                # enforces memory + wall-clock + output caps and validates the input,
                # and reports back the child's true in-process compute time.
                return await loop.run_in_executor(
                    None, functools.partial(run_sandboxed, func, payload,
                                            self._limits, return_timing=True))
            t0 = time.perf_counter()
            result = await loop.run_in_executor(self._pool, run_function, func, payload)
            return result, (time.perf_counter() - t0) * 1000.0
        finally:
            self._running = max(0, self._running - 1)

    def reclaim(self) -> None:
        """Instant kill-switch: stop all donated work now, refuse new work."""
        if self._draining:
            return
        self._draining = True
        self._kill_pool()
        log.warning("resources reclaimed by owner — all donated work stopped")

    def resume(self) -> None:
        if not self._draining:
            return
        self._draining = False
        self._rebuild_pool()
        log.info("donation resumed", slots=self._slots)

    def shutdown(self) -> None:
        self._draining = True
        self._kill_pool()


class DockerRuntime:
    """Stub: same interface, container-per-unit with cgroup/Job-Object quotas and
    an isolated virtual network segment (spec §5). Activated when Docker exists."""

    def __init__(self, cores: int, policy: DonationPolicy) -> None:  # pragma: no cover
        raise NotImplementedError(
            "DockerRuntime requires Docker Engine; use ProcessRuntime in this environment.")


def start_app_process(app: str, port: int, replica_id: str = "r?",
                      node_name: str = "", site_dir: str = "") -> subprocess.Popen:
    """Launch a hosted web-app replica as a child process (hosting engine).

    In production this is a container; here it runs a built-in demo app module so
    the hosting engine (placement, LB, failover) is fully exercisable."""
    env = dict(os.environ)
    env["MESHPOOL_APP"] = app
    env["MESHPOOL_APP_PORT"] = str(port)
    env["MESHPOOL_REPLICA_ID"] = replica_id
    env["MESHPOOL_NODE_NAME"] = node_name
    env["MESHPOOL_SITE_DIR"] = site_dir
    return subprocess.Popen(
        [sys.executable, "-m", "meshpool.hosting.demo_app"],
        env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )


class LogBuffer:
    """Thread-safe, bounded ring buffer of a hosted app's most recent stdout/stderr
    lines, so a deployer can see *why* their app is unhealthy instead of staring at a
    silent process. Bounded so a chatty app can never exhaust the node's memory."""

    def __init__(self, maxlen: int = 500) -> None:
        self._dq: "collections.deque[str]" = collections.deque(maxlen=maxlen)
        self._lock = threading.Lock()

    def append(self, line: str) -> None:
        with self._lock:
            self._dq.append(line)

    def tail(self, n: int = 200) -> list[str]:
        with self._lock:
            items = list(self._dq)
        return items[-n:] if 0 < n < len(items) else items


def _pump_logs(stream, buf: "LogBuffer") -> None:
    """Drain a child's merged stdout/stderr into `buf` line by line until the pipe
    closes (process exit). Runs on a daemon thread so a full OS pipe buffer can never
    block the hosted app."""
    try:
        for raw in iter(stream.readline, ""):
            if raw:
                buf.append(raw.rstrip("\n"))
    except Exception:  # noqa: BLE001
        pass
    finally:
        try:
            stream.close()
        except Exception:  # noqa: BLE001
            pass


def start_proc_app(work_dir: str, start_cmd: str, port: int, replica_id: str = "r?",
                   node_name: str = "", env_extra: Optional[dict] = None,
                   log_buffer: Optional["LogBuffer"] = None) -> subprocess.Popen:
    """Launch a user-supplied **dynamic** web app (its own server process) as a hosting
    replica: run ``start_cmd`` inside the unpacked bundle ``work_dir`` with ``PORT`` set,
    so any framework that honours ``$PORT`` (Flask/FastAPI/Express/Django/...) binds the
    port the load balancer will proxy to. A literal ``{port}`` in the command is also
    substituted, so ``python server.py --port {port}`` works too.

    Security: this executes the deployer's command, so the node agent only ever calls it
    on a **trusted** (non-sandbox / LAN-office) node — it refuses on a public donor."""
    env = dict(os.environ)
    if env_extra:
        # User config first, then the reserved platform vars on top so PORT and the
        # MESHPOOL_* the launcher depends on can never be shadowed by user values.
        env.update({str(k): str(v) for k, v in env_extra.items()})
    env["PORT"] = str(port)
    env["MESHPOOL_APP_PORT"] = str(port)
    env["MESHPOOL_REPLICA_ID"] = replica_id
    env["MESHPOOL_NODE_NAME"] = node_name
    args = shlex.split(start_cmd.replace("{port}", str(port)), posix=(os.name != "nt"))
    if not args:
        raise ValueError("empty start command")
    if log_buffer is not None:
        # Capture the app's merged stdout/stderr so the deployer can see *why* it is
        # unhealthy. A daemon reader thread drains the pipe line by line (a full pipe
        # buffer must never be able to block/deadlock the hosted app).
        proc = subprocess.Popen(
            args, cwd=work_dir, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, encoding="utf-8", errors="replace",
        )
        threading.Thread(target=_pump_logs, args=(proc.stdout, log_buffer),
                         daemon=True).start()
        return proc
    return subprocess.Popen(
        args, cwd=work_dir, env=env,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )


def free_port() -> int:
    """Ask the OS for a free ephemeral port for a new replica/shard server."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("", 0))
    port = s.getsockname()[1]
    s.close()
    return port
