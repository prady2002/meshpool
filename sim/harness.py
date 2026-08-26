"""MeshPool simulation harness (spec §9 acceptance testing).

Because this environment has no Docker, the harness launches the control plane and
N node-agents as separate OS processes on one machine, pinning heterogeneous fake
capabilities (cores/RAM/FLOPS) onto each so the schedulers and topology-aware
partitioner behave as they would across real machines. Kill an agent to exercise
failover/rescheduling.

Usage:
  python -m sim.harness --agents 3                 # run interactively
  (or import Harness in tests)
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import httpx

if TYPE_CHECKING:
    from sim.topology import Topology

REPO_ROOT = Path(__file__).resolve().parent.parent

# How long a freshly-spawned control plane or agent gets to answer /health.
#
# This was 25 s, which sounds generous until you count what happens first: the
# agent runs a full hardware probe *before* binding its port, and that probe used
# to take 10 s idle and 22 s under load. Three agents starting at once on a busy
# machine blew straight through the budget, and the resulting
# "service not ready at 127.0.0.1:8126" reads like the agent crashed rather than
# like it was still booting -- so it got retried as flakiness for a long time.
#
# The probe is now parallel (~3 s), but the budget stays wide deliberately. The
# two failure modes are not symmetric: timing out a *healthy* service produces a
# red suite and a wild goose chase, while waiting longer on a genuinely dead one
# only delays a failure that was going to happen anyway. Buy the margin.
_READY_TIMEOUT = float(os.environ.get("MESHPOOL_SIM_READY_TIMEOUT", "75"))


def _free_port(preferred: int) -> int:
    """Return ``preferred`` if it is bindable, otherwise a free port.

    Gate scenarios hard-code their ports, and a 2-agent cluster silently claims
    three of them (``cp_port`` plus ``cp_port + 1 + idx`` per agent), so the
    ranges overlap in several places even though every declared control-plane
    port is distinct. The gates run serially, so an overlap only bites when the
    previous gate's sockets have not been released yet -- and then it surfaces
    as ``TimeoutError: service not ready at 127.0.0.1:<port>`` 25 seconds later,
    which reads like the agent failed to boot rather than like a port clash.

    Treating the hard-coded port as a *hint* removes the whole failure class
    without renumbering every scenario. When the port is free -- the normal
    case -- behaviour is exactly as before, so the documented ports still hold.

    ``preferred=0`` means "just give me anything". It cannot be handled by the
    bind-and-keep path, because binding port 0 always succeeds and would hand
    back a literal 0 that then gets written into ``MESHPOOL_PORT`` -- the server
    would pick its own ephemeral port and the harness would spend 25 seconds
    health-checking a port nothing is listening on.
    """
    import socket
    if preferred and preferred > 0:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", preferred))
                return preferred
            except OSError:
                pass
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        chosen = s.getsockname()[1]
    if preferred and preferred > 0:
        print(f"[harness] port {preferred} is busy; using {chosen} instead")
    return chosen


# heterogeneous node profiles cycled across agents (cores, ram_mb, flops, vram_mb)
PROFILES = [
    {"cores": 8, "ram": 16000, "flops": 60.0, "vram": 0},
    {"cores": 4, "ram": 8000, "flops": 30.0, "vram": 0},
    {"cores": 2, "ram": 4000, "flops": 15.0, "vram": 0},
    {"cores": 6, "ram": 12000, "flops": 45.0, "vram": 0},
    {"cores": 2, "ram": 3000, "flops": 12.0, "vram": 0},
]


def _kill_tree(proc: Optional[subprocess.Popen]) -> None:
    if proc is None or proc.poll() is not None:
        return
    if sys.platform == "win32":
        # PID-based tree kill (allowed; not name-based). Frees pool child procs too.
        subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                       capture_output=True)
    else:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


def _kill_now(proc: Optional[subprocess.Popen]) -> None:
    """Stop a process *immediately*, then clean up its children in the background.

    Measured on this machine: ``taskkill /T /F`` takes ~367 ms to return while
    ``Popen.kill`` (TerminateProcess) takes ~0.3 ms -- a 1200x difference. That gap
    is not academic. A gate that kills a node "mid-stream" and then waits a third
    of a second for the killer to return can easily have the whole workload finish
    inside the kill, producing a PASS that proves the failover path was never
    touched. Any gate timing a failure against live traffic must use this.

    The tree sweep still happens, just off the critical path, so pool children are
    not leaked.
    """
    if proc is None or proc.poll() is not None:
        return
    proc.kill()
    if sys.platform == "win32":
        threading.Thread(
            target=lambda: subprocess.run(
                ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                capture_output=True),
            daemon=True).start()


class Harness:
    def __init__(self, n_agents: int = 3, host: str = "127.0.0.1", cp_port: int = 8080,
                 data_dir: Optional[str] = None, scale: str = "home",
                 tunnel: bool = False, sandbox: bool = False,
                 liar_indices: Optional[set] = None,
                 stall_ms: Optional[dict] = None,
                 topology: Optional["Topology"] = None,
                 fresh: bool = True) -> None:
        self.n_agents = n_agents
        self.host = host
        self.cp_port = cp_port
        self.scale = scale
        # Global mode: agents reach the coordinator only via an outbound reverse
        # tunnel and advertise an unreachable sentinel URL, so any successful
        # control-plane → agent operation proves it travelled through the tunnel.
        self.tunnel = tunnel
        # Public-mesh options used by the Tier-2 acceptance scenarios: run units in
        # the locked-down sandbox, and (for the trust gate) mark specific agent
        # indices as dishonest so the coordinator's spot-check must catch them.
        self.sandbox = sandbox
        self.liar_indices = set(liar_indices or ())
        # agent index -> milliseconds of delay per shard hop. Restart-safe: an agent
        # restarted through start_agent picks the delay up again, and set_stall()
        # exists so a gate can make a node slow *after* it is already serving.
        self.stall_ms: dict = dict(stall_ms or {})
        # Optional network topology for emulated WAN conditions.  When supplied,
        # callers can retrieve a NetEmTransport from netem_transport().
        self.topology: Optional["Topology"] = topology
        self.data_dir = data_dir or str(Path(os.environ.get("TEMP", "/tmp")) / "meshpool-sim")
        # Scenarios pin a fixed data_dir so a run is inspectable afterwards. That makes
        # state *persist between runs*, which silently invalidates gates: a node that
        # still has last run's weight blobs on disk never exercises the distribution
        # path, so the gate passes once and then measures nothing. Wipe by default;
        # pass fresh=False only when a scenario deliberately inherits prior state.
        if fresh:
            shutil.rmtree(self.data_dir, ignore_errors=True)
        self.cp_proc: Optional[subprocess.Popen] = None
        self.agent_procs: list[Optional[subprocess.Popen]] = []
        self.agent_ports: list[int] = []
        self._cp_port_resolved = False

    def cp_url(self) -> str:
        return f"http://{self.host}:{self.cp_port}"

    # --- lifecycle ------------------------------------------------------- #
    def _base_env(self) -> dict:
        env = dict(os.environ)
        env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
        env["PYTHONUNBUFFERED"] = "1"
        # common/faultinject.py refuses to activate any deliberate misbehaviour unless
        # the process is explicitly opted in. Scenarios that need a dishonest node
        # (verify_trust_credits) set MESHPOOL_LIE on one agent, but that flag is inert
        # on its own -- without this the liar behaves honestly and the Tier-2
        # spot-check gate silently proves nothing.
        env["MESHPOOL_SIM_HARNESS"] = "1"
        return env

    def start_control_plane(self) -> None:
        Path(self.data_dir).mkdir(parents=True, exist_ok=True)
        # Resolve the port once, on first boot only. A restart (verify_chaos kills
        # the coordinator and brings it back) must reuse the very same port, or
        # the already-running agents would keep dialling the old URL and the
        # self-heal this harness exists to test could never succeed.
        if not self._cp_port_resolved:
            self.cp_port = _free_port(self.cp_port)
            self._cp_port_resolved = True
        env = self._base_env()
        env.update({
            "MESHPOOL_HOST": self.host,
            "MESHPOOL_ADVERTISE_HOST": self.host,
            "MESHPOOL_PORT": str(self.cp_port),
            "MESHPOOL_CONTROL_PLANE_URL": self.cp_url(),
            "MESHPOOL_DATA_DIR": self.data_dir,
            "MESHPOOL_SCALE": self.scale,
        })
        self.cp_proc = subprocess.Popen(
            [sys.executable, "-m", "meshpool.controlplane.app"],
            env=env, cwd=str(REPO_ROOT))
        self._wait_http(f"{self.cp_url()}/api/v1/health", timeout=_READY_TIMEOUT)

    def start_agent(self, idx: int) -> None:
        if idx < len(self.agent_ports):
            # Restarting an existing agent: keep the port it already advertised.
            port = self.agent_ports[idx]
        else:
            port = _free_port(self.cp_port + 1 + idx)
        prof = PROFILES[idx % len(PROFILES)]
        env = self._base_env()
        env.update({
            "MESHPOOL_CONTROL_PLANE_URL": self.cp_url(),
            "MESHPOOL_AGENT_HOST": self.host,
            "MESHPOOL_AGENT_BIND": self.host,
            "MESHPOOL_AGENT_PORT": str(port),
            "MESHPOOL_NAME": f"sim-node-{idx + 1}",
            "MESHPOOL_SIM_CORES": str(prof["cores"]),
            "MESHPOOL_SIM_RAM_MB": str(prof["ram"]),
            "MESHPOOL_SIM_FLOPS": str(prof["flops"]),
            "MESHPOOL_SIM_VRAM_MB": str(prof["vram"]),
            "MESHPOOL_DONATE_CPU_PCT": os.environ.get("MESHPOOL_DONATE_CPU_PCT", "60"),
            "MESHPOOL_DONATE_RAM_MB": str(prof["ram"] // 2),
        })
        # Each simulated agent models a *separate machine*, so it gets its own data
        # dir and therefore its own persistent device identity. Without this,
        # co-located sim agents would share one ~/.meshpool identity and collapse
        # onto a single device_id (and thus a single node) in the registry.
        env["MESHPOOL_DATA_DIR"] = str(Path(self.data_dir) / f"agent-{idx}")
        if self.tunnel:
            # Global mode: dial out only; the daemon advertises the unreachable
            # sentinel URL itself, so don't pin a reachable agent host.
            env["MESHPOOL_TUNNEL"] = "1"
            env.pop("MESHPOOL_AGENT_HOST", None)
        if self.sandbox:
            env["MESHPOOL_SANDBOX"] = "1"
        if idx in self.liar_indices:
            # Test-only: this agent returns wrong results; the Tier-2 spot-check must
            # catch it, reject its work, and keep it from being paid.
            env["MESHPOOL_LIE"] = "1"
        if idx in self.stall_ms:
            # Test-only: this agent stays perfectly correct and healthy but answers
            # slowly, which is the shape of thermal throttling or a busy machine.
            # Killing a node cannot emulate it, and a straggler is the failure that
            # actually caps a pipeline's throughput.
            env["MESHPOOL_STALL_MS"] = str(self.stall_ms[idx])
        proc = subprocess.Popen(
            [sys.executable, "-m", "meshpool.agent.daemon"], env=env, cwd=str(REPO_ROOT))
        if idx < len(self.agent_procs):
            self.agent_procs[idx] = proc
        else:
            self.agent_procs.append(proc)
            self.agent_ports.append(port)
        self._wait_http(f"http://{self.host}:{port}/health", timeout=_READY_TIMEOUT)

    def start(self) -> None:
        self.start_control_plane()
        for i in range(self.n_agents):
            self.start_agent(i)
        self.wait_nodes_online(self.n_agents)

    def set_stall(self, idx: int, ms: Optional[float]) -> None:
        """Make agent ``idx`` answer slowly, or clear the delay with ``None``.

        Raises if the agent refuses, because a scenario that believes it created a
        straggler and did not would report a passing straggler test with no straggler
        in it -- the most flattering possible failure.
        """
        url = f"http://{self.host}:{self.agent_ports[idx]}/fault/stall"
        r = httpx.post(url, json={"ms": ms}, timeout=10)
        r.raise_for_status()
        if ms is None:
            self.stall_ms.pop(idx, None)
        else:
            self.stall_ms[idx] = ms

    def set_corrupt_blobs(self, idx: int, on: Optional[bool]) -> None:
        """Make agent ``idx`` serve wrong bytes for content-addressed blobs.

        Raises on refusal for the same reason set_stall does: an integrity test with
        no dishonest peer in it passes trivially and proves nothing.
        """
        url = f"http://{self.host}:{self.agent_ports[idx]}/fault/corrupt_blobs"
        r = httpx.post(url, json={"on": on}, timeout=10)
        r.raise_for_status()

    def stop_agent(self, idx: int, fast: bool = False) -> None:
        """Stop agent ``idx``. ``fast=True`` returns in microseconds (see _kill_now)."""
        (_kill_now if fast else _kill_tree)(self.agent_procs[idx])
        self.agent_procs[idx] = None

    def restart_agent(self, idx: int) -> None:
        self.start_agent(idx)

    def stop(self) -> None:
        for p in self.agent_procs:
            _kill_tree(p)
        _kill_tree(self.cp_proc)

    def netem_transport(self, inner: object, src_node_id: str,
                        time_scale: float = 1.0) -> object:
        """Return a :class:`~sim.netem.NetEmTransport` wrapping *inner* under the
        harness topology.  Returns *inner* unchanged if no topology was supplied,
        preserving backwards compatibility for all existing scenarios.
        """
        if self.topology is None:
            return inner
        from sim.netem import NetEm, NetEmTransport
        netem = NetEm(self.topology, time_scale=time_scale)
        return NetEmTransport(inner, netem, src_node_id)

    # --- helpers --------------------------------------------------------- #
    @staticmethod
    def _wait_http(url: str, timeout: float = 20.0) -> None:
        deadline = time.time() + timeout
        last = None
        while time.time() < deadline:
            try:
                r = httpx.get(url, timeout=2.0)
                if r.status_code < 500:
                    return
            except Exception as exc:  # noqa: BLE001
                last = exc
            time.sleep(0.3)
        raise TimeoutError(f"service not ready at {url}: {last}")

    def wait_nodes_online(self, n: int, timeout: float = 25.0) -> list[dict]:
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                r = httpx.get(f"{self.cp_url()}/api/v1/nodes", timeout=3.0)
                nodes = r.json().get("nodes", [])
                online = [x for x in nodes if x["status"] == "online"]
                if len(online) >= n:
                    return online
            except Exception:  # noqa: BLE001
                pass
            time.sleep(0.4)
        raise TimeoutError(f"only see fewer than {n} online nodes after {timeout}s")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--agents", type=int, default=3)
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--scale", default="home")
    ap.add_argument("--tunnel", action="store_true",
                    help="global mode: agents connect via outbound reverse tunnel only")
    args = ap.parse_args()
    h = Harness(n_agents=args.agents, cp_port=args.port, scale=args.scale, tunnel=args.tunnel)
    print(f"Starting MeshPool: control plane + {args.agents} agents "
          f"({'tunnel' if args.tunnel else 'lan'} mode) ...")
    h.start()
    print(f"Dashboard:  {h.cp_url()}")
    print(f"Nodes:      {h.cp_url()}/api/v1/nodes")
    print("Ctrl-C to stop.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("stopping ...")
    finally:
        h.stop()


if __name__ == "__main__":
    main()
