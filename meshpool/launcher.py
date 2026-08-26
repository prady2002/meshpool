"""One-command launcher — the "unzip and run" front door for a shipped MeshPool.

This is the friendly entry point a non-technical user reaches (via ``start.ps1`` /
``start.sh`` / a double-clickable ``.cmd``, or directly ``python -m meshpool.launcher``).
It has three jobs, none of which require the test-only ``sim`` harness so it ships in
the build zip:

* ``solo`` (default) — bring up a **complete MeshPool on this one machine**: a control
  plane + one local agent contributing to it, then open the dashboard. Zero config, no
  IP/token entry, no firewall prompt (everything binds loopback). This is "just enough
  to run and use the application" that the user ships today.
* ``contribute --cloud <url>`` — donate this machine's spare compute to a hosted
  **MeshPool Cloud** coordinator over the outbound reverse tunnel (works behind NAT /
  firewalls; nothing to port-forward). This is the global-mesh path.
* ``self-test`` — bring the solo stack up on an ephemeral port, run one real job end to
  end, tear everything down, and report ``SELFTEST_OK``. Doubles as a user "did my
  install work?" check and as the Phase-3 onboarding acceptance gate.

Only the Python stdlib + ``httpx`` (already a runtime dependency) are used, so the same
launcher behaves identically from a source checkout or an extracted zip.
"""
from __future__ import annotations

import argparse
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import webbrowser
from pathlib import Path
from typing import Optional

import httpx

# Repo/extract root = the directory that contains the ``meshpool`` package, so
# ``python -m meshpool.controlplane.app`` resolves whether run from source or a zip.
PKG_ROOT = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------- #
# process helpers
# --------------------------------------------------------------------------- #
def _base_env() -> dict:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(PKG_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONUNBUFFERED"] = "1"
    return env


def _kill_tree(proc: Optional[subprocess.Popen]) -> None:
    """Terminate a child *and its descendants* (the control plane spawns a worker
    pool; the agent spawns sandbox workers), cross-platform and PID-based only."""
    if proc is None or proc.poll() is not None:
        return
    if sys.platform == "win32":
        subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                       capture_output=True)
    else:
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        except Exception:  # noqa: BLE001
            pass



_client: "Optional[httpx.Client]" = None


def _http() -> "httpx.Client":
    """One reused HTTP client for launcher polling.

    ``httpx.get``/``httpx.post`` construct a fresh client per call, and constructing
    one loads the system TLS trust store: ~1.0 s on Windows, measured. The wait loops
    below poll every 0.3-0.4 s, so that cost did not merely slow start-up -- it ate the
    timeout budget. ``_wait_http(url, timeout=30)`` intends ~100 probes in 30 s and got
    roughly 23, each one spending most of its time in certificate parsing rather than
    waiting for the service. Reusing one client restores the intended poll rate and
    keeps the connection warm between probes.
    """
    global _client
    if _client is None:
        _client = httpx.Client(follow_redirects=True)
    return _client

def _wait_http(url: str, timeout: float = 30.0) -> bool:
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        try:
            r = _http().get(url, timeout=2.5)
            if r.status_code < 500:
                return True
        except Exception as exc:  # noqa: BLE001
            last = exc
        time.sleep(0.3)
    if last:
        print(f"   (last error reaching {url}: {last})")
    return False


def _wait_nodes_online(cp_url: str, n: int = 1, timeout: float = 30.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            nodes = _http().get(f"{cp_url}/api/v1/nodes", timeout=3.0).json().get("nodes", [])
            if len([x for x in nodes if x.get("status") == "online"]) >= n:
                return True
        except Exception:  # noqa: BLE001
            pass
        time.sleep(0.4)
    return False


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _start_control_plane(port: int, data_dir: str, scale: str = "home",
                         host: str = "127.0.0.1") -> subprocess.Popen:
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    env = _base_env()
    env.update({
        "MESHPOOL_HOST": host,                 # bind loopback -> no firewall prompt
        "MESHPOOL_ADVERTISE_HOST": host,
        "MESHPOOL_PORT": str(port),
        "MESHPOOL_CONTROL_PLANE_URL": f"http://{host}:{port}",
        "MESHPOOL_DATA_DIR": data_dir,
        "MESHPOOL_SCALE": scale,
    })
    return subprocess.Popen([sys.executable, "-m", "meshpool.controlplane.app"],
                            env=env, cwd=str(PKG_ROOT))


def _start_local_agent(cp_url: str, data_dir: str, name: str, cpu_pct: int,
                       agent_port: int, host: str = "127.0.0.1") -> subprocess.Popen:
    """A local agent that advertises loopback, so the co-located control plane reaches
    it without any inbound network exposure (no firewall prompt on a solo machine)."""
    env = _base_env()
    env.update({
        "MESHPOOL_CONTROL_PLANE_URL": cp_url,
        "MESHPOOL_AGENT_HOST": host,
        "MESHPOOL_AGENT_BIND": host,
        "MESHPOOL_AGENT_PORT": str(agent_port),
        "MESHPOOL_NAME": name,
        "MESHPOOL_DONATE_CPU_PCT": str(cpu_pct),
        # Its own data dir => its own device identity, distinct from the control
        # plane's, so it is a real separate node in the registry.
        "MESHPOOL_DATA_DIR": data_dir,
    })
    return subprocess.Popen([sys.executable, "-m", "meshpool.agent.daemon"],
                            env=env, cwd=str(PKG_ROOT))


def _start_cloud_agent(cloud_url: str, data_dir: str, name: str,
                       cpu_pct: int) -> subprocess.Popen:
    """Contribute to a hosted coordinator via the outbound reverse tunnel: the agent
    dials out (NAT/firewall-friendly) and advertises an unreachable sentinel, so it is
    reached only through the tunnel."""
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    env = _base_env()
    env.update({
        "MESHPOOL_CONTROL_PLANE_URL": cloud_url,
        "MESHPOOL_TUNNEL": "1",
        "MESHPOOL_NAME": name,
        "MESHPOOL_DONATE_CPU_PCT": str(cpu_pct),
        "MESHPOOL_DATA_DIR": data_dir,
    })
    env.pop("MESHPOOL_AGENT_HOST", None)
    return subprocess.Popen([sys.executable, "-m", "meshpool.agent.daemon"],
                            env=env, cwd=str(PKG_ROOT))


def _default_name() -> str:
    return os.environ.get("MESHPOOL_NAME") or socket.gethostname() or "my-machine"


# --------------------------------------------------------------------------- #
# commands
# --------------------------------------------------------------------------- #
def cmd_solo(args: argparse.Namespace) -> int:
    host = "127.0.0.1"
    port = args.port
    data_dir = args.data_dir or str(Path.home() / ".meshpool")
    cp_url = f"http://{host}:{port}"
    name = args.name or _default_name()

    print("=" * 60)
    print(" MeshPool - starting your personal mesh on this machine")
    print("=" * 60)
    cp = agent = None
    try:
        print(f"-> starting control plane on {cp_url} ...")
        cp = _start_control_plane(port, data_dir, scale="home", host=host)
        if not _wait_http(f"{cp_url}/api/v1/health", timeout=40):
            print("!! control plane did not become healthy. Is the port in use?")
            print(f"   Try a different port:  python -m meshpool.launcher solo --port {port + 1}")
            return 1
        print("   control plane is up.")

        print(f"-> contributing this machine ('{name}', {args.cpu}% CPU) to the pool ...")
        agent = _start_local_agent(cp_url, str(Path(data_dir) / "agent-local"),
                                   name, args.cpu, port + 1, host=host)
        if not _wait_nodes_online(cp_url, 1, timeout=40):
            print("!! the local agent did not come online in time (it may still be starting).")
        else:
            print("   this machine is now in the pool.")

        print()
        print("-" * 60)
        print(f"  MeshPool is ready:   {cp_url}")
        print("  Open that address in your browser to Use or Contribute.")
        print("  Press Ctrl+C here to stop MeshPool.")
        print("-" * 60)
        if not args.no_browser:
            try:
                webbrowser.open(cp_url)
            except Exception:  # noqa: BLE001
                pass

        _idle_until_interrupt([cp, agent])
        return 0
    finally:
        print("\n-> stopping MeshPool ...")
        _kill_tree(agent)
        _kill_tree(cp)
        print("   stopped. Thanks for using MeshPool.")


def cmd_contribute(args: argparse.Namespace) -> int:
    cloud = args.cloud.rstrip("/")
    data_dir = args.data_dir or str(Path.home() / ".meshpool")
    name = args.name or _default_name()

    print("=" * 60)
    print(" MeshPool - contributing your spare compute to the global mesh")
    print("=" * 60)
    print(f"-> coordinator: {cloud}")
    # Friendly reachability preflight (non-fatal: the tunnel client keeps retrying).
    if _wait_http(f"{cloud}/api/v1/health", timeout=8):
        print("   coordinator reachable.")
    else:
        print("   (could not reach the coordinator yet - will keep retrying in the background)")

    agent = None
    try:
        print(f"-> connecting as '{name}', donating {args.cpu}% CPU (outbound tunnel; "
              f"nothing to port-forward) ...")
        agent = _start_cloud_agent(cloud, data_dir, name, args.cpu)
        print()
        print("-" * 60)
        print("  You are now contributing to MeshPool.")
        print(f"  Manage/consume from the coordinator dashboard: {cloud}")
        print("  Press Ctrl+C here to stop contributing.")
        print("-" * 60)
        if args.open_dashboard and not args.no_browser:
            try:
                webbrowser.open(cloud)
            except Exception:  # noqa: BLE001
                pass
        _idle_until_interrupt([agent])
        return 0
    finally:
        print("\n-> disconnecting ...")
        _kill_tree(agent)
        print("   stopped. Thanks for contributing to MeshPool.")


def cmd_self_test(args: argparse.Namespace) -> int:
    """Bring up the solo stack on an ephemeral port, run one real job end to end, and
    tear it all down. Prints SELFTEST_OK / SELFTEST_FAIL and exits 0 / 1 accordingly."""
    host = "127.0.0.1"
    port = args.port or _free_port()
    cp_url = f"http://{host}:{port}"
    data_dir = args.data_dir or tempfile.mkdtemp(prefix="meshpool-selftest-")
    cleanup_dir = args.data_dir is None
    cp = agent = None
    ok = False
    try:
        print(f"[self-test] starting control plane on {cp_url} ...")
        cp = _start_control_plane(port, data_dir, scale="home", host=host)
        if not _wait_http(f"{cp_url}/api/v1/health", timeout=40):
            print("[self-test] FAIL: control plane did not become healthy")
            return 1
        print("[self-test] starting a local agent ...")
        agent = _start_local_agent(cp_url, str(Path(data_dir) / "agent-local"),
                                   "selftest-node", 40, port + 1, host=host)
        if not _wait_nodes_online(cp_url, 1, timeout=40):
            print("[self-test] FAIL: no node came online")
            return 1

        print("[self-test] submitting a batch job (square 0..7) ...")
        r = _http().post(f"{cp_url}/api/v1/jobs", json={
            "name": "selftest", "func": "square",
            "payloads": [{"x": i} for i in range(8)],
            "cpu_pct": 20, "ram_mb": 64}, timeout=15)
        if r.status_code != 200:
            print(f"[self-test] FAIL: job submit returned {r.status_code}: {r.text[:200]}")
            return 1
        job_id = r.json()["job_id"]

        deadline = time.time() + 60
        job = None
        while time.time() < deadline:
            job = _http().get(f"{cp_url}/api/v1/jobs/{job_id}", timeout=10).json()
            if job["status"].startswith("completed"):
                break
            time.sleep(0.4)
        if not job or job["status"] != "completed":
            print(f"[self-test] FAIL: job did not complete (status={job and job['status']})")
            return 1
        for u in job["units"]:
            if u["result"] != u["payload"]["x"] ** 2:
                print(f"[self-test] FAIL: wrong result {u}")
                return 1

        ok = True
        print("[self-test] SELFTEST_OK - control plane, agent, and a real job all work.")
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"[self-test] FAIL: {type(exc).__name__}: {exc}")
        return 1
    finally:
        _kill_tree(agent)
        _kill_tree(cp)
        if cleanup_dir:
            shutil.rmtree(data_dir, ignore_errors=True)
        if not ok:
            print("[self-test] SELFTEST_FAIL")


def _idle_until_interrupt(procs: list) -> None:
    """Block until Ctrl+C, but exit early (with a clear message) if a child dies."""
    try:
        while True:
            for p in procs:
                if p is not None and p.poll() is not None:
                    print(f"\n!! a MeshPool process exited unexpectedly (code {p.returncode}).")
                    return
            time.sleep(1)
    except KeyboardInterrupt:
        pass


# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="meshpool.launcher",
        description="Run MeshPool: use it on your own machine, or contribute to the mesh.")
    sub = ap.add_subparsers(dest="command")

    p_solo = sub.add_parser("solo", help="run a complete MeshPool on this machine (default)")
    p_solo.add_argument("--port", type=int, default=8080, help="dashboard port (default 8080)")
    p_solo.add_argument("--cpu", type=int, default=50, help="%% CPU to contribute (default 50)")
    p_solo.add_argument("--name", default="", help="name for this machine in the pool")
    p_solo.add_argument("--data-dir", default="", help="where to store identity + state")
    p_solo.add_argument("--no-browser", action="store_true", help="do not open the browser")
    p_solo.set_defaults(func=cmd_solo)

    p_con = sub.add_parser("contribute", help="donate compute to a hosted MeshPool Cloud")
    p_con.add_argument("--cloud", required=True, help="coordinator URL, e.g. https://cloud.meshpool.app")
    p_con.add_argument("--cpu", type=int, default=50, help="%% CPU to contribute (default 50)")
    p_con.add_argument("--name", default="", help="name for this machine in the pool")
    p_con.add_argument("--data-dir", default="", help="where to store identity + state")
    p_con.add_argument("--open-dashboard", action="store_true", help="open the coordinator dashboard")
    p_con.add_argument("--no-browser", action="store_true", help="do not open the browser")
    p_con.set_defaults(func=cmd_contribute)

    p_st = sub.add_parser("self-test", help="verify this install works end to end")
    p_st.add_argument("--port", type=int, default=0, help="port (default: pick a free one)")
    p_st.add_argument("--data-dir", default="", help="state dir (default: a temp dir)")
    p_st.set_defaults(func=cmd_self_test)
    return ap


def main(argv: Optional[list] = None) -> int:
    ap = build_parser()
    args = ap.parse_args(argv)
    if not getattr(args, "command", None):
        # No subcommand => the friendliest default: run MeshPool on this machine.
        args = ap.parse_args(["solo", *(argv or [])])
    # Make Ctrl+C behave the same on Windows and POSIX.
    try:
        signal.signal(signal.SIGINT, signal.default_int_handler)
    except Exception:  # noqa: BLE001
        pass
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
