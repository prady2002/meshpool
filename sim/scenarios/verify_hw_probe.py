"""Acceptance gate: real hardware probes register with a real control plane.

Runs on the one available Windows host.  macOS/Linux command parsing is covered by
unit fixtures; this gate is deliberately honest about what it actually executes.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from meshpool.agent import probe as py_probe  # noqa: E402
from sim.harness import Harness, _kill_tree  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
GO_DIR = REPO_ROOT / "agent-go"
CP_PORT = 8360
PY_PORT = 8361
GO_PORT = 8362


def _wait_http(url: str, timeout: float = 75.0) -> None:
    """Wait for a freshly-spawned agent to answer /health.

    This was 30s, which the Go agent missed: meshd runs its full hardware probe
    *before* binding its port, and that probe used to shell out to seven
    PowerShell CIM queries in sequence for ~45s total. The gate then failed with
    "service not ready at 127.0.0.1:8362" -- an error that reads like a crash
    rather than like a node still booting.

    The probe is concurrent now (~17s in CI, ~3s warm), but the budget stays
    wide on purpose: timing out a healthy agent costs a red suite and a wrong
    diagnosis, while waiting longer on a dead one only delays the inevitable.
    """
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        try:
            if httpx.get(url, timeout=2.0).status_code < 500:
                return
        except Exception as exc:  # noqa: BLE001
            last = exc
        time.sleep(0.3)
    raise TimeoutError(f"service not ready at {url}: {last}")


def _wait_node(cp: str, name: str, timeout: float = 75.0) -> dict:
    deadline = time.time() + timeout
    seen = None
    while time.time() < deadline:
        nodes = httpx.get(f"{cp}/api/v1/nodes", timeout=5).json().get("nodes", [])
        seen = [n.get("name") for n in nodes]
        for node in nodes:
            if node.get("name") == name and node.get("status") == "online":
                return node
        time.sleep(0.4)
    raise AssertionError(f"{name} did not register online; saw {seen}")


def _require_measured(name: str, cap: dict, link: dict) -> None:
    assert cap["cpu_cores"] >= 1, f"{name} cpu_cores not measured: {cap}"
    assert cap["ram_mb"] >= 8192, f"{name} ram_mb implausible/not measured: {cap}"
    assert cap["disk_mb"] > 0, f"{name} disk_mb not measured: {cap}"
    assert cap["flops_estimate"] > 0, f"{name} flops_estimate not measured: {cap}"
    assert link["latency_ms"] >= 0, f"{name} latency missing: {link}"
    assert link["bandwidth_mbps"] > 0, (
        f"{name} bandwidth not measured against real control-plane probe endpoint: {link}"
    )


def main() -> int:
    go = shutil.which("go")
    if go is None:
        print("FAILED: Go toolchain is required for the hardware-probe gate")
        return 1

    data_root = Path(__file__).resolve().parent / "_data_hw_probe"
    shutil.rmtree(data_root, ignore_errors=True)
    data_root.mkdir(parents=True, exist_ok=True)

    # Non-vacuity guards for parser-only platform paths: these fixtures encode
    # defects that would produce plausible but wrong placement inputs.
    assert py_probe._parse_proc_meminfo("MemTotal:       32800664 kB\n") == 32031
    assert py_probe._parse_windows_video_json(
        '[{"Name":"Intel(R) Iris(R) Xe Graphics","AdapterRAM":536870912}]'
    )[1] == 0

    h = Harness(n_agents=0, cp_port=CP_PORT, data_dir=str(data_root / "cluster"))
    py_proc = None
    go_proc = None
    ok = True
    try:
        print("== building meshd ==")
        exe = data_root / ("meshd.exe" if os.name == "nt" else "meshd")
        build = subprocess.run([go, "build", "-o", str(exe), "./cmd/meshd"],
                               cwd=str(GO_DIR), capture_output=True, text=True)
        print(f"  go build exit={build.returncode}")
        if build.returncode != 0:
            print(build.stdout[-2000:])
            print(build.stderr[-2000:])
            return 1

        print("\n== starting real control plane on reserved port 8360 ==")
        h.start_control_plane()
        cp = h.cp_url()

        base_env = dict(os.environ)
        for key in list(base_env):
            if key.startswith("MESHPOOL_SIM_"):
                base_env.pop(key)
        base_env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + base_env.get("PYTHONPATH", "")
        base_env["PYTHONUNBUFFERED"] = "1"
        base_env["MESHPOOL_CONTROL_PLANE_URL"] = cp
        base_env["MESHPOOL_AGENT_HOST"] = "127.0.0.1"
        base_env["MESHPOOL_AGENT_BIND"] = "127.0.0.1"

        print("\n== starting Python agent with real hardware probe ==")
        py_env = dict(base_env)
        py_env.update({
            "MESHPOOL_AGENT_PORT": str(PY_PORT),
            "MESHPOOL_NAME": "hw-probe-python",
            "MESHPOOL_DATA_DIR": str(data_root / "python-agent"),
        })
        py_proc = subprocess.Popen([sys.executable, "-m", "meshpool.agent.daemon"],
                                   env=py_env, cwd=str(REPO_ROOT))
        _wait_http(f"http://127.0.0.1:{PY_PORT}/health")
        py_node = _wait_node(cp, "hw-probe-python")
        py_health = httpx.get(f"http://127.0.0.1:{PY_PORT}/health", timeout=5).json()
        _require_measured("python", py_node["capability"], py_node["link"])

        print("\n== starting Go agent with real hardware probe ==")
        go_env = dict(base_env)
        go_env.update({
            "MESHPOOL_AGENT_PORT": str(GO_PORT),
            "MESHPOOL_NAME": "hw-probe-go",
            "MESHPOOL_DATA_DIR": str(data_root / "go-agent"),
        })
        go_proc = subprocess.Popen([str(exe)], env=go_env, cwd=str(REPO_ROOT))
        _wait_http(f"http://127.0.0.1:{GO_PORT}/health")
        go_node = _wait_node(cp, "hw-probe-go")
        go_health = httpx.get(f"http://127.0.0.1:{GO_PORT}/health", timeout=5).json()
        _require_measured("go", go_node["capability"], go_node["link"])

        py_ram = py_node["capability"]["ram_mb"]
        go_ram = go_node["capability"]["ram_mb"]
        ratio = min(py_ram, go_ram) / max(py_ram, go_ram)
        assert ratio > 0.90, f"Python/Go RAM disagree: python={py_ram} go={go_ram}"

        # Non-vacuity against previous Go hardcoded-zero defects and Python's
        # historical guessed fallback: the real control plane stored the measured
        # values it will place on, not just agent-local JSON.
        assert py_node["capability"]["ram_mb"] != 4096, "Python RAM looks like old fallback guess"
        assert go_node["capability"]["ram_mb"] != 0, "Go RAM regressed to the old hardcoded zero"
        assert go_node["link"]["bandwidth_mbps"] != 0, "Go link regressed to the old hardcoded zero"
        if "radeon" in go_health["capability"].get("cpu_model", "").lower():
            assert py_node["capability"]["vram_mb"] == 0, (
                "Python treated this integrated Radeon GPU's shared memory as dedicated VRAM")
            assert go_node["capability"]["vram_mb"] == 0, (
                "Go treated this integrated Radeon GPU's shared memory as dedicated VRAM")

        table = {
            "python_cp": {"capability": py_node["capability"], "link": py_node["link"]},
            "python_health": py_health["capability"],
            "go_cp": {"capability": go_node["capability"], "link": go_node["link"]},
            "go_health": go_health["capability"],
        }
        print("\n== measured values ==")
        print(json.dumps(table, indent=2, sort_keys=True))
        print("\nPASS: real control plane accepted measured Python and Go hardware probes")
    except AssertionError as exc:
        print(f"\nFAILED: {exc}")
        ok = False
    except Exception as exc:  # noqa: BLE001
        print(f"\nFAILED unexpected: {type(exc).__name__}: {exc}")
        ok = False
    finally:
        _kill_tree(py_proc)
        _kill_tree(go_proc)
        h.stop()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
