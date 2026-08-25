"""Real multi-node acceptance over DISTINCT network addresses + REAL hardware probes.

Every other scenario runs through the sim harness, which pins the control plane and
all agents to ``127.0.0.1``. That proves the engines, but it can NOT prove the one
thing that makes real laptops work: that the control plane reaches each node at the
address the node *advertised*, rather than a hardcoded localhost.

This scenario closes that gap. It launches:
  * a control plane that binds and advertises a non-default address,
  * agents that each advertise their OWN distinct address,
with NO capability overrides — each agent runs a real hardware probe — and then
checks that batch dispatch, hosted-app proxying, and inference activation hops all
reach every node at its advertised address.

By default it uses the ``127.0.0.0/8`` loopback block (``127.0.0.2`` control plane,
``127.0.0.3`` / ``127.0.0.4`` agents) so it needs no firewall changes and stays
deterministic in CI. Point it at a real LAN by exporting ``E2E_CP_HOST`` and
``E2E_AGENT_HOSTS`` (comma-separated) with routable IPs once the control-plane port
is allowed through each host firewall (``deploy/join.ps1 -OpenFirewall`` /
``deploy/join.sh --open-firewall``).

macOS only brings ``127.0.0.1`` up by default; if the extra loopback aliases can't
be bound the scenario prints ``SKIP`` and exits 0 (Linux and Windows support the
whole block out of the box).
"""
from __future__ import annotations

import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx

REPO = Path(__file__).resolve().parents[2]
CP_HOST = os.environ.get("E2E_CP_HOST", "127.0.0.2")
AGENT_HOSTS = os.environ.get("E2E_AGENT_HOSTS", "127.0.0.3,127.0.0.4").split(",")
CP_PORT = 8090
AGENT_PORT0 = 9101
CP = f"http://{CP_HOST}:{CP_PORT}"
DATA = str(REPO / "_data_lan")
procs: list[subprocess.Popen] = []
fails: list[str] = []


def _env(**extra: str) -> dict:
    e = {k: v for k, v in os.environ.items() if not k.startswith("MESHPOOL_")}
    e["PYTHONPATH"] = str(REPO) + os.pathsep + e.get("PYTHONPATH", "")
    e["PYTHONUNBUFFERED"] = "1"
    e.update(extra)
    return e


def _can_bind(host: str) -> bool:
    """Can we bind a socket to this address on this machine? (loopback aliases)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind((host, 0))
        return True
    except OSError:
        return False
    finally:
        s.close()


def _wait(url: str, timeout: float = 30.0) -> bool:
    end = time.time() + timeout
    while time.time() < end:
        try:
            if httpx.get(url, timeout=2.0).status_code < 500:
                return True
        except Exception:  # noqa: BLE001
            pass
        time.sleep(0.3)
    return False


def _safe_get(url: str) -> int:
    try:
        return httpx.get(url, timeout=8).status_code
    except Exception:  # noqa: BLE001
        return 0


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""), flush=True)
    if not ok:
        fails.append(name)


def _kill(p: subprocess.Popen) -> None:
    try:
        if sys.platform == "win32":
            subprocess.run(["taskkill", "/PID", str(p.pid), "/T", "/F"], capture_output=True)
        else:
            p.terminate()
    except Exception:  # noqa: BLE001
        pass


def main() -> int:
    print(f"== Real multi-node E2E: CP {CP_HOST}, agents {AGENT_HOSTS} (real hardware probe) ==", flush=True)

    # Portability guard: distinct loopback aliases aren't up on every OS (macOS).
    unbindable = [h for h in [CP_HOST, *AGENT_HOSTS] if not _can_bind(h)]
    if unbindable:
        print(f"  [SKIP] cannot bind {unbindable} on this OS — set E2E_CP_HOST / "
              f"E2E_AGENT_HOSTS to bindable addresses to run this proof.", flush=True)
        return 0

    Path(DATA).mkdir(parents=True, exist_ok=True)

    cp = subprocess.Popen([sys.executable, "-m", "meshpool.controlplane.app"],
                          env=_env(MESHPOOL_HOST=CP_HOST, MESHPOOL_ADVERTISE_HOST=CP_HOST,
                                   MESHPOOL_PORT=str(CP_PORT), MESHPOOL_DATA_DIR=DATA,
                                   MESHPOOL_SCALE="home"), cwd=str(REPO))
    procs.append(cp)
    if not _wait(f"{CP}/api/v1/health", 30):
        check("control plane reachable at advertised address", False, f"{CP} not responding")
        return 1
    check("control plane reachable at advertised address", True, CP)

    ob = httpx.get(f"{CP}/api/v1/onboarding", timeout=5).json()
    check("join command embeds the advertised address (not loopback default)",
          CP_HOST in ob["join_command"] and ob["control_plane_url"] == CP,
          ob["control_plane_url"])

    for i, host in enumerate(AGENT_HOSTS):
        port = AGENT_PORT0 + i
        p = subprocess.Popen([sys.executable, "-m", "meshpool.agent.daemon"],
                             env=_env(MESHPOOL_CONTROL_PLANE_URL=CP, MESHPOOL_AGENT_HOST=host,
                                      MESHPOOL_AGENT_BIND=host, MESHPOOL_AGENT_PORT=str(port),
                                      MESHPOOL_NAME=f"lan-node-{i + 1}",
                                      MESHPOOL_DATA_DIR=str(Path(DATA) / f"agent-{i}"),
                                      MESHPOOL_DONATE_CPU_PCT="60"), cwd=str(REPO))
        procs.append(p)
        _wait(f"http://{host}:{port}/health", 30)

    online = []
    end = time.time() + 30
    while time.time() < end:
        online = [n for n in httpx.get(f"{CP}/api/v1/nodes", timeout=5).json()["nodes"]
                  if n["status"] == "online"]
        if len(online) >= len(AGENT_HOSTS):
            break
        time.sleep(0.4)
    check(f"{len(AGENT_HOSTS)} real agents online", len(online) >= len(AGENT_HOSTS),
          ", ".join(f"{n['name']}={n['capability']['cpu_cores']}c/{n['capability']['ram_mb']}MB/"
                    f"{n['capability']['flops_estimate']}GF" for n in online))
    check("each node advertises its OWN distinct address",
          {n["url"].split("//")[1].split(":")[0] for n in online} == set(AGENT_HOSTS[:len(online)]),
          ", ".join(n["url"] for n in online))

    # --- BATCH: control plane dispatches to nodes at their advertised address -
    r = httpx.post(f"{CP}/api/v1/jobs", json={"name": "square", "func": "square",
                   "payloads": [{"x": i} for i in range(12)], "cpu_pct": 10, "ram_mb": 64}, timeout=10)
    job_id = r.json()["job_id"]
    done, nodes_used = False, set()
    end = time.time() + 40
    while time.time() < end:
        j = httpx.get(f"{CP}/api/v1/jobs/{job_id}", timeout=5).json()
        units = j.get("units", [])
        for u in units:
            if u.get("assigned_node"):
                nodes_used.add(u["assigned_node"])
        if units and sum(1 for u in units if u.get("status") == "done") == len(units):
            done = True
            break
        time.sleep(0.5)
    check("batch job distributed + completed", done, f"12 units across {len(nodes_used)} node(s)")

    # --- INFERENCE: activation hops to shards at their advertised address -----
    _r = httpx.post(f"{CP}/api/v1/models", json={"model": "nano-gpt"}, timeout=60)
    dep = _r.json()
    check("model sharded across the pool", bool(dep.get("stages")),
          f"{len(dep.get('stages', []))} stage(s): " +
          ", ".join(f"{s.get('node_name')}[{s.get('first')}-{s.get('last')}]"
                    for s in dep.get("stages", []))
          + (f" | HTTP {_r.status_code} body={_r.text[:400]}"
             if not dep.get("stages") else ""))
    chat = httpx.post(f"{CP}/api/v1/inference/chat", json={"prompt": "MeshPool lets you",
                      "max_new_tokens": 16}, timeout=60).json()
    check("sharded model answered (activation hops over the network)",
          chat.get("tokens", 0) > 0 and chat.get("tokens_per_sec", 0) > 0,
          f"{chat.get('tokens')} tok @ {chat.get('tokens_per_sec', 0):.1f} tok/s, "
          f"{len(chat.get('stages', []))} stage(s)")

    # --- HOSTING: proxy to replicas at their advertised address --------------
    d = httpx.post(f"{CP}/api/v1/deployments", json={"name": "hello-app", "app": "hello",
                   "replicas": 2}, timeout=10).json()
    dep_id = d.get("id") or d.get("deployment_id")
    time.sleep(4)
    got = httpx.get(f"{CP}/app/{dep_id}/", timeout=10)
    check("hosted app served through the load balancer", got.status_code == 200 and len(got.content) > 0,
          f"HTTP {got.status_code}, served-by {got.headers.get('X-MeshPool-Served-By')}")

    # --- FAILOVER (destructive, runs last): kill a node, app stays up --------
    _kill(procs[1])   # kill first agent
    time.sleep(1)
    survived = 0
    for _ in range(10):
        if _safe_get(f"{CP}/app/{dep_id}/") == 200:
            survived += 1
        time.sleep(0.4)
    check("hosting survived a node dying (automatic failover)", survived >= 8, f"{survived}/10 ok after kill")

    print(f"\n== {'ALL PASS' if not fails else 'FAILURES: ' + ', '.join(fails)} ==", flush=True)
    return 0 if not fails else 1


if __name__ == "__main__":
    rc = 1
    try:
        rc = main()
    finally:
        for p in procs:
            _kill(p)
        shutil.rmtree(DATA, ignore_errors=True)
    sys.exit(rc)
