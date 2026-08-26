"""Acceptance gate: QUIC failure transparently falls back to the HTTP tunnel.

Honest scope
------------
Measured with local OS processes. The Python agent intentionally has no QUIC
client, so a coordinator configured for QUIC must use the existing outbound HTTP
reverse tunnel. This emulates UDP-blocked networks and proves fallback is active
and non-vacuous; it is not a real-internet NAT success-rate measurement.

Exit 0 = PASS.  Exit 1 = FAIL.
"""
from __future__ import annotations

import os
import socket
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from meshpool.common.netutil import TUNNEL_SENTINEL_URL  # noqa: E402
from sim.harness import Harness  # noqa: E402


def _free_udp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _node(cp: str) -> dict:
    nodes = httpx.get(f"{cp}/api/v1/nodes", timeout=5).json()["nodes"]
    assert nodes, "no nodes registered"
    return nodes[0]


def main() -> int:
    old_transport = os.environ.get("MESHPOOL_TRANSPORT")
    old_quic = os.environ.get("MESHPOOL_QUIC_PORT")
    os.environ["MESHPOOL_TRANSPORT"] = "quic"
    os.environ["MESHPOOL_QUIC_PORT"] = str(_free_udp_port())

    data_root = Path(__file__).resolve().parent / "_data_transport_fallback"
    h = Harness(n_agents=1, cp_port=8428, data_dir=str(data_root), tunnel=True)
    ok = True
    try:
        print("== test 1: coordinator is configured for QUIC but node is tunnel-only ==")
        h.start()
        cp = h.cp_url()
        node = _node(cp)
        assert node["url"] == TUNNEL_SENTINEL_URL, (
            f"non-vacuity: node advertised {node['url']!r}; expected tunnel sentinel")
        end = time.time() + 45
        while time.time() < end:
            node = _node(cp)
            if node.get("reachable_via") == "tunnel":
                break
            time.sleep(0.5)
        assert node.get("reachable_via") == "tunnel", (
            f"fallback did not engage; reachable_via={node.get('reachable_via')!r}")
        print("  PASS: QUIC is unavailable and the console reports reachable_via=tunnel")

        print("\n== test 2: fallback tunnel carries control work with no lost request ==")
        body = {"cpu_pct": 37, "ram_mb": 1024, "gpu_allowed": False,
                "schedule": "always", "idle_minutes": 10}
        r = httpx.post(f"{cp}/api/v1/nodes/{node['id']}/policy", json=body, timeout=20)
        assert r.status_code == 200, f"policy call failed: {r.status_code} {r.text[:200]}"
        got = r.json()["policy"]
        assert got["cpu_pct"] == 37, (
            f"policy response did not round-trip over fallback tunnel: {got}")
        node2 = _node(cp)
        assert node2["policy"]["cpu_pct"] == 37, (
            "registry was not updated by the fallback result; request may have been lost")
        assert node2.get("reachable_via") == "tunnel", (
            "non-vacuity: work succeeded only after path changed away from tunnel")
        print("  PASS: the fallback request completed and updated node policy")

        print("\nALL TRANSPORT FALLBACK CHECKS PASSED")
    except AssertionError as exc:
        print(f"\nFAILED: {exc}")
        ok = False
    except Exception as exc:  # noqa: BLE001
        print(f"\nFAILED: {type(exc).__name__}: {exc}")
        ok = False
    finally:
        h.stop()
        if old_transport is None:
            os.environ.pop("MESHPOOL_TRANSPORT", None)
        else:
            os.environ["MESHPOOL_TRANSPORT"] = old_transport
        if old_quic is None:
            os.environ.pop("MESHPOOL_QUIC_PORT", None)
        else:
            os.environ["MESHPOOL_QUIC_PORT"] = old_quic
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
