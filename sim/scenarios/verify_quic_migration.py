"""Acceptance gate: QUIC survives an address change while a request is in flight.

Honest scope
------------
Measured under local emulation with aioquic: this proves MeshPool's QUIC fabric
uses a real QUIC connection that observes a client UDP address change and keeps an
in-flight request stream alive. It is not a real-internet success-rate claim.

Exit 0 = PASS.  Exit 1 = FAIL.
"""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import socket
import ssl
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from aioquic.asyncio.client import connect  # noqa: E402
from aioquic.quic.configuration import QuicConfiguration  # noqa: E402

from meshpool.cloud.quic import (  # noqa: E402
    ALPN, KEEPALIVE_S, QuicHub, QuicTransport, auth_challenge, start_quic_server,
)
from meshpool.common.identity import DeviceIdentity  # noqa: E402
from meshpool.common.netutil import TUNNEL_SENTINEL_URL  # noqa: E402
from sim.harness import Harness  # noqa: E402

GO_DIR = Path(__file__).resolve().parents[2] / "agent-go"


class Registry:
    def __init__(self, ident: DeviceIdentity) -> None:
        self.node_id = "node-quic-migrate"
        self.token = "nt_quic_migration"
        self.node = SimpleNamespace(id=self.node_id, device_id=ident.device_id)

    def node_id_for_token(self, token: str) -> str | None:
        return self.node_id if token == self.token else None

    def get(self, node_id: str):
        return self.node if node_id == self.node_id else None


def _free_udp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


async def _move_udp_socket(protocol) -> None:
    """Move the aioquic client to a fresh UDP port, emulating a NAT rebind."""
    old = protocol._transport  # noqa: SLF001
    loop = asyncio.get_running_loop()
    sock = socket.socket(socket.AF_INET6, socket.SOCK_DGRAM)
    sock.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)
    sock.bind(("::", 0, 0, 0))
    await loop.create_datagram_endpoint(lambda: protocol, sock=sock)
    protocol.change_connection_id()
    await protocol.ping()
    old.close()


async def main_async() -> int:
    data_root = Path(__file__).resolve().parent / "_data_quic_migration"
    shutil.rmtree(data_root, ignore_errors=True)
    ident = DeviceIdentity.load_or_create(str(data_root / "agent"))
    registry = Registry(ident)
    hub = QuicHub(registry)
    port = _free_udp_port()
    server = await start_quic_server("127.0.0.1", port, str(data_root / "cp"), hub)
    call_seen = asyncio.Event()
    release_reply = asyncio.Event()
    handled: list[dict] = []
    failures: list[str] = []

    async def handle_stream_async(reader, writer):
        raw = await reader.read(1 << 20)
        frame = json.loads(raw.decode().strip())
        handled.append(frame)
        call_seen.set()
        await release_reply.wait()
        writer.write(json.dumps({"ok": True, "body": {"seq": frame["payload"]["seq"]}}).encode())
        writer.write_eof()

    def handle_stream(reader, writer):
        asyncio.create_task(handle_stream_async(reader, writer))

    try:
        print("== test 1: connect-by-Ed25519-key QUIC auth ==")
        cfg = QuicConfiguration(is_client=True, alpn_protocols=[ALPN],
                                verify_mode=ssl.CERT_NONE)
        async with connect("127.0.0.1", port, configuration=cfg,
                           stream_handler=handle_stream) as protocol:
            ts = int(time.time() * 1000)
            sig = ident.sign(auth_challenge(registry.node_id, ident.device_id, ts,
                                            registry.token))
            reader, writer = await protocol.create_stream()
            writer.write(json.dumps({
                "op": "auth", "node_token": registry.token, "device_id": ident.device_id,
                "public_key": ident.public_key_b64, "ts_ms": ts, "signature": sig,
            }).encode())
            writer.write_eof()
            auth = json.loads((await reader.read(1 << 20)).decode())
            assert auth.get("ok") is True, auth
            assert hub.is_connected(registry.node_id), "hub did not attach authenticated peer"
            print(f"  PASS: authenticated as {ident.device_id} using existing Ed25519 key")

            print("\n== test 2: keepalive interval is within NAT binding budget ==")
            assert 20.0 <= KEEPALIVE_S <= 25.0, f"KEEPALIVE_S={KEEPALIVE_S}, want 20-25"
            print(f"  PASS: KEEPALIVE_S={KEEPALIVE_S:.0f}s")

            print("\n== test 3: address change mid-stream loses no request ==")
            task = asyncio.create_task(
                QuicTransport(hub).call(registry.node_id, "echo", {"seq": 1}, timeout=10))
            await asyncio.wait_for(call_seen.wait(), timeout=5)
            assert handled, "non-vacuity: server never delivered the request stream"
            assert hub.migration_count(registry.node_id) == 0, (
                "non-vacuity: migration count was already non-zero before moving")

            await _move_udp_socket(protocol)
            end = time.time() + 5
            while time.time() < end and hub.migration_count(registry.node_id) == 0:
                await asyncio.sleep(0.05)
            assert hub.migration_count(registry.node_id) > 0, (
                "non-vacuity: server never observed a peer address change")

            release_reply.set()
            result = await task
            assert result == {"seq": 1}, f"request was lost or corrupted: {result!r}"
            print("  PASS: server observed migration and the in-flight request completed")
    except AssertionError as exc:
        print(f"\nFAILED: {exc}")
        failures.append(str(exc))
    except Exception as exc:  # noqa: BLE001
        print(f"\nFAILED: {type(exc).__name__}: {exc}")
        failures.append(str(exc))
    finally:
        server.close()
    if failures:
        return 1
    print("\nALL QUIC MIGRATION CHECKS PASSED")
    return 0


def _kill_tree(proc: subprocess.Popen | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                       capture_output=True)
    else:
        proc.terminate()
    try:
        proc.wait(timeout=15)
    except Exception:  # noqa: BLE001
        proc.kill()


def _wait_http(url: str, timeout: float = 60.0) -> None:
    import httpx
    end = time.time() + timeout
    last = None
    while time.time() < end:
        try:
            if httpx.get(url, timeout=2.0).status_code < 500:
                return
        except Exception as exc:  # noqa: BLE001
            last = exc
        time.sleep(0.5)
    raise TimeoutError(f"not ready at {url}: {last}")


def go_quic_smoke() -> int:
    """Prove the shipped Go agent actually takes the QUIC path."""
    import httpx

    go = shutil.which("go")
    if go is None:
        print("\n== test 4: Go meshd QUIC path ==")
        print("  SKIP: no Go toolchain on PATH")
        return 0

    print("\n== test 4: Go meshd connects over QUIC ==")
    data_root = Path(__file__).resolve().parent / "_data_quic_go"
    shutil.rmtree(data_root, ignore_errors=True)
    exe = data_root / ("meshd.exe" if os.name == "nt" else "meshd")
    exe.parent.mkdir(parents=True, exist_ok=True)
    build = subprocess.run([go, "build", "-o", str(exe), "./cmd/meshd"],
                           cwd=str(GO_DIR), capture_output=True, text=True)
    if build.returncode != 0:
        print("FAILED: meshd did not build")
        print(build.stdout[-2000:])
        print(build.stderr[-2000:])
        return 1

    quic_port = _free_udp_port()
    old_transport, old_qport = os.environ.get("MESHPOOL_TRANSPORT"), os.environ.get("MESHPOOL_QUIC_PORT")
    os.environ["MESHPOOL_TRANSPORT"] = "quic"
    os.environ["MESHPOOL_QUIC_PORT"] = str(quic_port)
    h = Harness(n_agents=0, cp_port=8438, data_dir=str(data_root / "cluster"))
    proc = None
    try:
        h.start()
        cp = h.cp_url()
        env = dict(os.environ)
        env.update({
            "MESHPOOL_CONTROL_PLANE_URL": cp,
            "MESHPOOL_QUIC_ADDR": f"127.0.0.1:{quic_port}",
            "MESHPOOL_AGENT_HOST": "127.0.0.1",
            "MESHPOOL_AGENT_BIND": "127.0.0.1",
            "MESHPOOL_AGENT_PORT": "8449",
            "MESHPOOL_ADVERTISE_URL": TUNNEL_SENTINEL_URL,
            "MESHPOOL_NAME": "meshd-quic-1",
            "MESHPOOL_DATA_DIR": str(data_root / "meshd"),
            "MESHPOOL_SIM_CORES": "4",
            "MESHPOOL_SIM_RAM_MB": "8192",
        })
        proc = subprocess.Popen([str(exe)], env=env, cwd=str(Path(__file__).resolve().parents[2]))
        _wait_http("http://127.0.0.1:8449/health")
        end = time.time() + 60
        node = None
        while time.time() < end:
            nodes = httpx.get(f"{cp}/api/v1/nodes", timeout=5).json()["nodes"]
            node = next((n for n in nodes if n["name"] == "meshd-quic-1"), None)
            if node and node.get("reachable_via") == "quic":
                break
            time.sleep(0.5)
        assert node is not None, "meshd never registered"
        assert node.get("reachable_via") == "quic", (
            f"non-vacuity: reachable_via={node.get('reachable_via')!r}, not quic")
        r = httpx.post(f"{cp}/api/v1/nodes/{node['id']}/reclaim", json={}, timeout=20)
        assert r.status_code == 200, f"QUIC reclaim call failed: {r.status_code} {r.text[:200]}"
        assert r.json()["status"] == "draining"
        r2 = httpx.post(f"{cp}/api/v1/nodes/{node['id']}/resume", json={}, timeout=20)
        assert r2.status_code == 200, f"QUIC resume call failed: {r2.status_code} {r2.text[:200]}"
        print("  PASS: Go meshd authenticated over QUIC and handled reclaim/resume calls")
        return 0
    except AssertionError as exc:
        print(f"FAILED: {exc}")
        return 1
    except Exception as exc:  # noqa: BLE001
        print(f"FAILED: {type(exc).__name__}: {exc}")
        return 1
    finally:
        _kill_tree(proc)
        h.stop()
        if old_transport is None:
            os.environ.pop("MESHPOOL_TRANSPORT", None)
        else:
            os.environ["MESHPOOL_TRANSPORT"] = old_transport
        if old_qport is None:
            os.environ.pop("MESHPOOL_QUIC_PORT", None)
        else:
            os.environ["MESHPOOL_QUIC_PORT"] = old_qport


if __name__ == "__main__":
    first = asyncio.run(main_async())
    raise SystemExit(first or go_quic_smoke())
