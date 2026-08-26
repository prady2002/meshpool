from __future__ import annotations

import asyncio
import json
import socket
import ssl
from types import SimpleNamespace

import pytest
from aioquic.asyncio.client import connect
from aioquic.quic.configuration import QuicConfiguration

from meshpool.cloud.quic import ALPN, QuicHub, QuicTransport, auth_challenge, start_quic_server
from meshpool.common.identity import DeviceIdentity
from meshpool.common.transport import TransportError


class _Registry:
    def __init__(self, node_id: str, token: str, device_id: str) -> None:
        self.node_id = node_id
        self.token = token
        self.node = SimpleNamespace(id=node_id, device_id=device_id)

    def node_id_for_token(self, token: str) -> str | None:
        return self.node_id if token == self.token else None

    def get(self, node_id: str):
        return self.node if node_id == self.node_id else None


def _free_udp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


async def _auth_client(port: int, ident: DeviceIdentity, token: str,
                       handler_calls: list[dict]):
    async def handle_stream_async(reader, writer):
        raw = await reader.read(1 << 20)
        frame = json.loads(raw.decode().strip())
        handler_calls.append(frame)
        writer.write(json.dumps({"ok": True, "body": {"echo": frame["payload"]}}).encode())
        writer.write_eof()

    def handle_stream(reader, writer):
        asyncio.create_task(handle_stream_async(reader, writer))

    cfg = QuicConfiguration(is_client=True, alpn_protocols=[ALPN], verify_mode=ssl.CERT_NONE)
    cm = connect("127.0.0.1", port, configuration=cfg, stream_handler=handle_stream)
    protocol = await cm.__aenter__()
    ts = 1_700_000_000_000
    import time
    ts = int(time.time() * 1000)
    sig = ident.sign(auth_challenge("node-1", ident.device_id, ts, token))
    reader, writer = await protocol.create_stream()
    writer.write(json.dumps({
        "op": "auth", "node_token": token, "device_id": ident.device_id,
        "public_key": ident.public_key_b64, "ts_ms": ts, "signature": sig,
    }).encode())
    writer.write_eof()
    raw = await reader.read(1 << 20)
    assert json.loads(raw.decode())["ok"] is True
    return cm, protocol


def test_quic_authenticates_with_existing_ed25519_identity(tmp_path):
    async def run():
        ident = DeviceIdentity.load_or_create(str(tmp_path / "id"))
        hub = QuicHub(_Registry("node-1", "nt_ok", ident.device_id))
        server = await start_quic_server("127.0.0.1", _free_udp_port(),
                                         str(tmp_path / "cp"), hub)
        port = server._transport.get_extra_info("sockname")[1]  # noqa: SLF001
        calls: list[dict] = []
        cm, _protocol = await _auth_client(port, ident, "nt_ok", calls)
        try:
            result = await QuicTransport(hub).call("node-1", "echo", {"x": 7}, timeout=5)
        finally:
            await cm.__aexit__(None, None, None)
            server.close()
        assert result == {"echo": {"x": 7}}
        assert calls and calls[0]["op"] == "call"
        assert hub.is_connected("node-1")

    asyncio.run(run())


def test_quic_rejects_wrong_ed25519_key(tmp_path):
    async def run():
        good = DeviceIdentity.load_or_create(str(tmp_path / "good"))
        bad = DeviceIdentity.load_or_create(str(tmp_path / "bad"))
        hub = QuicHub(_Registry("node-1", "nt_ok", good.device_id))
        reply = await hub.authenticate({
            "node_token": "nt_ok",
            "device_id": bad.device_id,
            "public_key": bad.public_key_b64,
            "ts_ms": 1,
            "signature": bad.sign(auth_challenge("node-1", bad.device_id, 1, "nt_ok")),
        }, protocol=object())
        assert reply["ok"] is False
        assert not hub.is_connected("node-1")

    asyncio.run(run())


def test_quic_request_to_missing_peer_raises():
    async def run():
        hub = QuicHub(_Registry("node-1", "nt_ok", "dev_missing"))
        with pytest.raises(TransportError):
            await QuicTransport(hub).call("node-1", "echo", {}, timeout=0.1)

    asyncio.run(run())
