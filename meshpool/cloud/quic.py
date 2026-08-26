"""Agent-initiated QUIC fabric for MeshPool global nodes.

QUIC is an upgrade behind the existing :class:`AgentTransport` seam: agents still
dial out to the coordinator (so NAT/CGNAT needs no port-forwarding), but control
frames then travel over independent QUIC streams.  The HTTPS reverse tunnel stays
the fallback path and remains the only path for agents that do not implement QUIC.
"""
from __future__ import annotations

import asyncio
import base64
import datetime as _dt
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID

from ..common.identity import device_id_for_public_key, verify_signature
from ..common.logging import get_logger
from ..common.transport import AgentTransport, ProxyResult, TransportError, _BufferedStream
from .tunnel import CompositeTransport, TunnelHub, _params_to_list

log = get_logger("quic")

ALPN = "meshpool-quic-v1"
# Residential NAT UDP bindings can expire in ~15-30 s.  A 22 s QUIC keepalive is
# deliberately in Tailscale's 20-26 s range: frequent enough to survive cheap home
# routers, infrequent enough not to waste meaningful bandwidth.
KEEPALIVE_S = 22.0
MAX_IDLE_TIMEOUT_S = 90.0


def auth_challenge(node_id: str, device_id: str, ts_ms: int, token: str) -> bytes:
    """Canonical bytes signed by the node's existing Ed25519 device key."""
    return f"meshpool-quic-auth:{node_id}:{device_id}:{int(ts_ms)}:{token}".encode()


@dataclass
class QuicConn:
    node_id: str
    device_id: str
    public_key: str
    protocol: Any
    peer_addr: Any = None
    migrations: int = 0


class QuicHub:
    """Tracks authenticated agent-initiated QUIC connections."""

    def __init__(self, registry: Any) -> None:
        self.registry = registry
        self._conns: dict[str, QuicConn] = {}

    def is_connected(self, node_id: str) -> bool:
        return node_id in self._conns

    def migration_count(self, node_id: str) -> int:
        conn = self._conns.get(node_id)
        return 0 if conn is None else conn.migrations

    def attach(self, node_id: str, device_id: str, public_key: str, protocol: Any) -> None:
        old = self._conns.get(node_id)
        if old is not None and old.protocol is not protocol:
            try:
                old.protocol.close()
            except Exception:  # noqa: BLE001
                pass
        self._conns[node_id] = QuicConn(
            node_id, device_id, public_key, protocol,
            peer_addr=getattr(protocol, "_meshpool_peer_addr", None))
        log.info("quic attached node=%s device=%s", node_id, device_id)

    def detach_protocol(self, protocol: Any) -> None:
        for node_id, conn in list(self._conns.items()):
            if conn.protocol is protocol:
                self._conns.pop(node_id, None)
                log.info("quic detached node=%s", node_id)

    def note_addr(self, protocol: Any, addr: Any) -> None:
        for conn in self._conns.values():
            if conn.protocol is not protocol:
                continue
            if conn.peer_addr is None:
                conn.peer_addr = addr
            elif addr != conn.peer_addr:
                conn.peer_addr = addr
                conn.migrations += 1
                log.info("quic migration node=%s migrations=%s addr=%s",
                         conn.node_id, conn.migrations, addr)

    async def authenticate(self, payload: dict, protocol: Any) -> dict:
        token = str(payload.get("node_token") or "")
        node_id = self.registry.node_id_for_token(token)
        if not node_id:
            return {"ok": False, "error": "invalid node token"}
        node = self.registry.get(node_id)
        if node is None:
            return {"ok": False, "error": "unknown node"}
        device_id = str(payload.get("device_id") or "")
        public_key = str(payload.get("public_key") or "")
        signature = str(payload.get("signature") or "")
        try:
            ts_ms = int(payload.get("ts_ms") or 0)
        except (TypeError, ValueError):
            ts_ms = 0
        if not (device_id and public_key and signature and ts_ms):
            return {"ok": False, "error": "incomplete device identity"}
        if node.device_id and node.device_id != device_id:
            return {"ok": False, "error": "device id does not match registered node"}
        if device_id_for_public_key(public_key) != device_id:
            return {"ok": False, "error": "device id does not match public key"}
        if abs(int(time.time() * 1000) - ts_ms) > 300_000:
            return {"ok": False, "error": "stale quic auth signature"}
        if not verify_signature(public_key, auth_challenge(node_id, device_id, ts_ms, token),
                                signature):
            return {"ok": False, "error": "invalid device signature"}
        self.attach(node_id, device_id, public_key, protocol)
        return {"ok": True, "node_id": node_id, "device_id": device_id,
                "keepalive_s": KEEPALIVE_S}

    async def request(self, node_id: str, frame: dict, timeout: float) -> dict:
        conn = self._conns.get(node_id)
        if conn is None:
            raise TransportError(f"{node_id} quic not connected")
        try:
            reader, writer = await conn.protocol.create_stream()
            writer.write((json.dumps(frame, separators=(",", ":")) + "\n").encode())
            writer.write_eof()
            raw = await asyncio.wait_for(reader.read(32 * 1024 * 1024), timeout)
            writer.close()
        except Exception as exc:  # noqa: BLE001
            self._conns.pop(node_id, None)
            raise TransportError(f"{node_id} quic request failed: {str(exc)[:160]}") from exc
        try:
            return json.loads(raw.decode("utf-8"))
        except Exception as exc:  # noqa: BLE001
            raise TransportError(f"{node_id} quic returned invalid JSON") from exc


class QuicTransport:
    """AgentTransport implementation using authenticated QUIC sessions."""

    def __init__(self, hub: QuicHub) -> None:
        self.hub = hub

    def peer_pull_ok(self, node_id: str) -> bool:
        return False

    async def call(self, node_id: str, path: str, payload: Optional[dict] = None, *,
                   method: str = "POST", timeout: float = 30.0) -> dict:
        reply = await self.hub.request(node_id, {
            "op": "call", "path": path, "method": method, "payload": payload or {},
        }, timeout)
        if not reply.get("ok", False):
            raise TransportError(reply.get("error") or f"{node_id} quic call failed",
                                 status=reply.get("status"))
        body = reply.get("body")
        return body if isinstance(body, dict) else {}

    async def proxy(self, node_id: str, method: str, url: str, *,
                    params: Any = None, content: bytes = b"",
                    headers: Optional[dict] = None, timeout: float = 15.0) -> ProxyResult:
        reply = await self.hub.request(node_id, {
            "op": "proxy", "method": method, "url": url,
            "params": _params_to_list(params), "headers": headers or {},
            "content_b64": base64.b64encode(content or b"").decode("ascii"),
        }, timeout)
        if not reply.get("ok", False):
            raise TransportError(reply.get("error") or f"{node_id} quic proxy failed",
                                 status=reply.get("status"))
        body = reply.get("body") or {}
        return ProxyResult(
            status_code=int(body.get("status", 502)),
            headers=dict(body.get("headers") or {}),
            content=base64.b64decode(body.get("content_b64") or ""),
        )

    async def proxy_open(self, node_id: str, method: str, url: str, *,
                         params: Any = None, content: bytes = b"",
                         headers: Optional[dict] = None, timeout: float = 15.0):
        return _BufferedStream(await self.proxy(node_id, method, url, params=params,
                                                content=content, headers=headers,
                                                timeout=timeout))

    async def close(self) -> None:
        return


class QuicFallbackTransport:
    """Prefer QUIC, then the HTTPS reverse tunnel, then direct HTTP."""

    def __init__(self, quic: QuicHub, fallback: CompositeTransport) -> None:
        self.quic = quic
        self.quic_transport = QuicTransport(quic)
        self.fallback = fallback
        self.last_path: dict[str, str] = {}

    def path_for(self, node_id: str) -> str:
        if self.quic.is_connected(node_id):
            return "quic"
        if self.fallback.hub.is_connected(node_id):
            return "tunnel"
        return "direct"

    def peer_pull_ok(self, node_id: str) -> bool:
        if self.quic.is_connected(node_id):
            return False
        return self.fallback.peer_pull_ok(node_id)

    async def call(self, node_id: str, path: str, payload: Optional[dict] = None, *,
                   method: str = "POST", timeout: float = 30.0) -> dict:
        if self.quic.is_connected(node_id):
            try:
                self.last_path[node_id] = "quic"
                return await self.quic_transport.call(node_id, path, payload,
                                                      method=method, timeout=timeout)
            except TransportError as exc:
                log.warning("quic call failed; falling back", node=node_id, err=str(exc)[:120])
        self.last_path[node_id] = self.path_for(node_id)
        return await self.fallback.call(node_id, path, payload, method=method, timeout=timeout)

    async def proxy(self, node_id: str, method: str, url: str, *,
                    params: Any = None, content: bytes = b"",
                    headers: Optional[dict] = None, timeout: float = 15.0) -> ProxyResult:
        if self.quic.is_connected(node_id):
            try:
                self.last_path[node_id] = "quic"
                return await self.quic_transport.proxy(node_id, method, url, params=params,
                                                       content=content, headers=headers,
                                                       timeout=timeout)
            except TransportError as exc:
                log.warning("quic proxy failed; falling back", node=node_id, err=str(exc)[:120])
        self.last_path[node_id] = self.path_for(node_id)
        return await self.fallback.proxy(node_id, method, url, params=params,
                                         content=content, headers=headers, timeout=timeout)

    async def proxy_open(self, node_id: str, method: str, url: str, *,
                         params: Any = None, content: bytes = b"",
                         headers: Optional[dict] = None, timeout: float = 15.0):
        if self.quic.is_connected(node_id):
            try:
                self.last_path[node_id] = "quic"
                return await self.quic_transport.proxy_open(node_id, method, url,
                                                            params=params, content=content,
                                                            headers=headers, timeout=timeout)
            except TransportError as exc:
                log.warning("quic stream failed; falling back", node=node_id, err=str(exc)[:120])
        self.last_path[node_id] = self.path_for(node_id)
        return await self.fallback.proxy_open(node_id, method, url, params=params,
                                              content=content, headers=headers,
                                              timeout=timeout)

    async def close(self) -> None:
        await self.fallback.close()


def _ensure_cert(data_dir: str) -> tuple[str, str]:
    root = Path(data_dir) / "quic"
    root.mkdir(parents=True, exist_ok=True)
    cert_path = root / "cert.pem"
    key_path = root / "key.pem"
    if cert_path.exists() and key_path.exists():
        return str(cert_path), str(key_path)
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    subject = issuer = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "meshpool-quic")])
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(_dt.datetime.now(_dt.UTC))
        .not_valid_after(_dt.datetime.now(_dt.UTC) + _dt.timedelta(days=3650))
        .sign(key, hashes.SHA256())
    )
    key_path.write_bytes(key.private_bytes(serialization.Encoding.PEM,
                                           serialization.PrivateFormat.TraditionalOpenSSL,
                                           serialization.NoEncryption()))
    cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    return str(cert_path), str(key_path)


async def start_quic_server(host: str, port: int, data_dir: str, hub: QuicHub) -> Any:
    try:
        from aioquic.asyncio import serve
        from aioquic.asyncio.protocol import QuicConnectionProtocol
        from aioquic.quic.configuration import QuicConfiguration
    except ImportError as exc:  # pragma: no cover - exercised by deployment misconfig
        raise RuntimeError("MESHPOOL_TRANSPORT=quic requires aioquic; install requirements") from exc

    class MeshQuicProtocol(QuicConnectionProtocol):
        def __init__(self, *args, **kwargs):
            kwargs.pop("stream_handler", None)
            super().__init__(*args, stream_handler=self._start_stream, **kwargs)

        def _start_stream(self, reader, writer):  # noqa: ANN001
            asyncio.create_task(self._handle_stream(reader, writer))

        def datagram_received(self, data, addr):  # noqa: ANN001
            self._meshpool_peer_addr = addr
            hub.note_addr(self, addr)
            super().datagram_received(data, addr)

        def connection_lost(self, exc):  # noqa: ANN001
            hub.detach_protocol(self)
            super().connection_lost(exc)

        async def _handle_stream(self, reader, writer):  # noqa: ANN001
            raw = await reader.read(1 << 20)
            try:
                payload = json.loads(raw.decode("utf-8").strip())
                if payload.get("op") != "auth":
                    response = {"ok": False, "error": "first client stream must authenticate"}
                else:
                    response = await hub.authenticate(payload, self)
            except Exception as exc:  # noqa: BLE001
                response = {"ok": False, "error": f"auth failed: {str(exc)[:120]}"}
            writer.write((json.dumps(response, separators=(",", ":")) + "\n").encode())
            writer.write_eof()

    cert, key = _ensure_cert(data_dir)
    cfg = QuicConfiguration(is_client=False, alpn_protocols=[ALPN],
                            idle_timeout=MAX_IDLE_TIMEOUT_S)
    cfg.load_cert_chain(cert, key)
    return await serve(host, port, configuration=cfg, create_protocol=MeshQuicProtocol)
