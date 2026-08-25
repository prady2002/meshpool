"""Reverse-tunnel data plane for the hosted coordinator.

Agents dial *out* to the coordinator and hold open a long-lived stream (they can do
this from behind any NAT/firewall — it is just an outbound HTTPS request). The
coordinator can then reach an agent by **pushing a framed request down that stream**
and awaiting a correlated reply, instead of opening an inbound connection to the
agent. This module provides:

* :class:`TunnelHub` — tracks one live connection per ``node_id``; hands out an
  outbound frame queue for the stream route to drain, and correlates replies back to
  the coroutine that issued the request (request/response over a one-way push +
  a reply endpoint).
* :class:`CompositeTransport` — an :class:`~meshpool.common.transport.AgentTransport`
  that uses the tunnel when a node is connected through it, and otherwise falls back
  to direct :class:`~meshpool.common.transport.HttpTransport`. This is what lets LAN
  and global nodes coexist in one pool with zero engine changes.

Framing (NDJSON, one JSON object per line) is deliberately simple and dependency
free. Request frames carry an ``id``; the agent echoes it in the reply.
"""
from __future__ import annotations

import asyncio
import base64
import uuid
from typing import Any, Optional

from ..common.logging import get_logger
from ..common.transport import (AgentTransport, ProxyResult, TransportError,
                                _BufferedStream)

log = get_logger("tunnel")

# Headers an agent presents on the stream / reply endpoints to identify itself.
HDR_NODE_ID = "X-MeshPool-Node-Id"
HDR_NODE_TOKEN = "X-MeshPool-Node-Token"

_MAX_QUEUE = 1000  # bound per-connection backlog so a stalled agent can't grow memory


class TunnelConn:
    """One agent's live tunnel: an outbound frame queue + pending reply futures."""

    def __init__(self, node_id: str) -> None:
        self.node_id = node_id
        self.out: asyncio.Queue[dict] = asyncio.Queue(maxsize=_MAX_QUEUE)
        self.pending: dict[str, asyncio.Future] = {}
        self.closed = False

    def fail_all(self, message: str) -> None:
        self.closed = True
        for fut in list(self.pending.values()):
            if not fut.done():
                fut.set_exception(TransportError(message))
        self.pending.clear()


class TunnelHub:
    """Registry of live agent tunnels + request/response correlation."""

    def __init__(self) -> None:
        self._conns: dict[str, TunnelConn] = {}

    # --- connection lifecycle (called by the stream route) --------------- #
    def attach(self, node_id: str) -> TunnelConn:
        """Register a fresh connection for ``node_id`` (replacing any stale one)."""
        old = self._conns.get(node_id)
        if old is not None:
            old.fail_all("tunnel replaced by a newer connection")
        conn = TunnelConn(node_id)
        self._conns[node_id] = conn
        log.info("tunnel attached node=%s", node_id)
        return conn

    def detach(self, node_id: str, conn: TunnelConn) -> None:
        if self._conns.get(node_id) is conn:
            self._conns.pop(node_id, None)
        conn.fail_all("tunnel disconnected")
        log.info("tunnel detached node=%s", node_id)

    def drop(self, node_id: str) -> None:
        """Forcibly tear down a node's tunnel (e.g. the reaper marked it offline).

        A dead agent's tunnel goes *silent* rather than refusing connections, so —
        unlike a direct TCP call — an in-flight tunnel request has no fast failure
        signal and would block until its per-op timeout. Failing every pending
        request the instant the heartbeat reaper declares the node offline gives
        tunnel-mode failover the same promptness as the LAN path.
        """
        conn = self._conns.pop(node_id, None)
        if conn is not None:
            conn.fail_all("node marked offline")
            log.info("tunnel dropped node=%s (offline)", node_id)

    def is_connected(self, node_id: str) -> bool:
        conn = self._conns.get(node_id)
        return conn is not None and not conn.closed

    def connected_nodes(self) -> list[str]:
        return [nid for nid, c in self._conns.items() if not c.closed]

    # --- request / response --------------------------------------------- #
    async def request(self, node_id: str, frame: dict, timeout: float) -> dict:
        """Push ``frame`` to the agent and await its correlated reply payload."""
        conn = self._conns.get(node_id)
        if conn is None or conn.closed:
            raise TransportError(f"{node_id} tunnel not connected")
        req_id = uuid.uuid4().hex
        loop = asyncio.get_event_loop()
        fut: asyncio.Future = loop.create_future()
        conn.pending[req_id] = fut
        try:
            conn.out.put_nowait({**frame, "id": req_id})
        except asyncio.QueueFull:
            conn.pending.pop(req_id, None)
            raise TransportError(f"{node_id} tunnel backlog full")
        try:
            return await asyncio.wait_for(fut, timeout)
        except asyncio.TimeoutError as exc:
            raise TransportError(f"{node_id} tunnel request timed out after {timeout}s") from exc
        finally:
            conn.pending.pop(req_id, None)

    def resolve(self, node_id: str, req_id: str, payload: dict) -> None:
        """Deliver an agent reply to the awaiting :meth:`request` coroutine."""
        conn = self._conns.get(node_id)
        if conn is None:
            return
        fut = conn.pending.pop(req_id, None)
        if fut is not None and not fut.done():
            fut.set_result(payload)


def _params_to_list(params: Any) -> list[list[str]]:
    """Normalise query params (possibly multi-valued) into JSON-safe pairs."""
    if params is None:
        return []
    if hasattr(params, "multi_items"):
        return [[str(k), str(v)] for k, v in params.multi_items()]
    if hasattr(params, "items"):
        return [[str(k), str(v)] for k, v in params.items()]
    try:
        return [[str(k), str(v)] for k, v in params]
    except TypeError:
        return []


class CompositeTransport:
    """Tunnel-aware transport: use the reverse tunnel if connected, else direct HTTP.

    Wrapping :class:`HttpTransport` (rather than replacing it) means a single pool can
    contain both LAN nodes reachable directly and global nodes reachable only through
    their tunnel, with the engines none the wiser.
    """

    def __init__(self, hub: TunnelHub, http: AgentTransport) -> None:
        self.hub = hub
        self.http = http

    def peer_pull_ok(self, node_id: str) -> bool:
        """A tunnel-attached node sits behind NAT: other nodes cannot reach its URL and
        it usually cannot reach theirs, so it must receive weight blobs relayed through
        the control plane. A node reached directly (LAN) can pull peer-to-peer."""
        if self.hub.is_connected(node_id):
            return False
        return self.http.peer_pull_ok(node_id)

    async def call(self, node_id: str, path: str, payload: Optional[dict] = None, *,
                   method: str = "POST", timeout: float = 30.0) -> dict:
        if not self.hub.is_connected(node_id):
            return await self.http.call(node_id, path, payload, method=method, timeout=timeout)
        frame = {"op": "call", "path": path, "method": method, "payload": payload or {}}
        reply = await self.hub.request(node_id, frame, timeout)
        if not reply.get("ok", False):
            raise TransportError(
                reply.get("error") or f"{node_id} {path} tunnel call failed",
                status=reply.get("status"))
        body = reply.get("body")
        return body if isinstance(body, dict) else {}

    async def proxy(self, node_id: str, method: str, url: str, *,
                    params: Any = None, content: bytes = b"",
                    headers: Optional[dict] = None,
                    timeout: float = 15.0) -> ProxyResult:
        if not self.hub.is_connected(node_id):
            return await self.http.proxy(node_id, method, url, params=params,
                                         content=content, headers=headers, timeout=timeout)
        frame = {
            "op": "proxy", "method": method, "url": url,
            "params": _params_to_list(params),
            "headers": headers or {},
            "content_b64": base64.b64encode(content or b"").decode("ascii"),
        }
        reply = await self.hub.request(node_id, frame, timeout)
        if not reply.get("ok", False):
            raise TransportError(
                reply.get("error") or f"{node_id} proxy {url} tunnel failed",
                status=reply.get("status"))
        body = reply.get("body") or {}
        return ProxyResult(
            status_code=int(body.get("status", 502)),
            headers=dict(body.get("headers") or {}),
            content=base64.b64decode(body.get("content_b64") or ""))

    async def proxy_open(self, node_id: str, method: str, url: str, *,
                         params: Any = None, content: bytes = b"",
                         headers: Optional[dict] = None,
                         timeout: float = 15.0):
        # The tunnel is a request/reply frame protocol, not a byte stream, so global
        # nodes fall back to a fully-buffered response wrapped in the streaming handle;
        # directly-reachable (LAN) nodes stream live via HttpTransport.
        if not self.hub.is_connected(node_id):
            return await self.http.proxy_open(node_id, method, url, params=params,
                                              content=content, headers=headers, timeout=timeout)
        result = await self.proxy(node_id, method, url, params=params,
                                  content=content, headers=headers, timeout=timeout)
        return _BufferedStream(result)

    async def close(self) -> None:
        await self.http.close()
