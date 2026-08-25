"""Agent transport abstraction — the seam that makes MeshPool work globally.

Every control-plane → node-agent call historically went out as a direct HTTP
request to the agent's advertised address (``node.url``). That only works when the
control plane can *reach* the agent inbound — i.e. on a shared LAN. To let agents
join from anywhere (behind home routers, NAT, firewalls) we route those same calls
through an ``AgentTransport`` keyed on the stable **node id** instead of a reachable
URL. Two implementations exist:

* :class:`HttpTransport` — direct HTTP to ``node.url``. The original same-network
  behaviour; still the default for LAN pools and the simulation harness.
* ``TunnelTransport`` (``meshpool/cloud/tunnel.py``) — sends each request down the
  agent's persistent, agent-initiated reverse tunnel to the hosted coordinator, so
  no inbound port on the agent is ever required.

The engines (batch / hosting / inference) and the control-plane routes depend only
on this interface, so swapping LAN ↔ global is a wiring change, not an engine
rewrite.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Protocol

import httpx

from .logging import get_logger

log = get_logger("transport")


class TransportError(Exception):
    """Raised when a node call cannot be delivered or the agent returns an error.

    Carries an optional ``status`` so callers can distinguish an agent-side HTTP
    error (>=400) from an unreachable agent (status is None).
    """

    def __init__(self, message: str, status: Optional[int] = None) -> None:
        super().__init__(message)
        self.status = status


@dataclass
class ProxyResult:
    """A minimal, transport-agnostic HTTP response (used by the app load balancer)."""

    status_code: int
    headers: dict[str, str]
    content: bytes


class _HttpStream:
    """A live streaming upstream response (LAN path): status/headers are available as
    soon as they arrive, and the body is pulled incrementally so SSE / chunked / long
    responses reach the client *as they are produced* instead of being fully buffered."""

    def __init__(self, cm: Any, resp: Any) -> None:
        self._cm = cm
        self._resp = resp
        self.status_code: int = resp.status_code
        self.headers: dict[str, str] = dict(resp.headers)

    def aiter_bytes(self) -> Any:
        return self._resp.aiter_bytes()

    async def aclose(self) -> None:
        await self._cm.__aexit__(None, None, None)


class _BufferedStream:
    """Fallback streaming handle over a fully-buffered :class:`ProxyResult` (the tunnel
    path cannot stream incrementally): exposes the same interface but yields once."""

    def __init__(self, result: ProxyResult) -> None:
        self.status_code: int = result.status_code
        self.headers: dict[str, str] = result.headers
        self._content = result.content

    async def aiter_bytes(self) -> Any:
        yield self._content

    async def aclose(self) -> None:
        return



class RegistryRef(Protocol):
    def get(self, node_id: str) -> Any: ...


class AgentTransport(Protocol):
    """What the engines need to talk to a node, regardless of how it is reached."""

    async def call(self, node_id: str, path: str, payload: Optional[dict] = None, *,
                   method: str = "POST", timeout: float = 30.0) -> dict:
        """Invoke the agent's own API (``<agent>/<path>``) and return decoded JSON.

        Raises :class:`TransportError` if the agent is unreachable or replies with a
        >=400 status (mirroring the old ``resp.raise_for_status()`` contract).
        """
        ...

    async def proxy(self, node_id: str, method: str, url: str, *,
                    params: Any = None, content: bytes = b"",
                    headers: Optional[dict] = None,
                    timeout: float = 15.0) -> ProxyResult:
        """Perform an arbitrary HTTP request *from the node's vantage point*.

        Used for hosted-app traffic and replica health checks: in LAN mode the
        control plane reaches the replica URL directly; in tunnel mode the agent
        performs the request against its own ``127.0.0.1`` replica port and streams
        the response back. ``headers`` are the client's forwarded request headers.
        Does **not** raise on HTTP status (the caller inspects ``status_code``); only
        raises :class:`TransportError` on delivery failure.
        """
        ...

    async def proxy_open(self, node_id: str, method: str, url: str, *,
                         params: Any = None, content: bytes = b"",
                         headers: Optional[dict] = None,
                         timeout: float = 15.0) -> Any:
        """Like :meth:`proxy` but returns a *streaming* handle exposing
        ``status_code``, ``headers``, ``aiter_bytes()`` and ``aclose()`` so the caller
        can relay the body incrementally (SSE / chunked). The LAN transport streams
        live; the tunnel transport falls back to a single buffered yield.
        """
        ...

    async def close(self) -> None:
        ...


class HttpTransport:
    """Direct-HTTP transport: reach each agent at its advertised ``node.url``."""

    def __init__(self, registry: RegistryRef,
                 client: Optional[httpx.AsyncClient] = None) -> None:
        self.registry = registry
        self.client = client or httpx.AsyncClient()
        self._own_client = client is None

    def _base_url(self, node_id: str) -> str:
        node = self.registry.get(node_id) if self.registry is not None else None
        if node is None:
            raise TransportError(f"unknown node {node_id!r}")
        return node.url

    def peer_pull_ok(self, node_id: str) -> bool:
        """LAN transport: every node is reached at its advertised URL on the same
        network, so a target node can pull a blob peer-to-peer from another node's URL.
        """
        return True

    async def call(self, node_id: str, path: str, payload: Optional[dict] = None, *,
                   method: str = "POST", timeout: float = 30.0) -> dict:
        url = f"{self._base_url(node_id)}/{path.lstrip('/')}"
        try:
            resp = await self.client.request(method, url, json=payload, timeout=timeout)
        except TransportError:
            raise
        except Exception as exc:  # noqa: BLE001 — connection/timeout → unreachable
            raise TransportError(f"{node_id} {path} unreachable: {str(exc)[:160]}") from exc
        if resp.status_code >= 400:
            raise TransportError(
                f"{node_id} {path} -> HTTP {resp.status_code}: {resp.text[:160]}",
                status=resp.status_code)
        if not resp.content:
            return {}
        try:
            return resp.json()
        except Exception:  # noqa: BLE001 — non-JSON success body
            return {}

    async def proxy(self, node_id: str, method: str, url: str, *,
                    params: Any = None, content: bytes = b"",
                    headers: Optional[dict] = None,
                    timeout: float = 15.0) -> ProxyResult:
        try:
            resp = await self.client.request(method, url, params=params,
                                             content=content, headers=headers,
                                             timeout=timeout)
        except Exception as exc:  # noqa: BLE001
            raise TransportError(f"{node_id} proxy {url} unreachable: {str(exc)[:160]}") from exc
        return ProxyResult(status_code=resp.status_code,
                           headers=dict(resp.headers), content=resp.content)

    async def proxy_open(self, node_id: str, method: str, url: str, *,
                         params: Any = None, content: bytes = b"",
                         headers: Optional[dict] = None,
                         timeout: float = 15.0) -> _HttpStream:
        cm = self.client.stream(method, url, params=params, content=content,
                                headers=headers, timeout=timeout)
        try:
            resp = await cm.__aenter__()
        except Exception as exc:  # noqa: BLE001
            raise TransportError(f"{node_id} proxy {url} unreachable: {str(exc)[:160]}") from exc
        return _HttpStream(cm, resp)

    async def close(self) -> None:
        if self._own_client:
            await self.client.aclose()
