"""Agent-side reverse tunnel client.

In global mode the agent has no reachable inbound address, so instead of waiting to
be called it dials **out** to the coordinator and holds open a streaming ``GET``.
The coordinator writes framed requests onto that stream; this client executes each
one against the agent's *own* loopback server (a ``call`` frame) or a loopback
replica (a ``proxy`` frame) and POSTs the reply back to ``/tunnel/reply``.

Executing frames as ordinary loopback HTTP to the existing agent endpoints means
**no agent handler changes** — the tunnel is a thin forwarder in front of the same
routes the LAN path already uses, so both transports share one code path and one
test surface. The client reconnects with exponential backoff and, if its session
token has gone stale (e.g. the coordinator restarted), transparently re-registers.
"""
from __future__ import annotations

import asyncio
import base64
import json
from typing import Any

import httpx

from ..cloud.tunnel import HDR_NODE_ID, HDR_NODE_TOKEN
from ..common.logging import get_logger

log = get_logger("tunnel-client")

_MAX_BACKOFF_S = 10.0


async def _handle_call(client: httpx.AsyncClient, local_url: str, frame: dict) -> dict:
    """Forward a ``call`` frame to the agent's own loopback API."""
    path = str(frame.get("path", "")).lstrip("/")
    method = str(frame.get("method", "POST"))
    payload = frame.get("payload") or {}
    url = f"{local_url}/{path}"
    try:
        resp = await client.request(method, url, json=payload, timeout=120.0)
    except Exception as exc:  # noqa: BLE001
        return {"id": frame.get("id"), "ok": False,
                "error": f"local call failed: {str(exc)[:160]}"}
    ok = resp.status_code < 400
    body: Any = {}
    if ok and resp.content:
        try:
            body = resp.json()
        except Exception:  # noqa: BLE001
            body = {}
    return {
        "id": frame.get("id"), "ok": ok, "status": resp.status_code,
        "body": body if ok else {},
        "error": None if ok else f"HTTP {resp.status_code}: {resp.text[:160]}",
    }


async def _handle_proxy(client: httpx.AsyncClient, frame: dict) -> dict:
    """Forward a ``proxy`` frame (hosted-app traffic / health check) from the agent's
    vantage point to its loopback replica, returning the raw response."""
    method = str(frame.get("method", "GET"))
    url = str(frame.get("url", ""))
    params = [tuple(p) for p in (frame.get("params") or [])]
    headers = frame.get("headers") or None
    content = base64.b64decode(frame.get("content_b64") or "")
    try:
        resp = await client.request(method, url, params=params, content=content,
                                    headers=headers, timeout=30.0)
    except Exception as exc:  # noqa: BLE001
        return {"id": frame.get("id"), "ok": False,
                "error": f"local proxy failed: {str(exc)[:160]}"}
    return {
        "id": frame.get("id"), "ok": True,
        "body": {
            "status": resp.status_code,
            "headers": dict(resp.headers),
            "content_b64": base64.b64encode(resp.content).decode("ascii"),
        },
    }


async def _dispatch(client: httpx.AsyncClient, local_url: str, frame: dict,
                    reply_url: str, headers: dict) -> None:
    op = frame.get("op")
    if op == "ping":
        return  # keepalive; nothing to reply
    if op == "call":
        reply = await _handle_call(client, local_url, frame)
    elif op == "proxy":
        reply = await _handle_proxy(client, frame)
    else:
        reply = {"id": frame.get("id"), "ok": False, "error": f"unknown op {op!r}"}
    try:
        await client.post(reply_url, json=reply, headers=headers, timeout=30.0)
    except Exception as exc:  # noqa: BLE001
        log.debug("reply post failed", err=str(exc)[:120])


async def run_tunnel_client(state) -> None:
    # Wait for the initial registration to mint our session token.
    while not (state.node_id and state.node_token):
        await asyncio.sleep(0.2)
    cp = state.control_plane_url
    stream_url = f"{cp}/tunnel/stream"
    reply_url = f"{cp}/tunnel/reply"
    headers = {HDR_NODE_TOKEN: state.node_token, HDR_NODE_ID: state.node_id or ""}
    backoff = 0.5
    inflight: set[asyncio.Task] = set()
    async with httpx.AsyncClient() as client:
        while True:
            try:
                async with client.stream("GET", stream_url, headers=headers,
                                         timeout=httpx.Timeout(None, connect=10.0)) as resp:
                    if resp.status_code == 200:
                        log.info("tunnel connected", cp=cp, node=state.node_id)
                        backoff = 0.5
                        async for line in resp.aiter_lines():
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                frame = json.loads(line)
                            except Exception:  # noqa: BLE001
                                continue
                            t = asyncio.create_task(
                                _dispatch(client, state.local_url, frame, reply_url, headers))
                            inflight.add(t)
                            t.add_done_callback(inflight.discard)
                    elif resp.status_code in (401, 404):
                        await resp.aread()
                        log.warning("tunnel token stale; re-registering")
                        from . import daemon
                        # Pass our live state explicitly: this module may be a second
                        # import distinct from the ``__main__`` daemon that owns the
                        # populated global state (see daemon._register_once).
                        if await daemon._register_once(client, state):  # noqa: SLF001
                            headers[HDR_NODE_TOKEN] = state.node_token
                            headers[HDR_NODE_ID] = state.node_id or ""
                    else:
                        await resp.aread()
                        log.warning("tunnel stream rejected", status=resp.status_code)
            except asyncio.CancelledError:
                for t in inflight:
                    t.cancel()
                raise
            except Exception as exc:  # noqa: BLE001
                log.warning("tunnel disconnected, retrying",
                            err=str(exc)[:140], backoff=backoff)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, _MAX_BACKOFF_S)
