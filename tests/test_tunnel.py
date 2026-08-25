"""Unit tests for the reverse-tunnel hub + tunnel-aware composite transport."""
from __future__ import annotations

import asyncio
import base64

import pytest

from meshpool.cloud.tunnel import CompositeTransport, TunnelHub
from meshpool.common.transport import ProxyResult, TransportError


class _FakeHttp:
    """Records fallback calls so tests can assert tunnel-vs-direct routing."""

    def __init__(self):
        self.calls = []
        self.proxies = []

    async def call(self, node_id, path, payload=None, *, method="POST", timeout=30.0):
        self.calls.append((node_id, path, payload, method))
        return {"via": "http"}

    async def proxy(self, node_id, method, url, *, params=None, content=b"", timeout=15.0):
        self.proxies.append((node_id, method, url))
        return ProxyResult(200, {}, b"http")

    async def close(self):
        pass


# --- TunnelHub correlation ------------------------------------------------- #
def test_is_connected_reflects_attach_and_drop():
    hub = TunnelHub()
    assert not hub.is_connected("n1")
    hub.attach("n1")
    assert hub.is_connected("n1") and hub.connected_nodes() == ["n1"]
    hub.drop("n1")
    assert not hub.is_connected("n1")


def test_hub_request_resolves_correlated_reply():
    async def run():
        hub = TunnelHub()
        conn = hub.attach("n1")

        async def agent():
            frame = await conn.out.get()
            assert frame["op"] == "call" and "id" in frame
            hub.resolve("n1", frame["id"], {"ok": True, "body": {"echo": frame["payload"]}})

        asyncio.create_task(agent())
        return await hub.request("n1", {"op": "call", "payload": {"x": 1}}, timeout=2.0)

    res = asyncio.run(run())
    assert res == {"ok": True, "body": {"echo": {"x": 1}}}


def test_hub_request_on_unconnected_node_raises():
    async def run():
        hub = TunnelHub()
        with pytest.raises(TransportError):
            await hub.request("nope", {"op": "call"}, timeout=0.2)

    asyncio.run(run())


def test_hub_request_times_out():
    async def run():
        hub = TunnelHub()
        hub.attach("n1")                     # connected but never replies
        with pytest.raises(TransportError):
            await hub.request("n1", {"op": "call"}, timeout=0.05)

    asyncio.run(run())


def test_hub_drop_fails_inflight_request_promptly():
    async def run():
        hub = TunnelHub()
        hub.attach("n1")

        async def dropper():
            await asyncio.sleep(0.02)
            hub.drop("n1")                    # e.g. reaper declared the node offline

        asyncio.create_task(dropper())
        with pytest.raises(TransportError):
            await hub.request("n1", {"op": "call"}, timeout=5.0)

    asyncio.run(run())


def test_reattach_fails_old_connection_pending():
    async def run():
        hub = TunnelHub()
        hub.attach("n1")

        async def reconnect():
            await asyncio.sleep(0.02)
            hub.attach("n1")                  # a newer connection replaces the old one

        asyncio.create_task(reconnect())
        with pytest.raises(TransportError):
            await hub.request("n1", {"op": "call"}, timeout=5.0)

    asyncio.run(run())


# --- CompositeTransport routing -------------------------------------------- #
def test_composite_falls_back_to_http_when_not_connected():
    async def run():
        http = _FakeHttp()
        ct = CompositeTransport(TunnelHub(), http)
        r = await ct.call("n1", "run_unit", {"a": 1})
        assert r == {"via": "http"}
        assert http.calls == [("n1", "run_unit", {"a": 1}, "POST")]

    asyncio.run(run())


def test_composite_uses_tunnel_when_connected():
    async def run():
        hub = TunnelHub()
        conn = hub.attach("n1")
        http = _FakeHttp()
        ct = CompositeTransport(hub, http)

        async def agent():
            frame = await conn.out.get()
            assert frame["op"] == "call" and frame["path"] == "run_unit"
            assert frame["payload"] == {"a": 1}
            hub.resolve("n1", frame["id"], {"ok": True, "body": {"r": 9}})

        asyncio.create_task(agent())
        r = await ct.call("n1", "run_unit", {"a": 1}, timeout=2.0)
        assert r == {"r": 9}
        assert http.calls == []                  # did NOT fall back to direct HTTP

    asyncio.run(run())


def test_composite_tunnel_call_error_propagates_status():
    async def run():
        hub = TunnelHub()
        conn = hub.attach("n1")
        ct = CompositeTransport(hub, _FakeHttp())

        async def agent():
            frame = await conn.out.get()
            hub.resolve("n1", frame["id"], {"ok": False, "status": 503, "error": "draining"})

        asyncio.create_task(agent())
        with pytest.raises(TransportError) as ei:
            await ct.call("n1", "run_unit", {}, timeout=2.0)
        assert ei.value.status == 503

    asyncio.run(run())


def test_composite_proxy_over_tunnel_returns_result():
    async def run():
        hub = TunnelHub()
        conn = hub.attach("n1")
        ct = CompositeTransport(hub, _FakeHttp())

        async def agent():
            frame = await conn.out.get()
            assert frame["op"] == "proxy" and frame["method"] == "GET"
            body = {"status": 200, "headers": {"content-type": "text/plain"},
                    "content_b64": base64.b64encode(b"hello").decode("ascii")}
            hub.resolve("n1", frame["id"], {"ok": True, "body": body})

        asyncio.create_task(agent())
        res = await ct.proxy("n1", "GET", "http://127.0.0.1:9/health", timeout=2.0)
        assert res.status_code == 200 and res.content == b"hello"
        assert res.headers.get("content-type") == "text/plain"

    asyncio.run(run())
