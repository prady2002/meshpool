"""Regression tests for coordinator-restart self-healing (Phase 4 chaos).

Two independent defects were found by the coordinator-restart chaos gate and are
guarded here so they can never silently return:

1. The agent's reverse-tunnel **re-registration** (run when the coordinator restarts
   and the node's session token goes stale) crashed with ``NoneType has no attribute
   'control_plane_url'``. Cause: the daemon runs as ``__main__`` (``python -m
   meshpool.agent.daemon``) so its populated global ``state`` lives on the ``__main__``
   module, but the tunnel client reaches ``_register_once`` through a *second* import
   (``from . import daemon``) whose global ``state`` is ``None``. The fix threads the
   live state through explicitly.

2. A tunnel-only node reactivated to ONLINE by its heartbeat after a restart looked
   schedulable even though its reverse tunnel had not re-attached, so the scheduler
   dispatched work that could only fail against the unreachable sentinel URL. The fix
   teaches the registry a reachability predicate so a tunnel-only node is schedulable
   only while its tunnel is attached.
"""
from __future__ import annotations

import asyncio

from meshpool.common.identity import DeviceIdentity, enrollment_challenge
from meshpool.common.models import (Capability, DonationPolicy, LinkStats,
                                     RegisterRequest, now_ms)
from meshpool.common.netutil import TUNNEL_SENTINEL_URL
from meshpool.controlplane.registry import EventBus, NodeRegistry


# --------------------------------------------------------------------------- #
# Fix 1 — tunnel re-registration works through the secondary module import
# --------------------------------------------------------------------------- #
class _FakeState:
    def __init__(self, cp_url: str, identity: DeviceIdentity) -> None:
        self.control_plane_url = cp_url
        self.identity = identity
        self.name = "n"
        self.url = TUNNEL_SENTINEL_URL
        self.owner = "default"
        self.enroll_token = None
        self.capability = Capability(cpu_cores=2, ram_mb=4000)
        self.node_id = None
        self.node_token = ""
        self.heartbeat_interval_s = 3
        self._policy = DonationPolicy()

    def policy(self) -> DonationPolicy:
        return self._policy


class _FakeResp:
    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict:
        return {"node_id": "node-abc", "node_token": "nt_xyz",
                "heartbeat_interval_s": 5}


class _FakeClient:
    def __init__(self) -> None:
        self.posted: list = []

    async def post(self, url, json=None, timeout=None):
        self.posted.append((url, json))
        return _FakeResp()


def test_register_once_uses_explicit_state_even_when_module_global_is_none(
        tmp_path, monkeypatch):
    from meshpool.agent import daemon
    # Simulate the tunnel client's view: this import of the module never had its
    # global ``state`` populated (that lives on the ``__main__`` copy).
    monkeypatch.setattr(daemon, "state", None)
    monkeypatch.setattr(daemon, "measure_link", lambda url: LinkStats(latency_ms=1.0))

    idn = DeviceIdentity.load_or_create(str(tmp_path))
    st = _FakeState("http://127.0.0.1:9", idn)
    client = _FakeClient()

    ok = asyncio.run(daemon._register_once(client, st))

    assert ok is True                                   # did not crash on a None global
    assert st.node_id == "node-abc"
    assert st.node_token == "nt_xyz"
    assert st.heartbeat_interval_s == 5
    url, body = client.posted[0]
    assert url.endswith("/api/v1/nodes/register")
    # the request carries a verifiable device signature over the enrollment challenge
    assert body["device_id"] == idn.device_id and body["signature"]


# --------------------------------------------------------------------------- #
# Fix 2 — a tunnel-only node is schedulable only while its tunnel is attached
# --------------------------------------------------------------------------- #
def _reg(store) -> NodeRegistry:
    return NodeRegistry(store, node_timeout_s=10, events=EventBus())


def _signed_req(idn: DeviceIdentity, name: str, url: str) -> RegisterRequest:
    ts = now_ms()
    sig = idn.sign(enrollment_challenge(idn.device_id, ts))
    return RegisterRequest(
        name=name, url=url, capability=Capability(cpu_cores=2, ram_mb=4000),
        device_id=idn.device_id, public_key=idn.public_key_b64, signature=sig, ts_ms=ts)


def test_schedulable_nodes_default_includes_everything(tmp_store, tmp_path):
    """No reachability predicate (LAN mode) -> behaviour byte-for-byte unchanged."""
    reg = _reg(tmp_store)
    idn = DeviceIdentity.load_or_create(str(tmp_path / "a"))
    reg.register(_signed_req(idn, "tun", TUNNEL_SENTINEL_URL), 3)
    reg.register(RegisterRequest(name="lan", url="http://192.168.1.5:9001",
                                 capability=Capability(cpu_cores=2, ram_mb=4000)), 3)
    assert reg.reachability is None
    assert len(reg.schedulable_nodes()) == 2


def test_schedulable_nodes_excludes_detached_tunnel_node(tmp_store, tmp_path):
    reg = _reg(tmp_store)
    idn = DeviceIdentity.load_or_create(str(tmp_path / "a"))
    reg.register(_signed_req(idn, "tun", TUNNEL_SENTINEL_URL), 3)
    reg.register(RegisterRequest(name="lan", url="http://192.168.1.5:9001",
                                 capability=Capability(cpu_cores=2, ram_mb=4000)), 3)
    connected: set[str] = set()
    reg.reachability = lambda n: n.url != TUNNEL_SENTINEL_URL or n.id in connected

    # tunnel detached -> only the directly-addressable LAN node is schedulable
    ids = {n.id for n in reg.schedulable_nodes()}
    assert len(ids) == 1
    lan = next(n for n in reg.list() if n.url != TUNNEL_SENTINEL_URL)
    assert lan.id in ids

    # attach the tunnel -> the global node becomes schedulable too
    tun = next(n for n in reg.list() if n.url == TUNNEL_SENTINEL_URL)
    connected.add(tun.id)
    assert len(reg.schedulable_nodes()) == 2


def test_control_plane_reachability_predicate_tracks_the_hub(tmp_path):
    """The wired-in ControlPlane predicate: LAN nodes always reachable; a tunnel-only
    node reachable exactly when the hub reports its tunnel attached."""
    from meshpool.common.config import Settings
    from meshpool.controlplane.app import ControlPlane

    cp = ControlPlane(Settings.load(scale="public", data_dir=str(tmp_path)))
    try:
        class _N:
            def __init__(self, node_id: str, url: str) -> None:
                self.id, self.url = node_id, url

        lan = _N("node-lan", "http://192.168.1.9:9001")
        tun = _N("node-tun", TUNNEL_SENTINEL_URL)

        assert cp._node_reachable(lan) is True
        assert cp._node_reachable(tun) is False       # tunnel not attached
        cp.hub.attach("node-tun")
        assert cp._node_reachable(tun) is True        # tunnel attached
        cp.hub.drop("node-tun")
        assert cp._node_reachable(tun) is False       # tunnel dropped again
    finally:
        asyncio.run(cp.client.aclose())
