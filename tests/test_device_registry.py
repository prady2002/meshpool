"""Node registry: cryptographic device identity for global (zero-signup) enrollment.

A device signs an enrollment challenge so the coordinator can key the node record on
a stable device_id (surviving IP/network changes) and refuse spoofed or unsigned
identity claims — while plain LAN registrations (no identity) keep their URL-keyed
behaviour untouched.
"""
from __future__ import annotations

import pytest

from meshpool.common.identity import DeviceIdentity, enrollment_challenge
from meshpool.common.models import Capability, RegisterRequest, now_ms
from meshpool.controlplane.registry import EventBus, NodeRegistry


def _reg(store) -> NodeRegistry:
    return NodeRegistry(store, node_timeout_s=10, events=EventBus())


def _signed_req(idn: DeviceIdentity, name: str, url: str) -> RegisterRequest:
    ts = now_ms()
    sig = idn.sign(enrollment_challenge(idn.device_id, ts))
    return RegisterRequest(
        name=name, url=url, capability=Capability(cpu_cores=2, ram_mb=4000),
        device_id=idn.device_id, public_key=idn.public_key_b64, signature=sig, ts_ms=ts)


def test_signed_device_registration_keys_on_device_id(tmp_store, tmp_path):
    reg = _reg(tmp_store)
    idn = DeviceIdentity.load_or_create(str(tmp_path))
    # Two registrations from the SAME device but DIFFERENT urls (e.g. it moved
    # networks / re-dialed the tunnel) must collapse to one node identity.
    r1 = reg.register(_signed_req(idn, "dev", "http://127.0.0.1:1"), 3)
    r2 = reg.register(_signed_req(idn, "dev", "http://10.0.0.5:9001"), 3)
    assert r2.node_id == r1.node_id
    node = reg.get(r1.node_id)
    assert node.device_id == idn.device_id


def test_distinct_devices_sharing_sentinel_url_stay_separate(tmp_store, tmp_path):
    """Many tunnel nodes advertise the same unreachable sentinel URL; they must not
    collapse onto one node — device_id keeps them distinct."""
    reg = _reg(tmp_store)
    a = DeviceIdentity.load_or_create(str(tmp_path / "a"))
    b = DeviceIdentity.load_or_create(str(tmp_path / "b"))
    ra = reg.register(_signed_req(a, "a", "http://127.0.0.1:1"), 3)
    rb = reg.register(_signed_req(b, "b", "http://127.0.0.1:1"), 3)
    assert ra.node_id != rb.node_id
    assert len(reg.list()) == 2


def test_bad_signature_is_rejected(tmp_store, tmp_path):
    reg = _reg(tmp_store)
    idn = DeviceIdentity.load_or_create(str(tmp_path))
    req = _signed_req(idn, "dev", "http://127.0.0.1:1")
    req.signature = idn.sign(b"a different message")     # signature over wrong bytes
    with pytest.raises(ValueError):
        reg.register(req, 3)


def test_device_id_not_matching_public_key_is_rejected(tmp_store, tmp_path):
    reg = _reg(tmp_store)
    a = DeviceIdentity.load_or_create(str(tmp_path / "a"))
    b = DeviceIdentity.load_or_create(str(tmp_path / "b"))
    req = _signed_req(a, "dev", "http://127.0.0.1:1")
    req.device_id = b.device_id                          # claim another id with a's key
    with pytest.raises(ValueError):
        reg.register(req, 3)


def test_incomplete_identity_is_rejected(tmp_store, tmp_path):
    reg = _reg(tmp_store)
    idn = DeviceIdentity.load_or_create(str(tmp_path))
    req = _signed_req(idn, "dev", "http://127.0.0.1:1")
    req.public_key = None                                # partial identity -> reject
    with pytest.raises(ValueError):
        reg.register(req, 3)


def test_stale_registration_signature_is_rejected(tmp_store, tmp_path):
    """A captured registration signature must not be replayable indefinitely."""
    reg = _reg(tmp_store)
    idn = DeviceIdentity.load_or_create(str(tmp_path))
    stale = now_ms() - 3_600_000                          # signed an hour ago
    sig = idn.sign(enrollment_challenge(idn.device_id, stale))
    req = RegisterRequest(
        name="dev", url="http://127.0.0.1:1",
        capability=Capability(cpu_cores=2, ram_mb=4000),
        device_id=idn.device_id, public_key=idn.public_key_b64,
        signature=sig, ts_ms=stale)
    with pytest.raises(ValueError):                      # outside the freshness window
        reg.register(req, 3)


def test_plain_lan_registration_still_keyed_by_url(tmp_store):
    """No identity fields -> original URL-keyed idempotent behaviour is preserved."""
    reg = _reg(tmp_store)
    req = RegisterRequest(name="lan", url="http://192.168.1.5:9001",
                          capability=Capability(cpu_cores=4, ram_mb=8000))
    a = reg.register(req, 3)
    b = reg.register(req, 3)                              # same url -> same node
    assert a.node_id == b.node_id
    assert reg.get(a.node_id).device_id is None
