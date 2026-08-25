from __future__ import annotations

from meshpool.common.db import Store
from meshpool.common.identity import DeviceIdentity, enrollment_challenge
from meshpool.common.models import Capability, RegisterRequest, now_ms
from meshpool.controlplane.registry import EventBus, NodeRegistry


def _reg(store: Store) -> NodeRegistry:
    return NodeRegistry(store, node_timeout_s=10, events=EventBus())


def _signed_req(idn: DeviceIdentity, name: str, url: str) -> RegisterRequest:
    ts = now_ms()
    sig = idn.sign(enrollment_challenge(idn.device_id, ts))
    return RegisterRequest(
        name=name,
        url=url,
        capability=Capability(cpu_cores=2, ram_mb=4000),
        device_id=idn.device_id,
        public_key=idn.public_key_b64,
        signature=sig,
        ts_ms=ts,
    )


def _restart(store: Store) -> tuple[Store, NodeRegistry]:
    path = store.path
    store.close()
    new_store = Store(path)
    return new_store, _reg(new_store)


def test_signed_device_token_survives_registry_restart(tmp_store, tmp_path):
    reg = _reg(tmp_store)
    idn = DeviceIdentity.load_or_create(str(tmp_path / "dev"))

    registered = reg.register(_signed_req(idn, "dev", "http://127.0.0.1:1"), 3)
    node_id = registered.node_id
    token = registered.node_token

    new_store, new_reg = _restart(tmp_store)
    try:
        assert new_reg.node_id_for_token(token) == node_id
    finally:
        new_store.close()


def test_plain_lan_token_survives_registry_restart(tmp_store):
    reg = _reg(tmp_store)
    req = RegisterRequest(
        name="lan",
        url="http://192.168.1.5:9001",
        capability=Capability(cpu_cores=4, ram_mb=8000),
    )

    registered = reg.register(req, 3)
    node_id = registered.node_id
    token = registered.node_token

    new_store, new_reg = _restart(tmp_store)
    try:
        assert new_reg.node_id_for_token(token) == node_id
    finally:
        new_store.close()


def test_re_registering_same_device_reuses_persisted_token(tmp_store, tmp_path):
    reg = _reg(tmp_store)
    idn = DeviceIdentity.load_or_create(str(tmp_path / "dev"))
    first = reg.register(_signed_req(idn, "dev", "http://127.0.0.1:1"), 3)

    new_store, new_reg = _restart(tmp_store)
    try:
        second = new_reg.register(_signed_req(idn, "dev", "http://10.0.0.5:9001"), 3)
        assert second.node_id == first.node_id
        assert second.node_token == first.node_token
    finally:
        new_store.close()
