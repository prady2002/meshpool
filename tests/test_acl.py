from __future__ import annotations

import asyncio

import pytest
from fastapi.testclient import TestClient

from meshpool.common.config import Settings
from meshpool.common.identity import DeviceIdentity, enrollment_challenge
from meshpool.common.models import now_ms
from meshpool.controlplane.app import build_app
from tests.test_encryption import _manager


def test_bucket_acl_denies_until_read_is_granted():
    mgr, _transport = _manager()
    mgr.create_bucket("private", "alice")
    asyncio.run(mgr.put_object_version(
        b"secret", actor="alice", bucket="private", key="s.txt", erasure=False))

    with pytest.raises(PermissionError):
        asyncio.run(mgr.get_object_version(actor="bob", bucket="private", key="s.txt"))

    mgr.grant_bucket("private", "alice", "bob", ["read"])
    assert asyncio.run(mgr.get_object_version(
        actor="bob", bucket="private", key="s.txt")) == b"secret"
    with pytest.raises(PermissionError):
        asyncio.run(mgr.put_object_version(
            b"overwrite", actor="bob", bucket="private", key="s.txt", erasure=False))


def _session(client: TestClient, base) -> str:
    ident = DeviceIdentity.load_or_create(str(base))
    ts = now_ms()
    sig = ident.sign(enrollment_challenge(ident.device_id, ts))
    r = client.post("/api/v1/auth/device", json={
        "device_id": ident.device_id,
        "public_key": ident.public_key_b64,
        "signature": sig,
        "ts_ms": ts,
    })
    assert r.status_code == 200, r.text
    return r.json()["session_token"]


def test_bucket_acl_route_denies_remote_user_without_loopback_escape_hatch(tmp_path):
    settings = Settings.load(scale="public", data_dir=str(tmp_path / "cp"),
                             host="127.0.0.1", port=9012, auth_mode="enforced")
    with TestClient(build_app(settings), client=("203.0.113.7", 44444)) as client:
        alice = _session(client, tmp_path / "alice")
        bob = _session(client, tmp_path / "bob")
        created = client.post("/api/v1/buckets/team",
                              headers={"X-MeshPool-Session": alice})
        assert created.status_code == 200, created.text

        denied = client.get("/api/v1/buckets/team/objects/missing.txt/versions",
                            headers={"X-MeshPool-Session": bob})
        assert denied.status_code == 403

        grant = client.post("/api/v1/buckets/team/acl",
                            json={"grantee": created.json()["owner"], "permissions": ["read"]},
                            headers={"X-MeshPool-Session": bob})
        assert grant.status_code == 403
