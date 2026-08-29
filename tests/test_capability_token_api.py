from __future__ import annotations

import time

from fastapi.routing import APIRoute
from fastapi.testclient import TestClient

from meshpool.common.auth import Scope
from meshpool.common.config import Settings
from meshpool.common.identity import DeviceIdentity, enrollment_challenge
from meshpool.common.models import now_ms
from meshpool.controlplane import app as appmod
from meshpool.controlplane.app import ROUTE_SCOPES, build_app


def _client(tmp_path):
    settings = Settings.load(
        scale="public",
        data_dir=str(tmp_path),
        host="127.0.0.1",
        port=9020,
        auth_mode="enforced",
    )
    return TestClient(build_app(settings))


def _session(client: TestClient, tmp_path) -> tuple[str, str]:
    ident = DeviceIdentity.load_or_create(str(tmp_path / "identity"))
    ts = now_ms()
    r = client.post("/api/v1/auth/device", json={
        "device_id": ident.device_id,
        "public_key": ident.public_key_b64,
        "signature": ident.sign(enrollment_challenge(ident.device_id, ts)),
        "ts_ms": ts,
    })
    assert r.status_code == 200, r.text
    return r.json()["owner"], r.json()["session_token"]


def _mint(client: TestClient, session: str, scopes: list[str],
          *, label: str = "ci", expires_in_seconds: float = 3600.0,
          spend_cap: float | None = None) -> dict:
    body = {"label": label, "scopes": scopes, "expires_in_seconds": expires_in_seconds}
    if spend_cap is not None:
        body["spend_cap"] = spend_cap
    r = client.post("/api/v1/tokens", json=body,
                    headers={"X-MeshPool-Session": session})
    assert r.status_code == 200, r.text
    return r.json()


def test_every_registered_route_has_declared_scope(tmp_path):
    app = build_app(Settings.load(scale="public", data_dir=str(tmp_path),
                                  host="127.0.0.1", port=9021, auth_mode="enforced"))
    missing = []
    for route in app.routes:
        if not isinstance(route, APIRoute):
            continue
        path = route.path
        if path.startswith("/static/"):
            continue
        for method in sorted(route.methods or []):
            if method in {"OPTIONS"} and f"{method} {path}" not in ROUTE_SCOPES:
                continue
            key = f"{method} {path}"
            if key not in ROUTE_SCOPES:
                missing.append(key)
    assert missing == []


def test_mint_lists_without_secret_and_read_token_cannot_write(tmp_path):
    with _client(tmp_path) as client:
        owner, session = _session(client, tmp_path)
        minted = _mint(
            client, session, [Scope.JOBS_READ.value],
            label="ci job reader", spend_cap=12.5,
        )

        assert minted["owner"] == owner
        assert minted["label"] == "ci job reader"
        assert minted["scopes"] == [Scope.JOBS_READ.value]
        assert minted["spend_cap"] == 12.5
        assert minted["token"].startswith("mpt_")
        assert minted["id"].startswith("tok-")

        list_response = client.get("/api/v1/tokens",
                                   headers={"X-MeshPool-Session": session})
        assert list_response.status_code == 200, list_response.text
        assert minted["token"] not in list_response.text
        assert "token_hash" not in list_response.text
        assert list_response.json()["tokens"] == [{
            "id": minted["id"],
            "label": "ci job reader",
            "scopes": [Scope.JOBS_READ.value],
            "expires_at_ms": minted["expires_at_ms"],
            "created_ms": minted["created_ms"],
            "last_used_ms": None,
            "spend_cap": 12.5,
            "spent": 0.0,
        }]

        positive = client.get("/api/v1/jobs",
                              headers={"X-MeshPool-Token": minted["token"]})
        assert positive.status_code == 200, positive.text

        denied = client.post("/api/v1/jobs", json={
            "name": "blocked", "func": "square", "payloads": [{"x": 2}],
            "owner": "attacker",
        }, headers={"X-MeshPool-Token": minted["token"]})
        assert denied.status_code == 403, denied.text
        assert "jobs:write" in denied.text

        full = client.post("/api/v1/jobs", json={
            "name": "allowed", "func": "square", "payloads": [{"x": 2}],
            "owner": "attacker",
        }, headers={"X-MeshPool-Session": session})
        assert full.status_code == 200, full.text
        assert full.json()["owner"] == owner


def test_capability_token_cannot_escalate_when_minting_another_token(tmp_path):
    with _client(tmp_path) as client:
        _, session = _session(client, tmp_path)
        broker = _mint(client, session, [Scope.TOKENS_WRITE.value],
                       label="broker")

        ok = client.post("/api/v1/tokens", json={
            "label": "same scope",
            "scopes": [Scope.TOKENS_WRITE.value],
            "expires_in_seconds": 60,
        }, headers={"X-MeshPool-Token": broker["token"]})
        assert ok.status_code == 200, ok.text

        denied = client.post("/api/v1/tokens", json={
            "label": "escalation",
            "scopes": [Scope.JOBS_WRITE.value],
            "expires_in_seconds": 60,
        }, headers={"X-MeshPool-Token": broker["token"]})
        assert denied.status_code == 403, denied.text
        assert "cannot grant scopes" in denied.text


def test_revoked_token_rejected_on_immediately_following_request(tmp_path):
    with _client(tmp_path) as client:
        _, session = _session(client, tmp_path)
        minted = _mint(client, session, [Scope.JOBS_READ.value], label="revoke")

        assert client.get("/api/v1/jobs",
                          headers={"X-MeshPool-Token": minted["token"]}).status_code == 200
        revoke = client.delete(f"/api/v1/tokens/{minted['id']}",
                               headers={"X-MeshPool-Session": session})
        assert revoke.status_code == 200, revoke.text
        assert revoke.json() == {"revoked": True, "id": minted["id"]}

        denied = client.get("/api/v1/jobs",
                            headers={"X-MeshPool-Token": minted["token"]})
        assert denied.status_code == 401, denied.text


def test_expired_token_rejected_after_being_valid(tmp_path):
    with _client(tmp_path) as client:
        _, session = _session(client, tmp_path)
        minted = _mint(client, session, [Scope.JOBS_READ.value],
                       label="short", expires_in_seconds=0.05)

        assert client.get("/api/v1/jobs",
                          headers={"X-MeshPool-Token": minted["token"]}).status_code == 200
        time.sleep(0.08)
        denied = client.get("/api/v1/jobs",
                            headers={"X-MeshPool-Token": minted["token"]})
        assert denied.status_code == 401, denied.text


def _last_used(client: TestClient, session: str, token_id: str):
    listed = client.get("/api/v1/tokens", headers={"X-MeshPool-Session": session})
    assert listed.status_code == 200, listed.text
    for row in listed.json()["tokens"]:
        if row["id"] == token_id:
            return row["last_used_ms"]
    raise AssertionError(f"token {token_id} missing from listing")


def test_rejected_token_is_not_recorded_as_used(tmp_path):
    """A revoked or expired token that is refused must not advance last_used_ms.

    ``last_used_ms`` means *last successful use*. If a rejected probe updated it,
    an operator revoking a leaked key would watch the timestamp keep advancing and
    reasonably conclude that revocation had not taken effect.
    """
    with _client(tmp_path) as client:
        _, session = _session(client, tmp_path)

        revoked = _mint(client, session, [Scope.JOBS_READ.value], label="revoked")
        assert client.get("/api/v1/jobs",
                          headers={"X-MeshPool-Token": revoked["token"]}).status_code == 200
        used_while_valid = _last_used(client, session, revoked["id"])
        assert used_while_valid is not None
        assert client.delete(f"/api/v1/tokens/{revoked['id']}",
                             headers={"X-MeshPool-Session": session}).status_code == 200

        time.sleep(0.01)
        assert client.get("/api/v1/jobs",
                          headers={"X-MeshPool-Token": revoked["token"]}).status_code == 401
        assert _last_used(client, session, revoked["id"]) == used_while_valid

        expiring = _mint(client, session, [Scope.JOBS_READ.value],
                         label="expiring", expires_in_seconds=0.05)
        assert client.get("/api/v1/jobs",
                          headers={"X-MeshPool-Token": expiring["token"]}).status_code == 200
        before_expiry = _last_used(client, session, expiring["id"])
        assert before_expiry is not None

        time.sleep(0.08)
        assert client.get("/api/v1/jobs",
                          headers={"X-MeshPool-Token": expiring["token"]}).status_code == 401
        assert _last_used(client, session, expiring["id"]) == before_expiry


def test_token_usage_is_attributed_to_grantor_and_receipt(tmp_path):
    with _client(tmp_path) as client:
        owner, session = _session(client, tmp_path)
        minted = _mint(client, session, [Scope.JOBS_WRITE.value, Scope.JOBS_READ.value],
                       label="ci runner")

        submitted = client.post("/api/v1/jobs", json={
            "name": "token job", "func": "square", "payloads": [{"x": 3}],
            "owner": "forged-owner",
        }, headers={"X-MeshPool-Token": minted["token"]})
        assert submitted.status_code == 200, submitted.text
        job_id = submitted.json()["job_id"]
        assert submitted.json()["owner"] == owner

        cp = appmod.CP
        assert cp is not None
        job = cp.batch.get(job_id)
        assert job is not None
        assert job.owner == owner
        assert job.capability_token_id == minted["id"]

        line = cp.billing.record_charge(
            "batch", job_id, job.owner, "node-1", "donor",
            measured_seconds=1.0, idempotency_key="unit-1",
            resource_class="batch", resource_units=1.0,
            detail=f"batch:{job_id}", consumer_token_id=job.capability_token_id,
        )
        assert line["consumer_token_id"] == minted["id"]
        receipt = cp.billing.receipt("batch", job_id)
        assert receipt is not None
        assert receipt["capability_token_id"] == minted["id"]
        assert receipt["owner"] == owner

        ledger = [e for e in cp.store.recent_ledger(10) if e.owner == owner]
        assert {e.kind for e in ledger} >= {"consumed"}
        assert any(f"token:{minted['id']}" in e.detail for e in ledger)
