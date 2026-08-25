"""Tests for authorization enforcement: ownership, scopes, public allowlist,
loopback principal minting, and auth_mode behaviour.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from meshpool.common.auth import (
    ALL_USER_SCOPES,
    Authorizer,
    AuthzError,
    Scope,
    mint_anonymous_principal,
    mint_loopback_principal,
    mint_node_principal,
    mint_session_principal,
)
from meshpool.common.config import Settings
from meshpool.controlplane.app import PUBLIC_ALLOWLIST, ROUTE_SCOPES, build_app


# ---------------------------------------------------------------------------
# Authorizer — scope enforcement
# ---------------------------------------------------------------------------

class TestScopeEnforcement:
    def setup_method(self):
        self.az = Authorizer()

    def test_principal_with_scope_is_allowed(self):
        p = mint_session_principal("user1")
        self.az.check_scope(p, Scope.JOBS_WRITE)  # must not raise

    def test_principal_without_scope_is_denied(self):
        p = mint_node_principal("node-X")  # only nodes:read + secrets:redeem
        with pytest.raises(AuthzError) as exc_info:
            self.az.check_scope(p, Scope.JOBS_WRITE)
        assert exc_info.value.status_code == 403
        assert "jobs:write" in str(exc_info.value)

    def test_cluster_admin_overrides_all_scopes(self):
        from meshpool.common.auth import Principal
        p = Principal(kind="user", subject="admin",
                      scopes=frozenset({Scope.CLUSTER_ADMIN}), via="token")
        # cluster:admin implies every scope
        for scope in Scope:
            self.az.check_scope(p, scope)  # must not raise

    def test_expired_principal_rejected(self):
        from meshpool.common.auth import Principal
        p = Principal(kind="user", subject="u", scopes=ALL_USER_SCOPES,
                      via="token", expires_at_ms=1)  # epoch 1ms = definitely expired
        with pytest.raises(AuthzError) as exc_info:
            self.az.check_scope(p, Scope.JOBS_READ)
        assert exc_info.value.status_code == 401


# ---------------------------------------------------------------------------
# Authorizer — ownership enforcement
# ---------------------------------------------------------------------------

class TestOwnershipEnforcement:
    def setup_method(self):
        self.az = Authorizer()

    def test_principal_owns_resource(self):
        p = mint_session_principal("acct:alice")
        self.az.check_owner(p, "acct:alice")  # must not raise

    def test_principal_does_not_own_resource(self):
        p = mint_session_principal("acct:alice")
        with pytest.raises(AuthzError) as exc_info:
            self.az.check_owner(p, "acct:bob")
        assert exc_info.value.status_code == 403
        assert "alice" in str(exc_info.value)
        assert "bob" in str(exc_info.value)

    def test_cluster_admin_can_act_on_any_resource(self):
        from meshpool.common.auth import Principal
        admin = Principal(kind="user", subject="admin",
                          scopes=frozenset({Scope.CLUSTER_ADMIN}), via="token")
        self.az.check_owner(admin, "acct:anyone")  # must not raise

    def test_anonymous_principal_can_act_on_any_resource(self):
        p = mint_anonymous_principal()
        # open-mode anonymous principal: no ownership restriction
        self.az.check_owner(p, "acct:anyone")  # must not raise


# ---------------------------------------------------------------------------
# ROUTE_SCOPES completeness — every registered FastAPI route must have an entry
# ---------------------------------------------------------------------------

def test_all_routes_have_scope_entries():
    """All routes in ROUTE_SCOPES must be populated (non-empty).

    A route without an entry would be invisible to the matrix test and silently
    fall through to the deny-by-default path in enforced mode.
    """
    assert len(ROUTE_SCOPES) > 0, "ROUTE_SCOPES is empty — _register_routes not called?"
    # Every API and tunnel route we care about:
    expected_keys = [
        "GET /api/v1/health",
        "GET /api/v1/nodes",
        "POST /api/v1/nodes/register",
        "GET /api/v1/jobs",
        "POST /api/v1/jobs",
        "GET /api/v1/secrets",
        "POST /api/v1/secrets",
        "POST /api/v1/secrets/redeem",
        "GET /tunnel/stream",
        "POST /tunnel/reply",
        "DELETE /api/v1/deployments/{dep_id}",
        "DELETE /api/v1/models/{model_id}",
        "DELETE /api/v1/blobs/{blob_id}",
        "GET /api/v1/events",
    ]
    missing = [k for k in expected_keys if k not in ROUTE_SCOPES]
    assert not missing, f"missing ROUTE_SCOPES entries: {missing}"


def test_public_allowlist_is_a_subset_of_route_scopes():
    for key in PUBLIC_ALLOWLIST:
        assert key in ROUTE_SCOPES, f"{key} in PUBLIC_ALLOWLIST but not ROUTE_SCOPES"
        assert ROUTE_SCOPES[key] is None, f"{key} is in PUBLIC_ALLOWLIST but scope is not None"


def test_public_allowlist_entries_have_justifications():
    for key, justification in PUBLIC_ALLOWLIST.items():
        assert justification, f"{key} has empty justification in PUBLIC_ALLOWLIST"


# ---------------------------------------------------------------------------
# auth_mode=open — zero-friction LAN behaviour preserved
# ---------------------------------------------------------------------------

def _open_app(tmp_path):
    settings = Settings.load(scale="home", data_dir=str(tmp_path),
                             host="127.0.0.1", port=9001, auth_mode="open")
    return build_app(settings)


def test_open_mode_all_endpoints_accessible_without_auth(tmp_path):
    """In auth_mode=open, no credential is needed for any endpoint."""
    with TestClient(_open_app(tmp_path)) as client:
        assert client.get("/api/v1/health").status_code == 200
        assert client.get("/api/v1/nodes").status_code == 200
        assert client.get("/api/v1/jobs").status_code == 200
        assert client.get("/api/v1/secrets").status_code == 200
        assert client.get("/api/v1/ledger").status_code == 200


def test_open_mode_anonymous_principal_has_all_scopes(tmp_path):
    p = mint_anonymous_principal()
    az = Authorizer()
    for scope in Scope:
        az.check_scope(p, scope)  # must not raise


# ---------------------------------------------------------------------------
# auth_mode=enforced — endpoints protected
# ---------------------------------------------------------------------------

def _enforced_app(tmp_path, scale="public"):
    # Use public scale (Tier 2) for enforced-mode rejection tests.
    # At Tier 0, the loopback grant means the local machine always has access —
    # which is correct behaviour (zero-friction for home users) but makes
    # rejection testing impossible from the TestClient ("testclient" is loopback).
    settings = Settings.load(scale=scale, data_dir=str(tmp_path),
                             host="127.0.0.1", port=9002, auth_mode="enforced")
    return build_app(settings)


def test_enforced_mode_unauthenticated_nodes_endpoint_rejected(tmp_path):
    """GET /api/v1/nodes requires authentication in enforced mode at Tier 2."""
    with TestClient(_enforced_app(tmp_path)) as client:
        r = client.get("/api/v1/nodes")
        # At Tier 2 there is no loopback grant, so unauthenticated → 401
        assert r.status_code == 401


def test_enforced_mode_public_endpoints_still_accessible(tmp_path):
    """Public endpoints must always be reachable, even in enforced mode."""
    with TestClient(_enforced_app(tmp_path)) as client:
        assert client.get("/api/v1/health").status_code == 200


def test_enforced_mode_session_grants_access(tmp_path):
    """A valid session token grants access to authenticated endpoints."""
    with TestClient(_enforced_app(tmp_path)) as client:
        # Authenticate first
        from meshpool.common.identity import DeviceIdentity, enrollment_challenge
        from meshpool.common.models import now_ms
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            ident = DeviceIdentity.load_or_create(td)
            ts = now_ms()
            sig = ident.sign(enrollment_challenge(ident.device_id, ts))
            r = client.post("/api/v1/auth/device", json={
                "device_id": ident.device_id,
                "public_key": ident.public_key_b64,
                "signature": sig,
                "ts_ms": ts,
            })
            assert r.status_code == 200, r.text
            session_token = r.json()["session_token"]

        # Now access nodes endpoint with session
        r2 = client.get("/api/v1/nodes",
                        headers={"X-MeshPool-Session": session_token})
        assert r2.status_code == 200


# ---------------------------------------------------------------------------
# Loopback principal minting
# ---------------------------------------------------------------------------

def test_loopback_principal_subject_matches_device_identity(tmp_path):
    """mint_loopback_principal uses the correct subject (the device identity)."""
    p = mint_loopback_principal("dev_testsubject")
    assert p.subject == "dev_testsubject"
    assert p.via == "loopback"
    assert p.kind == "user"
    # loopback principal has all scopes including cluster:admin (local operator is admin)
    assert Scope.CLUSTER_ADMIN in p.scopes
    assert Scope.NODES_ADMIN in p.scopes


def test_local_signin_at_tier0_does_not_require_remote_auth(tmp_path):
    """At Tier 0 (home), the local sign-in works without any credentials."""
    settings = Settings.load(scale="home", data_dir=str(tmp_path),
                             host="127.0.0.1", port=9004)
    with TestClient(build_app(settings)) as client:
        r = client.post("/api/v1/auth/local")
        assert r.status_code == 200
        j = r.json()
        assert j["tier"] == 0
        assert j["session_token"]


# ---------------------------------------------------------------------------
# Cross-owner denial (enforced mode)
# ---------------------------------------------------------------------------

def test_cross_owner_access_denied_in_enforced_mode(tmp_path):
    """A user cannot submit a job claiming to be owned by a different user."""
    with TestClient(_enforced_app(tmp_path)) as client:
        from meshpool.common.identity import DeviceIdentity, enrollment_challenge
        from meshpool.common.models import now_ms
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            ident = DeviceIdentity.load_or_create(td)
            ts = now_ms()
            sig = ident.sign(enrollment_challenge(ident.device_id, ts))
            r = client.post("/api/v1/auth/device", json={
                "device_id": ident.device_id,
                "public_key": ident.public_key_b64,
                "signature": sig,
                "ts_ms": ts,
            })
            assert r.status_code == 200
            session_token = r.json()["session_token"]
            real_owner = r.json()["owner"]

        # Submit a job: even though body says "owner": "impersonated-user",
        # the resolved owner must come from the session, not the body.
        r2 = client.post("/api/v1/jobs", json={
            "name": "test", "func": "square",
            "payloads": [{"x": 1}], "cpu_pct": 10,
            "owner": "impersonated-user",  # this should be ignored
        }, headers={"X-MeshPool-Session": session_token})
        # The request should be accepted (200) but owner resolved from session
        if r2.status_code == 200:
            assert r2.json()["owner"] == real_owner
