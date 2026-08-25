"""S3 credential durability and SigV4 replay-window tests.

Two properties that an in-memory credential dict silently failed to provide:

1. **Survival across restart.** SigV4 is a shared-secret scheme, so the server
   must hold the secret. Holding it only in memory means every boto3 client,
   rclone mount and ``aws s3`` script breaks the moment the control plane
   restarts -- and it breaks with ``SignatureDoesNotMatch``, which points the
   operator at the client rather than at the server that lost the credential.

2. **A bounded signature lifetime.** Without a clock-skew check a captured
   signed request replays forever. That matters more here than inside a VPC,
   because MeshPool traffic crosses untrusted networks by design.
"""
from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone

import pytest

from meshpool.common.config import Settings
from meshpool.controlplane.app import ControlPlane
from meshpool.storage.s3_api import (
    SIGV4_MAX_SKEW_S,
    _timestamp_is_fresh,
)


def _amz(dt: datetime) -> str:
    return dt.strftime("%Y%m%dT%H%M%SZ")


@pytest.fixture()
def cp(tmp_path):
    s = Settings.load(scale="home", data_dir=str(tmp_path),
                      host="127.0.0.1", port=9099)
    return ControlPlane(s)


class TestCredentialsSurviveRestart:
    def test_key_is_still_valid_after_a_restart(self, tmp_path):
        s = Settings.load(scale="home", data_dir=str(tmp_path),
                          host="127.0.0.1", port=9099)
        cp1 = ControlPlane(s)
        access, secret = cp1.s3_credentials.create("acct:local:alice")

        # A brand-new ControlPlane over the same data dir == a restart.
        s2 = Settings.load(scale="home", data_dir=str(tmp_path),
                           host="127.0.0.1", port=9099)
        cp2 = ControlPlane(s2)

        got = cp2.s3_credentials.get(access)
        assert got is not None, (
            "the access key vanished across a restart; every S3 client would "
            "now fail with SignatureDoesNotMatch")
        assert got == (secret, "acct:local:alice")

    def test_unknown_key_returns_none(self, cp):
        assert cp.s3_credentials.get("MESHPOOLDEADBEEF") is None

    def test_empty_key_returns_none(self, cp):
        assert cp.s3_credentials.get("") is None

    def test_keys_are_listed_with_owner(self, cp):
        a1, _ = cp.s3_credentials.create("acct:local:alice")
        a2, _ = cp.s3_credentials.create("acct:local:bob")
        owners = {k: v[1] for k, v in cp.s3_credentials.items()}
        assert owners[a1] == "acct:local:alice"
        assert owners[a2] == "acct:local:bob"

    def test_access_keys_are_unique(self, cp):
        keys = {cp.s3_credentials.create("acct:local:alice")[0] for _ in range(25)}
        assert len(keys) == 25

    def test_revoke_removes_the_key(self, cp):
        access, _ = cp.s3_credentials.create("acct:local:alice")
        assert cp.s3_credentials.revoke(access) is True
        assert cp.s3_credentials.get(access) is None

    def test_revoking_twice_is_reported_honestly(self, cp):
        access, _ = cp.s3_credentials.create("acct:local:alice")
        cp.s3_credentials.revoke(access)
        assert cp.s3_credentials.revoke(access) is False

    def test_revocation_survives_restart(self, tmp_path):
        s = Settings.load(scale="home", data_dir=str(tmp_path),
                          host="127.0.0.1", port=9099)
        cp1 = ControlPlane(s)
        access, _ = cp1.s3_credentials.create("acct:local:alice")
        cp1.s3_credentials.revoke(access)

        cp2 = ControlPlane(Settings.load(scale="home", data_dir=str(tmp_path),
                                         host="127.0.0.1", port=9099))
        assert cp2.s3_credentials.get(access) is None, (
            "a revoked key came back to life after a restart")

    def test_dict_style_access_still_works(self, cp):
        """The store replaced a plain dict; the old surface must still hold."""
        access, secret = cp.s3_credentials.create("acct:local:alice")
        assert access in cp.s3_credentials
        assert cp.s3_credentials[access] == (secret, "acct:local:alice")
        with pytest.raises(KeyError):
            cp.s3_credentials["MESHPOOLNOPE"]


class TestSigV4TimestampWindow:
    def test_now_is_fresh(self):
        assert _timestamp_is_fresh(_amz(datetime.now(timezone.utc))) is True

    def test_just_inside_the_window_is_fresh(self):
        dt = datetime.now(timezone.utc) - timedelta(seconds=SIGV4_MAX_SKEW_S - 60)
        assert _timestamp_is_fresh(_amz(dt)) is True

    def test_old_request_is_refused(self):
        dt = datetime.now(timezone.utc) - timedelta(seconds=SIGV4_MAX_SKEW_S + 60)
        assert _timestamp_is_fresh(_amz(dt)) is False, (
            "a captured request older than the skew window must not replay")

    def test_far_future_request_is_refused(self):
        dt = datetime.now(timezone.utc) + timedelta(hours=2)
        assert _timestamp_is_fresh(_amz(dt)) is False

    def test_clock_slightly_ahead_is_tolerated(self):
        """Client clocks run fast; a small forward skew must not break them."""
        dt = datetime.now(timezone.utc) + timedelta(seconds=60)
        assert _timestamp_is_fresh(_amz(dt)) is True

    @pytest.mark.parametrize("bad", ["", "not-a-date", "20240101", "0", "Z"])
    def test_unparseable_timestamp_is_refused(self, bad):
        """Refuse rather than wave through: a signature scheme that ignores the
        one field bounding its lifetime is not providing the property it seems
        to provide."""
        assert _timestamp_is_fresh(bad) is False

    def test_window_is_measured_against_supplied_now(self):
        base = time.time()
        dt = datetime.fromtimestamp(base, timezone.utc)
        assert _timestamp_is_fresh(_amz(dt), now=base + SIGV4_MAX_SKEW_S - 5) is True
        assert _timestamp_is_fresh(_amz(dt), now=base + SIGV4_MAX_SKEW_S + 5) is False


class TestCredentialEndpointsAreTenantScoped:
    """The list endpoint feeds the revoke endpoint, so a leak in one is a
    denial-of-service in the other: enumerate every access key, then revoke
    each. Both halves are checked here."""

    @staticmethod
    def _client(tmp_path):
        from fastapi.testclient import TestClient
        from meshpool.controlplane.app import build_app
        return TestClient(build_app(Settings.load(
            scale="home", data_dir=str(tmp_path), host="127.0.0.1", port=9098)))

    def test_list_only_returns_the_callers_keys(self, tmp_path):
        with self._client(tmp_path) as c:
            a = c.post("/api/v1/s3/credentials", params={"owner": "alice"}).json()
            b = c.post("/api/v1/s3/credentials", params={"owner": "bob"}).json()
            seen = c.get("/api/v1/s3/credentials",
                         params={"owner": "alice"}).json()["credentials"]
            keys = {x["access_key"] for x in seen}
            assert a["access_key"] in keys
            assert b["access_key"] not in keys, (
                "bob's access key was disclosed to alice")

    def test_cannot_revoke_another_tenants_key(self, tmp_path):
        with self._client(tmp_path) as c:
            b = c.post("/api/v1/s3/credentials", params={"owner": "bob"}).json()
            r = c.delete(f"/api/v1/s3/credentials/{b['access_key']}",
                         params={"owner": "alice"})
            assert r.status_code == 404, (
                "alice revoked bob's key: a cross-tenant kill switch")
            # and bob's key must still work
            still = c.get("/api/v1/s3/credentials",
                          params={"owner": "bob"}).json()["credentials"]
            assert b["access_key"] in {x["access_key"] for x in still}

    def test_owner_can_revoke_their_own_key(self, tmp_path):
        """Control: the check must not break the legitimate path."""
        with self._client(tmp_path) as c:
            a = c.post("/api/v1/s3/credentials", params={"owner": "alice"}).json()
            r = c.delete(f"/api/v1/s3/credentials/{a['access_key']}",
                         params={"owner": "alice"})
            assert r.status_code == 200 and r.json()["revoked"] is True
            left = c.get("/api/v1/s3/credentials",
                         params={"owner": "alice"}).json()["credentials"]
            assert a["access_key"] not in {x["access_key"] for x in left}

    def test_unknown_key_is_404_not_403(self, tmp_path):
        """404 for both missing and not-yours, so the endpoint is not an oracle
        for which access keys exist."""
        with self._client(tmp_path) as c:
            r = c.delete("/api/v1/s3/credentials/MESHPOOLNOSUCHKEY",
                         params={"owner": "alice"})
            assert r.status_code == 404
