"""Unit tests for the scoped, single-use secret broker (meshpool/common/secrets.py)."""
from __future__ import annotations

import pytest

from meshpool.common.secrets import SecretBroker, SecretError


class FakeClock:
    def __init__(self, t: float = 1000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


def test_put_list_never_exposes_values():
    b = SecretBroker()
    b.put_secret("OPENAI_KEY", "sk-super-secret")
    b.put_secret("DB_PASS", "hunter2")
    assert b.list_names() == ["DB_PASS", "OPENAI_KEY"]  # sorted names only
    # the value never appears in any introspection surface
    assert "sk-super-secret" not in repr(b.list_names())
    assert b.has_secret("OPENAI_KEY") and not b.has_secret("NOPE")


def test_mint_and_redeem_returns_value_once():
    b = SecretBroker()
    b.put_secret("API", "topsecret")
    lease = b.mint_lease("API", node_id="node-1", ttl_s=30)
    assert lease.id.startswith("lease_")
    assert b.redeem(lease.id, "node-1") == "topsecret"


def test_single_use_replay_is_refused():
    b = SecretBroker()
    b.put_secret("API", "topsecret")
    lease = b.mint_lease("API", "node-1")
    assert b.redeem(lease.id, "node-1") == "topsecret"
    with pytest.raises(SecretError, match="already redeemed"):
        b.redeem(lease.id, "node-1")


def test_node_scope_is_enforced():
    b = SecretBroker()
    b.put_secret("API", "topsecret")
    lease = b.mint_lease("API", "node-1")
    with pytest.raises(SecretError, match="not valid for this node"):
        b.redeem(lease.id, "node-2")
    # the rightful node can still use it (the failed attempt didn't consume it)
    assert b.redeem(lease.id, "node-1") == "topsecret"


def test_expiry_is_enforced():
    clk = FakeClock()
    b = SecretBroker(clock=clk)
    b.put_secret("API", "topsecret")
    lease = b.mint_lease("API", "node-1", ttl_s=10)
    clk.advance(11)
    with pytest.raises(SecretError, match="expired"):
        b.redeem(lease.id, "node-1")


def test_unknown_secret_and_unknown_lease():
    b = SecretBroker()
    with pytest.raises(SecretError, match="unknown secret"):
        b.mint_lease("MISSING", "node-1")
    with pytest.raises(SecretError, match="unknown or expired lease"):
        b.redeem("lease_does_not_exist", "node-1")


def test_sweep_drops_expired_leases():
    clk = FakeClock()
    b = SecretBroker(clock=clk)
    b.put_secret("API", "x")
    b.mint_lease("API", "node-1", ttl_s=5)
    b.mint_lease("API", "node-1", ttl_s=5)
    assert b.active_lease_count() == 2
    clk.advance(6)
    assert b.sweep() == 2
    assert b.active_lease_count() == 0


def test_deleted_secret_cannot_be_redeemed():
    b = SecretBroker()
    b.put_secret("API", "topsecret")
    lease = b.mint_lease("API", "node-1", ttl_s=30)
    assert b.delete_secret("API") is True
    with pytest.raises(SecretError, match="no longer exists"):
        b.redeem(lease.id, "node-1")
