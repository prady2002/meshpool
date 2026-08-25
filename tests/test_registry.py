"""Node registry: per-node session tokens that verify node identity server-side.

These back the secret-redemption hardening — the control plane derives the node id
from the registration-minted token instead of trusting a caller-supplied node_id, so
a leaked single-use lease id cannot be redeemed by a different node spoofing the id.
"""
from __future__ import annotations

import pytest

from meshpool.common.models import Capability, RegisterRequest
from meshpool.common.secrets import SecretBroker, SecretError
from meshpool.controlplane.registry import EventBus, NodeRegistry


def _reg(store) -> NodeRegistry:
    return NodeRegistry(store, node_timeout_s=10, events=EventBus())


def _req(name: str, url: str) -> RegisterRequest:
    return RegisterRequest(name=name, url=url,
                           capability=Capability(cpu_cores=2, ram_mb=4000))


def test_registration_mints_unique_session_tokens(tmp_store):
    reg = _reg(tmp_store)
    a = reg.register(_req("A", "http://a:9001"), 3)
    b = reg.register(_req("B", "http://b:9001"), 3)
    assert a.node_token and b.node_token
    assert a.node_token != b.node_token
    assert reg.node_id_for_token(a.node_token) == a.node_id
    assert reg.node_id_for_token(b.node_token) == b.node_id
    assert reg.node_id_for_token("nt_bogus") is None
    assert reg.node_id_for_token("") is None


def test_reregistration_by_url_keeps_same_identity_and_token(tmp_store):
    reg = _reg(tmp_store)
    a1 = reg.register(_req("A", "http://a:9001"), 3)
    a2 = reg.register(_req("A", "http://a:9001"), 3)   # same url -> idempotent
    assert a2.node_id == a1.node_id
    assert a2.node_token == a1.node_token


def test_token_scopes_secret_redemption(tmp_store):
    """A lease minted for node A cannot be redeemed by presenting node B's token,
    even though the broker itself would accept A's id — because the id is derived
    from the token, not from the (spoofable) request body."""
    reg = _reg(tmp_store)
    a = reg.register(_req("A", "http://a:9001"), 3)
    b = reg.register(_req("B", "http://b:9001"), 3)

    broker = SecretBroker()
    broker.put_secret("API_KEY", "s3cr3t")
    lease = broker.mint_lease("API_KEY", a.node_id)

    # Attacker holds the lease id but only its own (B) token -> server derives B's id.
    attacker_id = reg.node_id_for_token(b.node_token)
    with pytest.raises(SecretError):
        broker.redeem(lease.id, attacker_id)

    # Legitimate node A presents its own token -> derives A's id -> succeeds once.
    assert broker.redeem(lease.id, reg.node_id_for_token(a.node_token)) == "s3cr3t"
