"""Optional accounts + credit gating for the Tier-2 (public) mesh.

Anonymous device identity works with zero signup; a verified owner receives a
one-time starter grant; earn-to-spend gating only bites at Tier 2; and linking a
device to an account moves its owner and aggregates credits.
"""
from __future__ import annotations

import pytest

from meshpool.common.identity import DeviceIdentity, enrollment_challenge
from meshpool.common.models import now_ms
from meshpool.common.trust import ContributionLedger
from meshpool.controlplane.accounts import (
    ANON_STARTER_GRANT, ACCOUNT_LINK_BONUS, AccountManager,
)


def _identity(tmp_path, name="id"):
    return DeviceIdentity.load_or_create(str(tmp_path / name))


def _auth(am, idn, ts=None):
    ts = ts or now_ms()
    sig = idn.sign(enrollment_challenge(idn.device_id, ts))
    return am.authenticate_device(idn.device_id, idn.public_key_b64, sig, ts)


def test_device_auth_grants_starter_balance_once(tmp_store, tmp_path):
    am = AccountManager(tmp_store, ContributionLedger(tmp_store), tier=2)
    idn = _identity(tmp_path)
    owner, token = _auth(am, idn)
    assert owner == f"dev:{idn.device_id}"
    assert am.balance(owner) == ANON_STARTER_GRANT
    assert am.owner_for_session(token) == owner
    # a second login must NOT grant again (anti-sybil, exactly-once per owner)
    _auth(am, idn, ts=now_ms() + 1)
    assert am.balance(owner) == ANON_STARTER_GRANT


def test_bad_signature_is_rejected(tmp_store, tmp_path):
    am = AccountManager(tmp_store, ContributionLedger(tmp_store), tier=2)
    idn = _identity(tmp_path)
    ts = now_ms()
    good = idn.sign(enrollment_challenge(idn.device_id, ts))
    with pytest.raises(ValueError):
        am.authenticate_device(idn.device_id, idn.public_key_b64, good[:-2] + "AA", ts)
    with pytest.raises(ValueError):
        am.authenticate_device(idn.device_id, idn.public_key_b64, good, 0)  # missing ts


def test_stale_signature_is_rejected_as_replay(tmp_store, tmp_path):
    """A validly-signed auth from long ago must not re-authenticate forever."""
    am = AccountManager(tmp_store, ContributionLedger(tmp_store), tier=2)
    idn = _identity(tmp_path)
    stale = now_ms() - 3_600_000                      # signed an hour ago
    sig = idn.sign(enrollment_challenge(idn.device_id, stale))
    with pytest.raises(ValueError):                   # outside the freshness window
        am.authenticate_device(idn.device_id, idn.public_key_b64, sig, stale)


def test_earn_to_spend_gate_only_at_tier2(tmp_store):
    led = ContributionLedger(tmp_store)
    # Below Tier 2 nobody is gated (LAN/office single/trusted owners).
    am1 = AccountManager(tmp_store, led, tier=1)
    assert am1.can_spend("dev:nobody") is True
    assert am1.gating_enabled() is False
    # At Tier 2 a stranger with no credits cannot spend...
    am2 = AccountManager(tmp_store, led, tier=2)
    assert am2.can_spend("dev:stranger") is False
    # ...until they earn (or are granted) some.
    led.record_grant("dev:stranger", 10.0)
    assert am2.can_spend("dev:stranger") is True


def test_consuming_below_grant_still_leaves_balance(tmp_store):
    led = ContributionLedger(tmp_store)
    am = AccountManager(tmp_store, led, tier=2)
    am.ensure_starter_grant("dev:x")               # +60
    led.record_consumption("n1", "dev:x", 20.0)     # -20
    assert am.can_spend("dev:x") is True
    assert round(am.balance("dev:x"), 2) == round(ANON_STARTER_GRANT - 20.0, 2)
    led.record_consumption("n1", "dev:x", 100.0)    # overspend -> negative
    assert am.can_spend("dev:x") is False


def test_account_link_moves_owner_and_aggregates(tmp_store, tmp_path):
    am = AccountManager(tmp_store, ContributionLedger(tmp_store), tier=2)
    idn = _identity(tmp_path)
    owner, token = _auth(am, idn)
    begin = am.begin_link()
    res = am.complete_link(idn.device_id, begin["pending_id"],
                           f"{begin['link_code']}:octocat")
    assert res["owner"] == "acct:local:octocat"
    assert res["balance"] == ACCOUNT_LINK_BONUS
    # device now resolves to the account, and the live session was refreshed
    assert am.owner_for_device(idn.device_id) == res["owner"]
    assert am.owner_for_session(token) == res["owner"]


def test_link_requires_valid_code(tmp_store, tmp_path):
    am = AccountManager(tmp_store, ContributionLedger(tmp_store), tier=2)
    idn = _identity(tmp_path)
    _auth(am, idn)
    begin = am.begin_link()
    with pytest.raises(ValueError):
        am.complete_link(idn.device_id, begin["pending_id"], "WRONG:octocat")


def test_local_session_grants_and_binds_owner_at_tier2(tmp_store, tmp_path):
    """The dashboard's zero-signup sign-in: a session bound to the machine's own
    device, with the one-time starter grant, so 'Use' works without any keys in the
    browser and without an explicit device-signature round-trip."""
    am = AccountManager(tmp_store, ContributionLedger(tmp_store), tier=2)
    idn = _identity(tmp_path)
    owner, token = am.local_session(idn)
    assert owner == f"dev:{idn.device_id}"
    assert am.owner_for_session(token) == owner
    assert am.balance(owner) == ANON_STARTER_GRANT
    # a second local sign-in must not double-grant (exactly-once per owner)
    owner2, token2 = am.local_session(idn)
    assert owner2 == owner and token2 != token
    assert am.balance(owner) == ANON_STARTER_GRANT


def test_local_session_is_open_owner_below_tier2(tmp_store, tmp_path):
    """Below Tier 2 there is no gating and consumption records under 'default'; the
    local session represents 'default' so the Wallet view stays consistent, and no
    grant is issued (grants are a Tier-2 anti-sybil concept)."""
    am = AccountManager(tmp_store, ContributionLedger(tmp_store), tier=0)
    idn = _identity(tmp_path)
    owner, token = am.local_session(idn)
    assert owner == "default"
    assert am.owner_for_session(token) == "default"
    assert am.balance("default") == 0.0
