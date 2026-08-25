"""Tests for capability token issuance, verification, expiry, and revocation.

Covers:
- sign/verify round-trip
- expiry enforcement
- tampering detection (payload and signature bits)
- audience binding
- revocation
- rotation
- cross-node redeem prevention
"""
from __future__ import annotations

import base64
import json
import time

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from meshpool.common.auth import (
    ALL_USER_SCOPES,
    Scope,
    TokenIssuer,
    TokenVerifier,
    mint_anonymous_principal,
    mint_loopback_principal,
    mint_node_principal,
    mint_session_principal,
)


def _make_issuer_verifier():
    priv = Ed25519PrivateKey.generate()
    issuer = TokenIssuer(priv, issuer_id="test-cp")
    verifier = TokenVerifier(priv.public_key())
    return issuer, verifier


# ---------------------------------------------------------------------------
# Round-trip
# ---------------------------------------------------------------------------

def test_issue_and_verify_basic():
    issuer, verifier = _make_issuer_verifier()
    scopes = frozenset({Scope.JOBS_READ, Scope.JOBS_WRITE})
    tok = issuer.issue("dev_abc", "user", scopes)
    p = verifier.verify(tok)
    assert p is not None
    assert p.subject == "dev_abc"
    assert p.kind == "user"
    assert Scope.JOBS_READ in p.scopes
    assert Scope.JOBS_WRITE in p.scopes
    assert p.via == "token"
    assert not p.is_expired()


def test_all_scopes_survive_round_trip():
    issuer, verifier = _make_issuer_verifier()
    tok = issuer.issue("sub", "user", ALL_USER_SCOPES)
    p = verifier.verify(tok)
    assert p is not None
    assert p.scopes == ALL_USER_SCOPES


# ---------------------------------------------------------------------------
# Expiry
# ---------------------------------------------------------------------------

def test_expired_token_rejected():
    issuer, verifier = _make_issuer_verifier()
    tok = issuer.issue("sub", "user", ALL_USER_SCOPES, ttl_ms=-1)
    assert verifier.verify(tok) is None


def test_token_not_yet_expired():
    issuer, verifier = _make_issuer_verifier()
    tok = issuer.issue("sub", "user", ALL_USER_SCOPES, ttl_ms=3_600_000)
    p = verifier.verify(tok)
    assert p is not None
    assert not p.is_expired()


# ---------------------------------------------------------------------------
# Tampering detection
# ---------------------------------------------------------------------------

def test_flip_bit_in_payload_rejected():
    issuer, verifier = _make_issuer_verifier()
    tok = issuer.issue("sub", "user", ALL_USER_SCOPES)
    prefix, payload_b64, sig_b64 = tok.split(".")
    # Decode, flip a byte, re-encode
    rem = len(payload_b64) % 4
    padded = payload_b64 + "=" * ((4 - rem) % 4)
    raw = bytearray(base64.urlsafe_b64decode(padded))
    raw[5] ^= 0xFF
    tampered_b64 = base64.urlsafe_b64encode(bytes(raw)).rstrip(b"=").decode()
    bad_tok = f"{prefix}.{tampered_b64}.{sig_b64}"
    assert verifier.verify(bad_tok) is None


def test_flip_bit_in_signature_rejected():
    issuer, verifier = _make_issuer_verifier()
    tok = issuer.issue("sub", "user", ALL_USER_SCOPES)
    prefix, payload_b64, sig_b64 = tok.split(".")
    rem = len(sig_b64) % 4
    padded = sig_b64 + "=" * ((4 - rem) % 4)
    raw = bytearray(base64.urlsafe_b64decode(padded))
    raw[10] ^= 0x01
    tampered_sig = base64.urlsafe_b64encode(bytes(raw)).rstrip(b"=").decode()
    bad_tok = f"{prefix}.{payload_b64}.{tampered_sig}"
    assert verifier.verify(bad_tok) is None


def test_wrong_key_rejects_token():
    issuer, _ = _make_issuer_verifier()
    _, verifier2 = _make_issuer_verifier()  # different key pair
    tok = issuer.issue("sub", "user", ALL_USER_SCOPES)
    assert verifier2.verify(tok) is None


def test_malformed_token_rejected():
    _, verifier = _make_issuer_verifier()
    assert verifier.verify("") is None
    assert verifier.verify("not.a.token") is None
    assert verifier.verify("mpk_v1.abc") is None
    assert verifier.verify("mpk_v2.a.b.c") is None


# ---------------------------------------------------------------------------
# Audience binding
# ---------------------------------------------------------------------------

def test_audience_bound_token_accepted_for_correct_audience():
    issuer, verifier = _make_issuer_verifier()
    tok = issuer.issue("sub", "device", frozenset({Scope.SECRETS_REDEEM}),
                       audience="node-A")
    p = verifier.verify(tok, required_audience="node-A")
    assert p is not None
    assert p.subject == "sub"


def test_audience_bound_token_rejected_for_wrong_audience():
    issuer, verifier = _make_issuer_verifier()
    tok = issuer.issue("sub", "device", frozenset({Scope.SECRETS_REDEEM}),
                       audience="node-A")
    # node-B cannot redeem a token scoped to node-A
    assert verifier.verify(tok, required_audience="node-B") is None


def test_token_without_audience_accepted_when_no_audience_required():
    issuer, verifier = _make_issuer_verifier()
    tok = issuer.issue("sub", "user", ALL_USER_SCOPES)
    p = verifier.verify(tok)
    assert p is not None


# ---------------------------------------------------------------------------
# Revocation
# ---------------------------------------------------------------------------

def test_revoked_token_rejected():
    issuer, verifier = _make_issuer_verifier()
    tok = issuer.issue("sub", "user", ALL_USER_SCOPES)
    p_before = verifier.verify(tok)
    assert p_before is not None
    jti = json.loads(
        base64.urlsafe_b64decode(tok.split(".")[1] + "==").decode()
    )["jti"]
    verifier.revoke(jti)
    assert verifier.verify(tok) is None


# ---------------------------------------------------------------------------
# Rotation
# ---------------------------------------------------------------------------

def test_refresh_returns_new_token():
    issuer, verifier = _make_issuer_verifier()
    original = issuer.issue("sub", "user", frozenset({Scope.JOBS_READ}))
    refreshed = issuer.refresh(original, verifier)
    assert refreshed is not None
    assert refreshed != original
    p = verifier.verify(refreshed)
    assert p is not None
    assert p.subject == "sub"
    assert Scope.JOBS_READ in p.scopes


def test_refresh_of_expired_token_fails():
    issuer, verifier = _make_issuer_verifier()
    expired = issuer.issue("sub", "user", ALL_USER_SCOPES, ttl_ms=-1)
    assert issuer.refresh(expired, verifier) is None


def test_refresh_of_revoked_token_fails():
    issuer, verifier = _make_issuer_verifier()
    tok = issuer.issue("sub", "user", ALL_USER_SCOPES)
    jti = json.loads(
        base64.urlsafe_b64decode(tok.split(".")[1] + "==").decode()
    )["jti"]
    verifier.revoke(jti)
    assert issuer.refresh(tok, verifier) is None


# ---------------------------------------------------------------------------
# Cross-node redeem prevention
# ---------------------------------------------------------------------------

def test_node_a_token_cannot_redeem_for_node_b():
    issuer, verifier = _make_issuer_verifier()
    tok_a = issuer.issue("dev_A", "device",
                         frozenset({Scope.SECRETS_REDEEM}), audience="dev_A")
    # node B tries to use node A's token
    assert verifier.verify(tok_a, required_audience="dev_B") is None
    # but node A can use it
    p = verifier.verify(tok_a, required_audience="dev_A")
    assert p is not None


# ---------------------------------------------------------------------------
# Principal factories
# ---------------------------------------------------------------------------

def test_anonymous_principal_has_all_scopes():
    p = mint_anonymous_principal()
    assert p.kind == "anonymous"
    assert p.via == "none"
    # anonymous principal includes cluster:admin for open-mode full access
    assert Scope.CLUSTER_ADMIN in p.scopes
    assert not p.is_expired()


def test_loopback_principal_has_all_scopes():
    p = mint_loopback_principal("dev_test")
    assert p.kind == "user"
    assert p.via == "loopback"
    assert p.subject == "dev_test"
    # loopback principal includes cluster:admin (local operator is admin)
    assert Scope.CLUSTER_ADMIN in p.scopes


def test_node_principal_scopes():
    p = mint_node_principal("node-X")
    assert p.kind == "device"
    assert Scope.NODES_READ in p.scopes
    assert Scope.SECRETS_REDEEM in p.scopes
    assert Scope.JOBS_WRITE not in p.scopes


def test_session_principal_has_all_scopes():
    p = mint_session_principal("acct:user1")
    assert p.kind == "user"
    assert p.via == "session"
    assert p.scopes == ALL_USER_SCOPES


# ---------------------------------------------------------------------------
# Scope.describe()
# ---------------------------------------------------------------------------

def test_all_scopes_have_descriptions():
    for scope in Scope:
        desc = scope.describe()
        assert isinstance(desc, str) and len(desc) > 5, f"no description for {scope}"
