"""Authorization core for MeshPool: principals, scopes, and capability tokens.

Design philosophy:
- ``auth_mode=open`` (default) preserves 100% of the existing zero-friction LAN
  behaviour — no credentials required. Loopback principals are *explicitly* granted
  all user scopes, not silently bypassed.
- ``auth_mode=enforced`` requires authenticated credentials on every non-public
  endpoint and validates ownership so callers cannot act as arbitrary owners.
- No hand-rolled crypto. All signing uses ``cryptography``'s vetted Ed25519.
- Tokens are URL-safe, versioned strings: ``mpk_v1.<b64url-payload>.<b64url-sig>``.
"""
from __future__ import annotations

import base64
import json
import secrets
import time
from dataclasses import dataclass
from enum import StrEnum
from typing import FrozenSet, Optional

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)


# ---------------------------------------------------------------------------
# Scopes — explicit, human-readable, NOT free-form strings
# ---------------------------------------------------------------------------

class Scope(StrEnum):
    """Capability scopes for MeshPool principals.

    Deliberately NOT AWS IAM JSON. Each scope maps to a real, auditable API
    surface. Call ``Scope.X.describe()`` to get the plain-English description
    shown in the dashboard and CLI.
    """

    JOBS_READ = "jobs:read"
    JOBS_WRITE = "jobs:write"
    DEPLOYMENTS_READ = "deployments:read"
    DEPLOYMENTS_WRITE = "deployments:write"
    FUNCTIONS_READ = "functions:read"
    FUNCTIONS_WRITE = "functions:write"
    MODELS_READ = "models:read"
    MODELS_WRITE = "models:write"
    BLOBS_READ = "blobs:read"
    BLOBS_WRITE = "blobs:write"
    STORAGE_READ = "storage:read"
    STORAGE_WRITE = "storage:write"
    INFERENCE_INVOKE = "inference:invoke"
    SECRETS_READ = "secrets:read"
    SECRETS_WRITE = "secrets:write"
    SECRETS_REDEEM = "secrets:redeem"
    NODES_READ = "nodes:read"
    NODES_ADMIN = "nodes:admin"
    WALLET_READ = "wallet:read"
    WALLET_WRITE = "wallet:write"
    LEDGER_READ = "ledger:read"
    TOKENS_READ = "tokens:read"
    TOKENS_WRITE = "tokens:write"
    EVENTS_SUBSCRIBE = "events:subscribe"
    CLUSTER_ADMIN = "cluster:admin"

    _DESCRIPTIONS: dict  # declared below class body

    def describe(self) -> str:
        """Plain-English sentence describing this scope (shown in UI and docs)."""
        return _SCOPE_DESCRIPTIONS.get(self.value, self.value)


_SCOPE_DESCRIPTIONS: dict[str, str] = {
    "jobs:read": "Read job status and results.",
    "jobs:write": "Submit batch jobs to the cluster.",
    "deployments:read": "View hosted app deployments and their logs.",
    "deployments:write": "Create and remove hosted app deployments.",
    "functions:read": "View deployed serverless functions and their invocations.",
    "functions:write": "Deploy, update, and remove serverless functions.",
    "models:read": "View deployed inference models.",
    "models:write": "Deploy, upload, and remove inference models.",
    "blobs:read": "Download objects from the pooled blob store.",
    "blobs:write": "Upload and delete objects from the pooled blob store.",
    "storage:read": "Read from the pooled content-addressed object store.",
    "storage:write": "Write to the pooled content-addressed object store.",
    "inference:invoke": "Run inference on a deployed model.",
    "secrets:read": "List secret names (values are never returned).",
    "secrets:write": "Store and delete secrets.",
    "secrets:redeem": "Exchange a single-use node-scoped secret lease for its value.",
    "nodes:read": "View node status, capability, and the cluster roster.",
    "nodes:admin": "Reclaim, resume, and configure node donation policies.",
    "wallet:read": "View credit balance, ledger history, and reputation.",
    "wallet:write": "Set the hard spend cap and reset the spend window.",
    "ledger:read": "View the global credit ledger across all owners.",
    "tokens:read": "List scoped capability tokens for your account.",
    "tokens:write": "Create and revoke scoped capability tokens for your account.",
    "events:subscribe": "Subscribe to the real-time SSE event stream.",
    "cluster:admin": "Full cluster administration (super-user scope).",
}


# Scopes granted to a regular user principal (all operational scopes except cluster:admin).
# cluster:admin is a super-user scope that must be granted explicitly.
ALL_USER_SCOPES: FrozenSet[Scope] = frozenset(s for s in Scope if s != Scope.CLUSTER_ADMIN)

# All scopes including cluster:admin — only for anonymous (open mode) and local operators.
_ALL_SCOPES: FrozenSet[Scope] = frozenset(Scope)


# ---------------------------------------------------------------------------
# Principal — the authenticated (or explicitly anonymous) identity
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Principal:
    """Resolved, immutable identity for one request.

    ``kind``       — ``user`` | ``device`` | ``service`` | ``anonymous``
    ``subject``    — stable id: ``dev_<id>``, ``acct:<id>``, or ``"anonymous"``
    ``scopes``     — the exact set of capabilities this credential grants
    ``via``        — how it authenticated: ``session``, ``node_token``,
                     ``loopback``, ``token``, or ``none``
    ``expires_at_ms`` — epoch-ms expiry, or ``None`` for non-expiring credentials
    """

    kind: str
    subject: str
    scopes: FrozenSet[Scope]
    via: str
    expires_at_ms: Optional[int] = None
    token_id: Optional[str] = None

    def has_scope(self, scope: Scope) -> bool:
        """True if this principal holds ``scope`` or the omnipotent cluster:admin."""
        return scope in self.scopes or Scope.CLUSTER_ADMIN in self.scopes

    def is_expired(self) -> bool:
        if self.expires_at_ms is None:
            return False
        return int(time.time() * 1000) > self.expires_at_ms


# ---------------------------------------------------------------------------
# AuthzError — typed, actionable denial
# ---------------------------------------------------------------------------

class AuthzError(Exception):
    """Raised when an operation is not authorised.

    Always includes a message telling the caller exactly what to do next
    (matching the tone of the existing 401/402 error messages in app.py).
    """

    def __init__(self, message: str, status_code: int = 403) -> None:
        super().__init__(message)
        self.status_code = status_code


# ---------------------------------------------------------------------------
# Token format: mpk_v1.<b64url-payload>.<b64url-sig>
# ---------------------------------------------------------------------------

_TOKEN_VERSION = "mpk_v1"


def _b64u_enc(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _b64u_dec(text: str) -> bytes:
    rem = len(text) % 4
    if rem:
        text += "=" * (4 - rem)
    return base64.urlsafe_b64decode(text.encode("ascii"))


# ---------------------------------------------------------------------------
# TokenIssuer — signs scoped, expiring capability tokens
# ---------------------------------------------------------------------------

class TokenIssuer:
    """Issues signed, expiring, scoped capability tokens.

    Uses the control-plane's own ``DeviceIdentity`` Ed25519 key so no separate
    signing key needs to be provisioned.
    """

    DEFAULT_TTL_MS: int = 3_600_000  # 1 hour

    def __init__(self, private_key: Ed25519PrivateKey, issuer_id: str) -> None:
        self._priv = private_key
        self.issuer_id = issuer_id

    def issue(
        self,
        subject: str,
        kind: str,
        scopes: FrozenSet[Scope],
        *,
        ttl_ms: Optional[int] = None,
        audience: Optional[str] = None,
    ) -> str:
        """Issue a capability token.

        Returns a URL-safe string ``mpk_v1.<payload_b64>.<sig_b64>``.
        ``audience``, if set, binds the token to a specific resource id (e.g. a
        node id for secrets-redeem tokens) and is validated by ``TokenVerifier``.
        """
        now_ms = int(time.time() * 1000)
        exp_ms = now_ms + (ttl_ms if ttl_ms is not None else self.DEFAULT_TTL_MS)
        payload: dict = {
            "sub": subject,
            "kind": kind,
            "scopes": sorted(s.value for s in scopes),
            "iat": now_ms,
            "exp": exp_ms,
            "jti": secrets.token_urlsafe(12),
            "iss": self.issuer_id,
        }
        if audience is not None:
            payload["aud"] = audience
        payload_bytes = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        payload_b64 = _b64u_enc(payload_bytes)
        signed_data = f"{_TOKEN_VERSION}.{payload_b64}".encode("utf-8")
        sig_b64 = _b64u_enc(self._priv.sign(signed_data))
        return f"{_TOKEN_VERSION}.{payload_b64}.{sig_b64}"

    def refresh(
        self,
        token: str,
        verifier: "TokenVerifier",
        ttl_ms: Optional[int] = None,
    ) -> Optional[str]:
        """Issue a fresh token copying scopes from the existing one.

        Returns ``None`` if the current token is invalid or revoked — so agents
        can hold long-lived connections without long-lived credentials by
        periodically calling refresh while their token is still valid.
        """
        principal = verifier.verify(token)
        if principal is None:
            return None
        return self.issue(
            principal.subject,
            principal.kind,
            principal.scopes,
            ttl_ms=ttl_ms,
        )


# ---------------------------------------------------------------------------
# TokenVerifier — checks signature, expiry, audience, revocation
# ---------------------------------------------------------------------------

class TokenVerifier:
    """Verifies capability tokens produced by ``TokenIssuer``."""

    def __init__(self, public_key: Ed25519PublicKey) -> None:
        self._pub = public_key
        self._revoked: set[str] = set()

    def revoke(self, jti: str) -> None:
        """Mark a token id as revoked. Subsequent ``verify`` calls return None."""
        self._revoked.add(jti)

    def verify(
        self,
        token: str,
        *,
        required_audience: Optional[str] = None,
    ) -> Optional[Principal]:
        """Verify ``token``. Returns a ``Principal`` on success, ``None`` on any failure.

        Validates (in order): format, Ed25519 signature, expiry, revocation,
        audience. Constant-time signature verification via ``cryptography``.
        """
        try:
            parts = token.split(".")
            if len(parts) != 3 or parts[0] != _TOKEN_VERSION:
                return None
            _, payload_b64, sig_b64 = parts
            signed_data = f"{_TOKEN_VERSION}.{payload_b64}".encode("utf-8")
            try:
                self._pub.verify(_b64u_dec(sig_b64), signed_data)
            except (InvalidSignature, ValueError):
                return None
            payload = json.loads(_b64u_dec(payload_b64).decode("utf-8"))
        except Exception:  # noqa: BLE001
            return None

        now_ms = int(time.time() * 1000)
        if payload.get("exp", 0) < now_ms:
            return None

        jti = payload.get("jti", "")
        if jti in self._revoked:
            return None

        if required_audience is not None:
            if payload.get("aud") != required_audience:
                return None

        raw_scopes: list[str] = payload.get("scopes", [])
        scopes: FrozenSet[Scope] = frozenset(
            Scope(s) for s in raw_scopes
            if s in Scope._value2member_map_  # type: ignore[attr-defined]
        )
        return Principal(
            kind=payload.get("kind", "user"),
            subject=payload.get("sub", ""),
            scopes=scopes,
            via="token",
            expires_at_ms=payload.get("exp"),
        )


# ---------------------------------------------------------------------------
# Principal factories
# ---------------------------------------------------------------------------

def mint_loopback_principal(subject: str) -> Principal:
    """Mint a full-access Principal for the local machine's operator.

    This is an *explicit, deliberate* grant of all user scopes bound to the
    machine's own DeviceIdentity subject — not an absence of checking.
    The justification: whoever can reach the locally-served dashboard is the
    operator of this machine (enforced by ``_is_loopback`` checks).
    Includes cluster:admin so the local operator can administer their own node.
    """
    return Principal(
        kind="user",
        subject=subject,
        scopes=_ALL_SCOPES,
        via="loopback",
        expires_at_ms=None,
    )


def mint_anonymous_principal() -> Principal:
    """Mint a full-access anonymous Principal for ``auth_mode=open`` operation.

    In open mode this preserves the existing zero-friction LAN experience — the
    anonymous principal is an *explicit* decision to trust all callers, not an
    accident. In ``auth_mode=enforced`` this principal is never returned to
    callers; every request must present real credentials.
    Includes cluster:admin so the anonymous open-mode principal has no restrictions.
    """
    return Principal(
        kind="anonymous",
        subject="anonymous",
        scopes=_ALL_SCOPES,
        via="none",
        expires_at_ms=None,
    )


def mint_node_principal(node_id: str) -> Principal:
    """Mint a Principal representing an authenticated node/agent."""
    return Principal(
        kind="device",
        subject=node_id,
        scopes=frozenset({Scope.NODES_READ, Scope.SECRETS_REDEEM}),
        via="node_token",
        expires_at_ms=None,
    )


def mint_session_principal(owner: str) -> Principal:
    """Mint a full-user Principal for a consumer who authenticated via session."""
    return Principal(
        kind="user",
        subject=owner,
        scopes=ALL_USER_SCOPES,
        via="session",
        expires_at_ms=None,
    )


# ---------------------------------------------------------------------------
# Authorizer — combines scope and ownership enforcement
# ---------------------------------------------------------------------------

class Authorizer:
    """Enforces scope and ownership rules for a single request.

    Raise ``AuthzError`` on denial; callers convert this to HTTP 401/403.
    """

    def check_scope(self, principal: Principal, scope: Scope) -> None:
        """Raise AuthzError if the principal does not hold ``scope``."""
        if principal.is_expired():
            raise AuthzError(
                "your credential has expired — re-authenticate to continue "
                "(POST /api/v1/auth/device or POST /api/v1/auth/local).",
                status_code=401,
            )
        if not principal.has_scope(scope):
            held = ", ".join(sorted(s.value for s in principal.scopes)) or "(none)"
            raise AuthzError(
                f"this operation requires the '{scope}' scope; "
                f"your credential grants: {held}. "
                f"Re-authenticate with a credential that includes '{scope}' or "
                f"contact your administrator.",
                status_code=403,
            )

    def check_owner(
        self,
        principal: Principal,
        resource_owner: str,
        admin_scope: Scope = Scope.CLUSTER_ADMIN,
    ) -> None:
        """Raise AuthzError if the principal cannot act on this owner's resources.

        Principals may act on resources whose owner matches their own subject, or
        when they hold ``admin_scope``. The anonymous principal (open mode) is
        always permitted — it represents the explicit zero-friction LAN grant.
        """
        if principal.kind == "anonymous":
            return   # open mode: deliberate full-access grant
        if admin_scope in principal.scopes:
            return
        if principal.subject == resource_owner:
            return
        raise AuthzError(
            f"you are authenticated as '{principal.subject}' but this resource "
            f"belongs to '{resource_owner}'; you may only act on your own "
            f"resources. Re-authenticate as the correct owner or request the "
            f"'{admin_scope}' scope from your administrator.",
            status_code=403,
        )

    def authorize(
        self,
        principal: Principal,
        scope: Scope,
        resource_owner: Optional[str] = None,
    ) -> None:
        """Combined scope + ownership check. Raises ``AuthzError`` on failure."""
        self.check_scope(principal, scope)
        if resource_owner is not None:
            self.check_owner(principal, resource_owner)
