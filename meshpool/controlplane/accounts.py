"""Optional accounts + credit gating for the global (Tier 2) mesh.

Design goals (spec §7, and the product goal "anyone, anywhere, zero-signup"):

* **Zero-signup by default.** Every device already has a cryptographic identity
  (``common/identity.py``). Its *owner* — the subject that earns/spends credits and
  accrues reputation — defaults to ``dev:<device_id>``. Nothing else is required to
  contribute or consume.
* **Optional lightweight account.** A user may link one or more devices to a single
  account (``acct:<id>``) so credits and reputation aggregate across their machines.
  Linking is done through a pluggable :class:`AuthProvider`; a real deployment wires
  GitHub/Google OAuth device-flow here, while the bundled ``LocalAuthProvider`` makes
  the whole flow runnable and testable with no external service.
* **Earn-to-spend.** At Tier 2 a consumer must hold credits to spend them. A newly
  verified owner receives a small one-time starter grant so it can try the mesh, then
  must donate to keep consuming. The grant is idempotent per owner (anti-sybil marker
  in the store); real deployments tie larger grants to verified OAuth accounts.

Tiers below 2 (home/office/enterprise) do **not** gate — ``can_spend`` is always True
and no session is required — so existing LAN behaviour is unchanged.
"""
from __future__ import annotations

import secrets as _secrets
from typing import Optional, Protocol

from ..common.identity import (
    challenge_is_fresh, device_id_for_public_key, enrollment_challenge,
    verify_signature,
)
from ..common.logging import get_logger
from ..common.models import new_id, now_ms
from ..common.trust import ContributionLedger

log = get_logger("accounts")

# One-time starter balance (compute-credits) for a freshly verified anonymous device.
# Deliberately small: creating device keypairs is cheap, so the anonymous grant only
# needs to cover "kick the tyres". Linking a real OAuth account raises the ceiling.
ANON_STARTER_GRANT = 60.0
ACCOUNT_LINK_BONUS = 240.0
# Max concurrently-active jobs a single owner may have in flight at Tier 2 (abuse guard).
MAX_ACTIVE_JOBS_PER_OWNER = 25


class AuthProvider(Protocol):
    """Pluggable identity provider for optional account linking.

    ``begin`` starts a login (returning something the UI shows the user, e.g. a
    device-flow ``user_code`` + ``verification_uri``). ``complete`` exchanges the
    pending login for a stable ``(provider, subject, handle)`` once the user has
    authorised it. A production provider implements OAuth 2.0 device flow here.
    """

    name: str

    def begin(self) -> dict: ...
    def complete(self, pending_id: str, proof: str) -> Optional[tuple[str, str]]: ...


class LocalAuthProvider:
    """Self-contained provider so account linking works with zero external setup.

    It issues a short ``link_code``; the caller "authorises" by presenting that same
    code together with the desired handle. This is intentionally simple but *real*
    (the code must round-trip), and it occupies exactly the seam a GitHub/Google
    OAuth device-flow provider would drop into.
    """

    name = "local"

    def __init__(self) -> None:
        self._pending: dict[str, str] = {}   # pending_id -> link_code

    def begin(self) -> dict:
        pending_id = new_id("lnk")
        code = _secrets.token_hex(3).upper()   # e.g. "9F2A1C"
        self._pending[pending_id] = code
        return {"provider": self.name, "pending_id": pending_id, "link_code": code,
                "instructions": "Enter this link code together with your chosen handle "
                                "to finish linking this device."}

    def complete(self, pending_id: str, proof: str) -> Optional[tuple[str, str]]:
        code = self._pending.get(pending_id)
        if not code:
            return None
        # proof is "<link_code>:<handle>"
        got, _, handle = proof.partition(":")
        if got.strip().upper() != code:
            return None
        self._pending.pop(pending_id, None)
        handle = handle.strip() or "user"
        return (self.name, handle)


class AccountManager:
    def __init__(self, store, ledger: ContributionLedger, tier: int,
                 provider: Optional[AuthProvider] = None) -> None:
        self.store = store
        self.ledger = ledger
        self.tier = tier
        self.provider: AuthProvider = provider or LocalAuthProvider()
        # session_token -> {"owner":..., "device_id":...} (in-memory; consumers
        # re-auth after a CP restart).
        self._sessions: dict[str, dict] = {}

    # --- owner resolution ------------------------------------------------ #
    def owner_for_device(self, device_id: str) -> str:
        """Stable owner subject for a device: its linked account, else the device."""
        acct = self.store.account_for_device(device_id)
        return f"acct:{acct}" if acct else f"dev:{device_id}"

    def ensure_starter_grant(self, owner: str, amount: float = ANON_STARTER_GRANT,
                             detail: str = "starter grant") -> float:
        """Idempotently grant a one-time starter balance to a verified owner.

        Returns the amount actually granted (0.0 if the owner was already granted).
        The store marker makes this exactly-once even under concurrent calls.
        """
        if self.store.mark_grant(owner, now_ms()):
            self.ledger.record_grant(owner, amount, detail=detail)
            log.info("starter grant issued", owner=owner, amount=amount)
            return amount
        return 0.0

    # --- device authentication (consumer side) --------------------------- #
    def authenticate_device(self, device_id: str, public_key: str, signature: str,
                            ts_ms: int) -> tuple[str, str]:
        """Verify a device signature and issue a consumer session.

        Returns ``(owner, session_token)``. Raises ``ValueError`` if the identity is
        malformed or the signature does not prove key possession. Grants the starter
        balance on first authentication of a new owner.
        """
        if not (device_id and public_key and signature and ts_ms):
            raise ValueError("incomplete device identity")
        if device_id_for_public_key(public_key) != device_id:
            raise ValueError("device_id does not match public_key")
        if not challenge_is_fresh(ts_ms, now_ms()):
            raise ValueError("stale device signature (clock skew or replay); retry")
        challenge = enrollment_challenge(device_id, ts_ms)
        if not verify_signature(public_key, challenge, signature):
            raise ValueError("invalid device signature")
        owner = self.owner_for_device(device_id)
        self.ensure_starter_grant(owner)
        token = "cs_" + _secrets.token_urlsafe(24)
        self._sessions[token] = {"owner": owner, "device_id": device_id}
        return owner, token

    def local_session(self, identity) -> tuple[str, str]:
        """Issue a consumer session for the operator of *this* machine.

        The dashboard is served same-origin by the local MeshPool process, which
        holds a device keypair (``common/identity.py``). Whoever can reach that
        locally-served dashboard is, by definition, the machine's operator — so we
        can mint a session without the browser ever handling a private key (the
        route that calls this is loopback-gated at Tier 2). This is what makes the
        "Use" flow zero-signup on the machine you installed MeshPool on.

        Below Tier 2 there is no credit gating and consumption records under the
        caller-supplied ``owner`` (default ``"default"``), so the local session
        represents ``"default"`` there to keep the Wallet view consistent. At Tier 2
        it represents the device's real owner (its linked account, else the device)
        and receives the one-time starter grant.
        """
        if self.tier < 2:
            owner = "default"
        else:
            owner = self.owner_for_device(identity.device_id)
            self.ensure_starter_grant(owner)
        token = "cs_" + _secrets.token_urlsafe(24)
        self._sessions[token] = {"owner": owner, "device_id": identity.device_id}
        return owner, token

    def owner_for_session(self, token: str) -> Optional[str]:
        info = self._sessions.get(token) if token else None
        return info["owner"] if info else None

    def session_info(self, token: str) -> Optional[dict]:
        return self._sessions.get(token) if token else None

    # --- account linking (optional overlay) ------------------------------ #
    def begin_link(self) -> dict:
        return self.provider.begin()

    def complete_link(self, device_id: str, pending_id: str, proof: str) -> dict:
        """Finish linking ``device_id`` to an account via the auth provider.

        Moves the device's owner from ``dev:<id>`` to ``acct:<id>`` and tops up the
        account with a one-time bonus the first time it is created.
        """
        result = self.provider.complete(pending_id, proof)
        if not result:
            raise ValueError("link not authorised (bad or expired code)")
        provider_name, handle = result
        # One account per (provider, handle) for the local provider; a real OAuth
        # provider would key on the immutable provider subject id instead.
        account_id = f"{provider_name}:{handle}"
        if not self.store.get_account(account_id):
            self.store.put_account(account_id, handle, provider_name, now_ms())
            self.ensure_starter_grant(f"acct:{account_id}", ACCOUNT_LINK_BONUS,
                                      detail="account link bonus")
        self.store.link_device(device_id, account_id, now_ms())
        owner = f"acct:{account_id}"
        # Refresh any live session that belonged to this device's old owner so the
        # consumer immediately spends/earns under the linked account.
        for info in self._sessions.values():
            if info.get("device_id") == device_id:
                info["owner"] = owner
        return {"account_id": account_id, "handle": handle, "owner": owner,
                "balance": round(self.balance(owner), 2)}

    # --- credit gating (Tier 2 only) ------------------------------------- #
    def balance(self, owner: str) -> float:
        return self.store.credits_by_owner().get(owner, 0.0)

    def can_spend(self, owner: str) -> bool:
        """Earn-to-spend gate. Below Tier 2 there is no gating (single/trusted org)."""
        if self.tier < 2:
            return True
        return self.balance(owner) > 0.0

    def gating_enabled(self) -> bool:
        return self.tier >= 2
