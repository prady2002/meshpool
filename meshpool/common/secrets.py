"""Scoped, short-lived secret brokering (spec §5, §9.8; see docs/security.md).

Tier 0 threat model: the owner sometimes needs to hand a workload a credential
(an API key, a database password) **without** baking it into the job definition or
writing it to disk on a worker. The control plane keeps secret *values* in memory
only and never returns them in listings, logs, or job specs. Instead it issues
single-use, node-scoped, TTL-bounded **leases**:

* A lease is minted server-side at dispatch for one ``(secret, node)`` pair.
* The worker redeems the lease exactly once, over the already-authenticated
  node<->control-plane channel, holds the value in memory for the unit, and never
  persists it.
* A lease that is expired, already redeemed, or presented by a different node is
  refused. So a leaked lease id is useless after its short TTL or first use.

This is deliberately **not** hand-rolled crypto (an explicit non-goal, spec §0/§3):
it is an access-scoping and lifetime primitive. The production upgrade path
(docs/security.md) puts mTLS on the wire so lease ids cannot be sniffed, and can
back the value store with Vault/KMS instead of process memory.
"""
from __future__ import annotations

import secrets as _secrets
import time
from dataclasses import dataclass
from typing import Callable, Optional


class SecretError(Exception):
    """Raised when a secret cannot be stored, minted, or redeemed."""


@dataclass
class Lease:
    id: str
    name: str
    node_id: str
    owner: str
    expires_at: float
    consumed: bool = False


class SecretBroker:
    """In-memory secret store + single-use, node-scoped, TTL-bounded lease broker.

    Values live only here and are never serialised into API responses, job specs,
    or logs. Callers get names (for the dashboard) and lease ids (for workers),
    never raw values — except the single ``redeem`` that hands one value to the one
    node it was minted for.
    """

    def __init__(self, clock: Callable[[], float] = time.time) -> None:
        self._clock = clock
        self._values: dict[tuple[str, str], str] = {}   # (owner, name) -> value
        self._leases: dict[str, Lease] = {}             # lease_id -> Lease

    # --- owner-facing (Tier 0: the single owner manages the vault) --------- #
    def put_secret(self, name: str, value: str, owner: str = "default") -> None:
        if not name or not isinstance(name, str):
            raise SecretError("secret name must be a non-empty string")
        if value is None:
            raise SecretError("secret value must not be None")
        self._values[(owner, name)] = str(value)

    def delete_secret(self, name: str, owner: str = "default") -> bool:
        return self._values.pop((owner, name), None) is not None

    def list_names(self, owner: str = "default") -> list[str]:
        """Names only — values are never exposed here."""
        return sorted(n for (o, n) in self._values if o == owner)

    def has_secret(self, name: str, owner: str = "default") -> bool:
        return (owner, name) in self._values

    # --- dispatch-time (server mints; workers redeem) --------------------- #
    def mint_lease(self, name: str, node_id: str, ttl_s: float = 30.0,
                   owner: str = "default") -> Lease:
        if (owner, name) not in self._values:
            raise SecretError(f"unknown secret: {name!r}")
        if not node_id:
            raise SecretError("a lease must be scoped to a node")
        self.sweep()
        lease = Lease(
            id="lease_" + _secrets.token_urlsafe(18),
            name=name, node_id=node_id, owner=owner,
            expires_at=self._clock() + max(0.0, ttl_s),
        )
        self._leases[lease.id] = lease
        return lease

    def redeem(self, lease_id: str, node_id: str) -> str:
        """Return the secret value for a valid, unused, in-scope, unexpired lease.

        Enforces, in order: existence, node scope, expiry, single-use. The lease is
        marked consumed on success so a replay yields a clear ``already redeemed``.
        """
        lease = self._leases.get(lease_id)
        if lease is None:
            raise SecretError("unknown or expired lease")
        if lease.node_id != node_id:
            # do not leak which of scope/existence failed beyond this
            raise SecretError("lease is not valid for this node")
        if self._clock() >= lease.expires_at:
            self._leases.pop(lease_id, None)
            raise SecretError("lease expired")
        if lease.consumed:
            raise SecretError("lease already redeemed")
        try:
            value = self._values[(lease.owner, lease.name)]
        except KeyError:
            raise SecretError("secret no longer exists")
        lease.consumed = True
        return value

    def sweep(self) -> int:
        """Drop expired leases. Returns the count removed."""
        now = self._clock()
        dead = [lid for lid, l in self._leases.items() if now >= l.expires_at]
        for lid in dead:
            self._leases.pop(lid, None)
        return len(dead)

    # --- introspection (never exposes values) ----------------------------- #
    def active_lease_count(self) -> int:
        self.sweep()
        return sum(1 for l in self._leases.values() if not l.consumed)
