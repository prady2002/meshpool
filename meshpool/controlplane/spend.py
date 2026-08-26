"""Hard spend caps — limits that stop work, not alerts that arrive too late.

WHY THIS EXISTS
---------------
Bill shock is the single most cited complaint about hyperscale clouds, and every
one of them "solves" it with *budget alerts*: an email that arrives after the
money is already gone.  AWS Budgets, GCP budget alerts and Azure cost alerts are
all fundamentally retrospective — by design, nothing stops.

MeshPool's plan of record (§6.3 "Anti-AWS by construction") takes the opposite
position: **a cap is a stop, not a notification.**  When an owner reaches their
cap, in-flight work is halted and new work is refused.  That guarantee is the
whole point; a cap that merely warns is worse than no cap, because it sells the
user a safety property that does not exist.

That distinction is why this module exists at all rather than being a column on
the account: the enforcement has to reach two places that are easy to miss.

    1. **Admission** — ``_require_credits`` in ``controlplane/app.py`` guards all
       twelve work-creating endpoints.  Capped owners are refused there with
       HTTP 402.
    2. **In flight** — ``BatchManager.dispatch_once`` re-checks before dispatching
       each further unit of an already-running job.  Without this second check a
       user could start a 10,000-unit job one credit under their cap and blow
       through it entirely, which is exactly the failure mode caps exist to
       prevent.

ACCOUNTING MODEL
----------------
Spend is measured from the existing ``ledger`` table, so caps cannot drift out of
sync with billing — there is deliberately no second accounting system.

    spent(owner) = gross_consumed(owner) - baseline(owner)

``gross_consumed`` is the owner's lifetime ``kind='consumed'`` total.
``baseline`` is that same total captured at the moment the cap was last set or
reset.  Storing a baseline rather than deleting ledger rows means "reset my
spend for this month" never destroys billing history, and the cap window is a
pure view over an append-only ledger.

An owner with no row and no configured default is **uncapped**, and
``status()`` reports ``enforced: False``.  This is reported honestly rather than
displaying a number that nothing enforces (an earlier build of the account page
showed a hardcoded "200.0" cap that was pure fiction; that is precisely the kind
of claim §10.4 of the plan forbids).
"""
from __future__ import annotations

import os
from typing import Optional

from ..common.logging import get_logger
from ..common.models import now_ms

log = get_logger("controlplane.spend")

# Env-configurable default cap applied to owners with no explicit cap row.
# Unset, empty, or <= 0 means "no default cap" (owners are uncapped until one is
# set).  A deployment that bills real money should set this.
DEFAULT_SPEND_CAP_ENV = "MESHPOOL_DEFAULT_SPEND_CAP"
PUBLIC_DEFAULT_SPEND_CAP = 1_000.0

_DDL = """
CREATE TABLE IF NOT EXISTS spend_caps (
    owner       TEXT PRIMARY KEY,
    cap_units   REAL,
    baseline    REAL NOT NULL DEFAULT 0.0,
    updated_ms  INTEGER NOT NULL
)
"""


def default_cap_from_env(fallback: Optional[float] = None) -> Optional[float]:
    """Read the deployment-wide default cap, or None when uncapped."""
    raw = (os.environ.get(DEFAULT_SPEND_CAP_ENV) or "").strip()
    if not raw:
        return fallback
    try:
        val = float(raw)
    except ValueError:
        log.warning("ignoring non-numeric default spend cap", value=raw[:32])
        return None
    return val if val > 0 else None


class SpendCapManager:
    """Per-owner hard spend caps enforced against the contribution ledger."""

    def __init__(self, store, default_cap: Optional[float] = None) -> None:
        self.store = store
        self._default_cap = (default_cap if default_cap is not None
                             else default_cap_from_env())
        self._ensure_table()

    def _ensure_table(self) -> None:
        # Idempotent DDL rather than a numbered migration: the table is additive,
        # has no foreign keys, and creating it here keeps spend caps self-contained
        # and safe to introduce into an already-populated database.
        with self.store._lock:
            self.store._conn.execute(_DDL)
            self.store._conn.commit()

    # --- raw accounting -------------------------------------------------- #
    def gross_consumed(self, owner: str) -> float:
        """Lifetime consumed compute-units for ``owner`` from the ledger."""
        totals = self.store.ledger_totals_by_owner().get(owner) or {}
        return float(totals.get("consumed", 0.0))

    def _row(self, owner: str) -> Optional[dict]:
        with self.store._lock:
            r = self.store._conn.execute(
                "SELECT owner, cap_units, baseline, updated_ms FROM spend_caps "
                "WHERE owner = ?", (owner,)).fetchone()
        return dict(r) if r is not None else None

    # --- cap management -------------------------------------------------- #
    def set_cap(self, owner: str, cap: Optional[float]) -> dict:
        """Set (or clear, with ``cap=None``) the hard cap for ``owner``.

        Setting a cap does **not** reset accrued spend — otherwise lowering a cap
        would hand the user a fresh allowance, which would make the cap trivially
        bypassable by repeatedly re-setting it.  Use :meth:`reset_spend` for that,
        deliberately as a separate action.
        """
        if cap is not None:
            cap = float(cap)
            if cap < 0:
                raise ValueError("spend cap must be >= 0")
        existing = self._row(owner)
        baseline = float(existing["baseline"]) if existing else 0.0
        with self.store._lock:
            self.store._conn.execute(
                "INSERT INTO spend_caps(owner, cap_units, baseline, updated_ms) "
                "VALUES(?,?,?,?) ON CONFLICT(owner) DO UPDATE SET "
                "cap_units=excluded.cap_units, updated_ms=excluded.updated_ms",
                (owner, cap, baseline, now_ms()))
            self.store._conn.commit()
        log.info("spend cap set", owner=owner, cap=cap)
        return self.status(owner)

    def reset_spend(self, owner: str) -> dict:
        """Start a new cap window: move the baseline up to current gross spend.

        Ledger history is untouched — this only moves the window, so billing and
        receipts stay complete and auditable.
        """
        baseline = self.gross_consumed(owner)
        existing = self._row(owner)
        cap = existing["cap_units"] if existing else self._default_cap
        with self.store._lock:
            self.store._conn.execute(
                "INSERT INTO spend_caps(owner, cap_units, baseline, updated_ms) "
                "VALUES(?,?,?,?) ON CONFLICT(owner) DO UPDATE SET "
                "baseline=excluded.baseline, updated_ms=excluded.updated_ms",
                (owner, cap, baseline, now_ms()))
            self.store._conn.commit()
        log.info("spend window reset", owner=owner, baseline=baseline)
        return self.status(owner)

    # --- enforcement ----------------------------------------------------- #
    def cap_for(self, owner: str) -> Optional[float]:
        """Effective cap for ``owner``: explicit row, else deployment default."""
        row = self._row(owner)
        if row is None:
            return self._default_cap
        cap = row["cap_units"]
        return None if cap is None else float(cap)

    def spent(self, owner: str) -> float:
        """Spend inside the current cap window."""
        row = self._row(owner)
        baseline = float(row["baseline"]) if row else 0.0
        return max(0.0, self.gross_consumed(owner) - baseline)

    def status(self, owner: str) -> dict:
        """Full cap state for APIs and the account UI.

        ``enforced`` is the honest signal the UI must show: when it is False there
        is no cap and none is claimed.

        Values are returned at **full precision on purpose**.  Rounding here once
        produced an incoherent response -- ``remaining: 0.0`` alongside
        ``exceeded: false`` -- because ``spent`` was rounded up for display while
        ``exceeded`` compared the unrounded value.  It also broke the obvious
        client gesture "cap me at what I have already spent": the rounded number
        round-tripped into :meth:`set_cap` landed a hair *above* real spend, so
        the cap silently did not bite.  Formatting is the UI's job.
        """
        cap = self.cap_for(owner)
        spent = self.spent(owner)
        return {
            "owner": owner,
            "cap": cap,
            "spent": spent,
            "remaining": (None if cap is None else max(0.0, cap - spent)),
            "enforced": cap is not None,
            "exceeded": cap is not None and spent >= cap,
        }

    def exceeded(self, owner: str) -> bool:
        """True when ``owner`` has reached their hard cap and work must stop."""
        cap = self.cap_for(owner)
        if cap is None:
            return False
        return self.spent(owner) >= cap
