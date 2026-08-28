"""One-dimensional MeshPool metering and per-job receipts.

The public economy has exactly one billable unit: the ``mesh-second``.  Different
resource classes are normalized into that unit by a single weight table, but the
ledger, receipts, caps and UI all see one dimension.  Receipts show gross donor
earnings and the platform fee as separate lines; nothing is silently netted.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Optional

from .models import LedgerEntry, now_ms
from .trust import ContributionLedger

MESH_SECOND = "mesh-second"
PLATFORM_OWNER = "meshpool:platform-fee"
PLATFORM_FEE_RATE_ENV = "MESHPOOL_PLATFORM_FEE_RATE"
DEFAULT_PLATFORM_FEE_RATE = 0.10

RESOURCE_CLASS_WEIGHTS: dict[str, float] = {
    "standard": 1.0,
    "cpu": 1.0,
    "functions": 1.0,
    "inference": 1.0,
    "hosting": 1.0,
    "storage": 1.0,
}

_DDL = """
CREATE TABLE IF NOT EXISTS billing_receipts (
    receipt_id        TEXT PRIMARY KEY,
    service           TEXT NOT NULL,
    job_id            TEXT NOT NULL,
    consumer_owner    TEXT NOT NULL,
    capability_token_id TEXT,
    unit              TEXT NOT NULL,
    pricing_dimensions INTEGER NOT NULL,
    platform_fee_rate REAL NOT NULL,
    created_ms        INTEGER NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_billing_receipts_service_job
ON billing_receipts(service, job_id);

CREATE TABLE IF NOT EXISTS billing_receipt_lines (
    line_id            TEXT PRIMARY KEY,
    receipt_id         TEXT NOT NULL,
    service            TEXT NOT NULL,
    job_id             TEXT NOT NULL,
    idempotency_key    TEXT NOT NULL,
    node_id            TEXT NOT NULL,
    provider_owner     TEXT NOT NULL,
    measured_seconds   REAL NOT NULL,
    resource_class     TEXT NOT NULL,
    resource_units     REAL NOT NULL,
    mesh_seconds       REAL NOT NULL,
    gross_earnings     REAL NOT NULL,
    platform_fee       REAL NOT NULL,
    total_charged      REAL NOT NULL,
    claim_tier         TEXT NOT NULL,
    consumer_token_id  TEXT,
    created_ms         INTEGER NOT NULL,
    UNIQUE(receipt_id, idempotency_key)
);
CREATE INDEX IF NOT EXISTS idx_billing_lines_job
ON billing_receipt_lines(service, job_id);
"""


def platform_fee_rate_from_env() -> float:
    raw = (os.environ.get(PLATFORM_FEE_RATE_ENV) or "").strip()
    if not raw:
        return DEFAULT_PLATFORM_FEE_RATE
    try:
        val = float(raw)
    except ValueError:
        return DEFAULT_PLATFORM_FEE_RATE
    return max(0.0, val)


class MeshSecondMeter:
    """Normalize all services into one unit: resource-weighted node seconds."""

    unit = MESH_SECOND
    pricing_dimensions = 1

    def measure(self, measured_seconds: float, resource_class: str = "standard",
                resource_units: float = 1.0) -> float:
        weight = RESOURCE_CLASS_WEIGHTS.get(resource_class, RESOURCE_CLASS_WEIGHTS["standard"])
        return max(0.0, measured_seconds) * max(0.0, resource_units) * weight

    def batch(self, elapsed_s: float, cpu_pct: float) -> float:
        return self.measure(elapsed_s, "batch", cpu_pct)


@dataclass(frozen=True)
class ReceiptLine:
    line_id: str
    node_id: str
    provider_owner: str
    measured_seconds: float
    resource_class: str
    resource_units: float
    mesh_seconds: float
    gross_earnings: float
    platform_fee: float
    total_charged: float
    claim_tier: str = "measured"


class BillingManager:
    """Append-only receipts plus balanced ledger movements."""

    def __init__(self, store, ledger: ContributionLedger,
                 platform_fee_rate: Optional[float] = None,
                 platform_owner: str = PLATFORM_OWNER) -> None:
        self.store = store
        self.ledger = ledger
        self.platform_fee_rate = (
            platform_fee_rate if platform_fee_rate is not None
            else platform_fee_rate_from_env()
        )
        self.platform_owner = platform_owner
        self.meter = MeshSecondMeter()
        self._ensure_table()

    def _ensure_table(self) -> None:
        with self.store._lock:
            self.store._conn.executescript(_DDL)
            cols = {
                r["name"] for r in self.store._conn.execute(
                    "PRAGMA table_info(billing_receipts)"
                ).fetchall()
            }
            if "capability_token_id" not in cols:
                self.store._conn.execute(
                    "ALTER TABLE billing_receipts ADD COLUMN capability_token_id TEXT"
                )
            line_cols = {
                r["name"] for r in self.store._conn.execute(
                    "PRAGMA table_info(billing_receipt_lines)"
                ).fetchall()
            }
            if "consumer_token_id" not in line_cols:
                self.store._conn.execute(
                    "ALTER TABLE billing_receipt_lines ADD COLUMN consumer_token_id TEXT"
                )
            self.store._conn.commit()

    @staticmethod
    def receipt_id(service: str, job_id: str) -> str:
        return f"{service}:{job_id}"

    def _ensure_receipt(self, service: str, job_id: str,
                        consumer_owner: str,
                        capability_token_id: Optional[str] = None) -> str:
        rid = self.receipt_id(service, job_id)
        with self.store._lock:
            self.store._conn.execute(
                "INSERT OR IGNORE INTO billing_receipts("
                "receipt_id,service,job_id,consumer_owner,capability_token_id,unit,"
                "pricing_dimensions,platform_fee_rate,created_ms"
                ") VALUES(?,?,?,?,?,?,?,?,?)",
                (rid, service, job_id, consumer_owner, capability_token_id, MESH_SECOND, 1,
                 self.platform_fee_rate, now_ms()),
            )
            if capability_token_id:
                self.store._conn.execute(
                    "UPDATE billing_receipts SET capability_token_id=? "
                    "WHERE receipt_id=? AND capability_token_id IS NULL",
                    (capability_token_id, rid),
                )
            self.store._conn.commit()
        return rid

    def record_charge(
        self,
        service: str,
        job_id: str,
        consumer_owner: str,
        node_id: str,
        provider_owner: str,
        measured_seconds: float,
        *,
        idempotency_key: str,
        resource_class: str = "standard",
        resource_units: float = 1.0,
        detail: str = "",
        consumer_token_id: Optional[str] = None,
        earning_recorder: Optional[Callable[[str, str, float, str], object]] = None,
    ) -> dict:
        """Record one accepted unit/request exactly once and return its line.

        ``idempotency_key`` is the retry/speculation guard: repeated calls for the
        same accepted unit return the existing receipt line and do not touch the
        ledger again.
        """
        idem = (idempotency_key or "").strip()
        if not idem:
            raise ValueError("billing idempotency_key is required")
        rid = self._ensure_receipt(service, job_id, consumer_owner, consumer_token_id)
        line_id = f"{rid}:{idem}"
        mesh_seconds = self.meter.measure(measured_seconds, resource_class, resource_units)
        fee = mesh_seconds * self.platform_fee_rate
        total = mesh_seconds + fee
        ledger_detail = f"{detail} token:{consumer_token_id}" if consumer_token_id else detail
        inserted = False
        with self.store._lock:
            cur = self.store._conn.execute(
                "INSERT OR IGNORE INTO billing_receipt_lines("
                "line_id,receipt_id,service,job_id,idempotency_key,node_id,"
                "provider_owner,measured_seconds,resource_class,resource_units,"
                "mesh_seconds,gross_earnings,platform_fee,total_charged,"
                "claim_tier,consumer_token_id,created_ms) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (line_id, rid, service, job_id, idem, node_id, provider_owner,
                 max(0.0, measured_seconds), resource_class, max(0.0, resource_units),
                 mesh_seconds, mesh_seconds, fee, total, "measured", consumer_token_id, now_ms()),
            )
            inserted = cur.rowcount == 1
            self.store._conn.commit()
        if inserted:
            # Gross donor earning is explicit.  Public-tier held-back escrow can
            # wrap this via ``earning_recorder``; either way the amount is gross.
            if earning_recorder is not None:
                earning_recorder(node_id, provider_owner, mesh_seconds, ledger_detail)
            else:
                self.ledger.record_donation(node_id, provider_owner, mesh_seconds,
                                            detail=ledger_detail)
            self.ledger.store.add_ledger_idempotent(LedgerEntry(
                id=f"{line_id}:consumer:gross", node_id=node_id,
                owner=consumer_owner, kind="consumed", compute_units=mesh_seconds,
                ts_ms=now_ms(), detail=ledger_detail))
            if fee > 0.0:
                self.ledger.store.add_ledger_idempotent(LedgerEntry(
                    id=f"{line_id}:platform:credit", node_id=node_id,
                    owner=self.platform_owner, kind="donated", compute_units=fee,
                    ts_ms=now_ms(), detail=f"platform-fee:{ledger_detail}"))
                self.ledger.store.add_ledger_idempotent(LedgerEntry(
                    id=f"{line_id}:platform:debit", node_id=node_id,
                    owner=consumer_owner, kind="consumed", compute_units=fee,
                    ts_ms=now_ms(), detail=f"platform-fee:{ledger_detail}"))
            if consumer_token_id:
                self.store.add_capability_token_spend(consumer_token_id, total)
        return self._line_to_dict(self._get_line(line_id))

    def _get_line(self, line_id: str):
        with self.store._lock:
            return self.store._conn.execute(
                "SELECT * FROM billing_receipt_lines WHERE line_id=?", (line_id,)
            ).fetchone()

    @staticmethod
    def _line_to_dict(row) -> dict:
        return {
            "line_id": row["line_id"],
            "node_id": row["node_id"],
            "provider_owner": row["provider_owner"],
            "measured_seconds": row["measured_seconds"],
            "resource_class": row["resource_class"],
            "resource_units": row["resource_units"],
            "mesh_seconds": row["mesh_seconds"],
            "gross_earnings": row["gross_earnings"],
            "platform_fee": row["platform_fee"],
            "total_charged": row["total_charged"],
            "claim_tier": row["claim_tier"],
            "consumer_token_id": row["consumer_token_id"],
        }

    def receipt(self, service: str, job_id: str) -> Optional[dict]:
        rid = self.receipt_id(service, job_id)
        with self.store._lock:
            head = self.store._conn.execute(
                "SELECT * FROM billing_receipts WHERE receipt_id=?", (rid,)
            ).fetchone()
            rows = self.store._conn.execute(
                "SELECT * FROM billing_receipt_lines WHERE receipt_id=? "
                "ORDER BY created_ms,line_id", (rid,)
            ).fetchall()
        if head is None:
            return None
        lines = [self._line_to_dict(r) for r in rows]
        gross = sum(float(l["gross_earnings"]) for l in lines)
        fee = sum(float(l["platform_fee"]) for l in lines)
        total = sum(float(l["total_charged"]) for l in lines)
        return {
            "receipt_id": rid,
            "service": head["service"],
            "job_id": head["job_id"],
            "owner": head["consumer_owner"],
            "capability_token_id": head["capability_token_id"],
            "unit": head["unit"],
            "pricing_dimensions": head["pricing_dimensions"],
            "claim_tier": "measured",
            "fee_rate": head["platform_fee_rate"],
            "fee_rate_tier": "calibrated projection",
            "lines": lines,
            "totals": {
                "gross_earnings": gross,
                "platform_fee": fee,
                "total_charged": total,
                "mesh_seconds": gross,
            },
            "platform_fee_line_item": {
                "owner": PLATFORM_OWNER,
                "amount": fee,
                "unit": MESH_SECOND,
                "claim_tier": "measured",
            },
        }

    def live_meter(self, service: str, job_id: str, owner: str,
                   running_units: int = 0) -> dict:
        receipt = self.receipt(service, job_id)
        measured = 0.0 if receipt is None else receipt["totals"]["total_charged"]
        return {
            "unit": MESH_SECOND,
            "pricing_dimensions": 1,
            "owner": owner,
            "measured_total": measured,
            "measured_total_tier": "measured",
            "running_units": running_units,
            "running_units_tier": "measured",
            "projected_additional": 0.0,
            "projected_additional_tier": "calibrated projection",
        }
