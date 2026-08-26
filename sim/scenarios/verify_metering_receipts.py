"""Acceptance gate: mesh-second metering, receipts, retries and caps.

Honest scope: this is an in-process billing/metering gate.  It exercises the real
BillingManager, ContributionLedger, SpendCapManager and BatchManager dispatch loop,
but it does not boot real agent processes.  The existing spend-cap gate covers the
multi-process hard-stop path.

Exit 0 = PASS.  Exit 1 = FAIL.
"""
from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from meshpool.batch.engine import BatchManager  # noqa: E402
from meshpool.batch.functions import run_function  # noqa: E402
from meshpool.common.billing import BillingManager, MESH_SECOND, MeshSecondMeter  # noqa: E402
from meshpool.common.db import Store  # noqa: E402
from meshpool.common.models import Capability, DonationPolicy, LedgerEntry, Node, NodeStatus, new_id  # noqa: E402
from meshpool.common.trust import ContributionLedger, TrustEngine  # noqa: E402
from meshpool.controlplane.spend import SpendCapManager  # noqa: E402


class _Events:
    def publish(self, *_, **__):
        pass


class _Registry:
    def __init__(self, nodes):
        self.nodes = nodes

    def schedulable_nodes(self):
        return [n for n in self.nodes if n.status is NodeStatus.ONLINE]

    def online(self):
        return self.schedulable_nodes()

    def list(self):
        return list(self.nodes)

    def get(self, node_id: str):
        return next((n for n in self.nodes if n.id == node_id), None)


class _Transport:
    def __init__(self, compute_ms: int = 1000, fail_first_units: set[str] | None = None):
        self.compute_ms = compute_ms
        self.fail_first_units = set(fail_first_units or set())
        self.calls_by_unit: dict[str, int] = {}

    async def call(self, node_id, op, payload, timeout=None):
        unit_id = payload["unit_id"]
        self.calls_by_unit[unit_id] = self.calls_by_unit.get(unit_id, 0) + 1
        if unit_id in self.fail_first_units and self.calls_by_unit[unit_id] == 1:
            raise RuntimeError("intentional first-attempt failure")
        return {
            "result": run_function(payload["func"], payload["payload"]),
            "compute_ms": self.compute_ms,
        }


def _node(node_id: str, cpu_slots: int = 1) -> Node:
    return Node(
        id=node_id,
        name=node_id,
        url=f"http://127.0.0.1/{node_id}",
        owner=f"dev:{node_id}",
        status=NodeStatus.ONLINE,
        capability=Capability(cpu_cores=cpu_slots, ram_mb=4096),
        policy=DonationPolicy(cpu_pct=100, ram_mb=4096),
    )


async def _run_until_stable(mgr: BatchManager, job_id: str, max_s: float = 5.0) -> None:
    deadline = time.time() + max_s
    while time.time() < deadline:
        await mgr.dispatch_once()
        job = mgr.get(job_id)
        if job and job.status.startswith("completed"):
            return
        await asyncio.sleep(0.02)
    return


def _manager(store: Store, nodes: list[Node], transport: _Transport,
             default_cap: float | None = None) -> tuple[BatchManager, ContributionLedger, BillingManager, SpendCapManager]:
    ledger = ContributionLedger(store)
    trust = TrustEngine(store, tier=1)
    billing = BillingManager(store, ledger, platform_fee_rate=0.10)
    caps = SpendCapManager(store, default_cap=default_cap)
    mgr = BatchManager(_Registry(nodes), store, ledger, trust, _Events(), transport)
    mgr.billing = billing
    mgr.spend_caps = caps
    return mgr, ledger, billing, caps


def _totals(store: Store) -> tuple[float, float]:
    rows = store.ledger_totals_by_owner()
    return (
        sum(float(v["donated"]) for v in rows.values()),
        sum(float(v["consumed"]) for v in rows.values()),
    )


def _assert_receipt_reconciles(store: Store, receipt: dict) -> None:
    rows = store.ledger_totals_by_owner()
    gross = receipt["totals"]["gross_earnings"]
    fee = receipt["totals"]["platform_fee"]
    total = receipt["totals"]["total_charged"]
    assert sum(float(v["consumed"]) for v in rows.values()) == total, rows
    assert sum(float(v["donated"]) for v in rows.values()) == total, rows
    provider_sum = sum(
        float(rows[line["provider_owner"]]["donated"])
        for line in receipt["lines"]
    )
    assert provider_sum == gross, (provider_sum, receipt)
    assert rows["meshpool:platform-fee"]["donated"] == fee, rows


def main() -> int:
    ok = True
    try:
        print("== test 1: the same workload costs the same via every entry path ==")
        meter = MeshSecondMeter()
        costs = {
            service: meter.measure(2.0, service, 10.0)
            for service in ("batch", "functions", "inference", "hosting", "storage")
        }
        assert len(set(costs.values())) == 1, costs
        assert meter.unit == MESH_SECOND and meter.pricing_dimensions == 1
        print(f"  PASS: one unit ({MESH_SECOND}), one price dimension, costs={costs}")

        print("\n== test 2: a retried unit is billed exactly once ==")
        store = Store(":memory:")
        node = _node("node1")
        node_retry = _node("node-retry")
        transport = _Transport(compute_ms=1000)
        mgr, ledger, billing, _ = _manager(store, [node, node_retry], transport)
        job = mgr.submit("retry", "square", [{"x": 7}], cpu_pct=10, ram_mb=64,
                         owner="cons:alice")
        unit_id = job.units[0].id
        transport.fail_first_units.add(unit_id)
        asyncio.run(_run_until_stable(mgr, job.id))
        receipt = billing.receipt("batch", job.id)
        assert transport.calls_by_unit[unit_id] == 2, transport.calls_by_unit
        assert receipt is not None and len(receipt["lines"]) == 1, receipt
        assert receipt["totals"]["gross_earnings"] == 10.0, receipt
        assert ledger.credits("cons:alice") == -11.0, store.ledger_totals_by_owner()
        print("  PASS: first attempt failed, retry succeeded, one receipt line recorded")

        print("\n== test 3: hitting a cap stops work rather than warning ==")
        store2 = Store(":memory:")
        node2 = _node("node2")
        mgr2, _, billing2, caps = _manager(
            store2, [node2], _Transport(compute_ms=1000), default_cap=11.0)
        job2 = mgr2.submit("cap", "square", [{"x": i} for i in range(3)],
                           cpu_pct=10, ram_mb=64, owner="cons:bob")
        asyncio.run(_run_until_stable(mgr2, job2.id, max_s=1.0))
        done = sum(1 for u in job2.units if u.status.value == "done")
        pending = sum(1 for u in job2.units if u.status.value == "pending")
        assert caps.exceeded("cons:bob") is True, caps.status("cons:bob")
        assert done == 1 and pending == 2, job2.model_dump(mode="json")
        assert billing2.receipt("batch", job2.id)["totals"]["total_charged"] == 11.0
        print("  PASS: cap reached after first unit; remaining work stayed pending")

        print("\n== test 4: receipt totals reconcile exactly with ledger movements ==")
        _assert_receipt_reconciles(store, receipt)
        donated, consumed = _totals(store)
        assert donated == consumed == receipt["totals"]["total_charged"], (
            donated, consumed, receipt)
        print(f"  PASS: donated={donated} consumed={consumed} receipt={receipt['totals']}")

        print("\n== test 5: NON-VACUITY reconciliation detects forged ledger movement ==")
        store.add_ledger(LedgerEntry(
            id=new_id("led"), node_id="mutant", owner="dev:mutant",
            kind="donated", compute_units=1.0, detail="MUTATION: forged credit"))
        try:
            _assert_receipt_reconciles(store, receipt)
        except AssertionError:
            print("  PASS: forged donation breaks receipt/ledger reconciliation")
        else:
            raise AssertionError("forged unbalanced donation was not detected")

        print("\nALL METERING/RECEIPT CHECKS PASSED")
    except Exception as exc:  # noqa: BLE001
        ok = False
        print(f"\nFAILED: {type(exc).__name__}: {exc}")
        import traceback
        traceback.print_exc()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
