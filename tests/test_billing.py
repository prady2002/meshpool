from __future__ import annotations

import pytest

from meshpool.common.billing import BillingManager, MESH_SECOND, MeshSecondMeter
from meshpool.common.trust import ContributionLedger


def test_mesh_second_is_one_dimension_across_services():
    meter = MeshSecondMeter()
    costs = [
        meter.measure(2.5, service, 10.0)
        for service in ("batch", "functions", "inference", "hosting", "storage")
    ]
    assert meter.unit == MESH_SECOND
    assert meter.pricing_dimensions == 1
    assert costs == [25.0] * 5


def test_receipt_records_gross_fee_and_balanced_ledger(tmp_store):
    ledger = ContributionLedger(tmp_store)
    billing = BillingManager(tmp_store, ledger, platform_fee_rate=0.10)

    billing.record_charge("batch", "job1", "cons:alice", "node1", "dev:node1",
                          measured_seconds=2.0, resource_class="batch",
                          resource_units=10.0, idempotency_key="unit1",
                          detail="batch:job1")
    receipt = billing.receipt("batch", "job1")

    assert receipt["unit"] == MESH_SECOND
    assert receipt["pricing_dimensions"] == 1
    assert receipt["totals"]["gross_earnings"] == pytest.approx(20.0)
    assert receipt["totals"]["platform_fee"] == pytest.approx(2.0)
    assert receipt["totals"]["total_charged"] == pytest.approx(22.0)

    totals = tmp_store.ledger_totals_by_owner()
    assert totals["dev:node1"]["donated"] == pytest.approx(20.0)
    assert totals["meshpool:platform-fee"]["donated"] == pytest.approx(2.0)
    assert totals["cons:alice"]["consumed"] == pytest.approx(22.0)
    assert sum(v["donated"] for v in totals.values()) == pytest.approx(
        sum(v["consumed"] for v in totals.values()))


def test_duplicate_accepted_result_bills_same_unit_once(tmp_store):
    ledger = ContributionLedger(tmp_store)
    billing = BillingManager(tmp_store, ledger, platform_fee_rate=0.10)

    for _ in range(2):
        billing.record_charge("batch", "job1", "cons:alice", "node1", "dev:node1",
                              measured_seconds=1.0, resource_class="batch",
                              resource_units=10.0, idempotency_key="unit1",
                              detail="batch:job1")

    receipt = billing.receipt("batch", "job1")
    assert len(receipt["lines"]) == 1
    assert receipt["totals"]["total_charged"] == pytest.approx(11.0)
    assert ledger.store.ledger_totals_by_owner()["cons:alice"]["consumed"] == pytest.approx(11.0)


def test_missing_idempotency_key_fails_loudly(tmp_store):
    ledger = ContributionLedger(tmp_store)
    billing = BillingManager(tmp_store, ledger, platform_fee_rate=0.10)

    with pytest.raises(ValueError, match="idempotency_key"):
        billing.record_charge("batch", "job1", "cons:alice", "node1", "dev:node1",
                              measured_seconds=1.0, resource_class="batch",
                              resource_units=10.0, idempotency_key="",
                              detail="batch:job1")

    assert billing.receipt("batch", "job1") is None
    assert tmp_store.ledger_totals_by_owner() == {}


def test_live_meter_labels_measured_and_projection_tiers(tmp_store):
    ledger = ContributionLedger(tmp_store)
    billing = BillingManager(tmp_store, ledger, platform_fee_rate=0.0)
    billing.record_charge("batch", "job1", "cons:alice", "node1", "dev:node1",
                          measured_seconds=1.0, resource_class="batch",
                          resource_units=5.0, idempotency_key="unit1",
                          detail="batch:job1")

    meter = billing.live_meter("batch", "job1", "cons:alice", running_units=2)

    assert meter["unit"] == MESH_SECOND
    assert meter["pricing_dimensions"] == 1
    assert meter["measured_total"] == pytest.approx(5.0)
    assert meter["measured_total_tier"] == "measured"
    assert meter["running_units_tier"] == "measured"
    assert meter["projected_additional_tier"] == "calibrated projection"
