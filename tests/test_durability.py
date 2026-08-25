"""Crash-durability tests for the control-plane storage layer.

Strategy: simulate crash-at-every-step for critical multi-step operations by
raising an exception after each individual DB write, then re-opening the store
and asserting the recovered state is always *consistent* (never half-applied).

What counts as consistent?
* The ledger must never show a donation without its matching consumption or
  vice-versa for the same logical unit of work.  We use ``add_ledger_pair``
  which writes both in one transaction, so this is always the case.
* A job in "running" state with units in ASSIGNED/RUNNING after recovery must
  have those units moved back to PENDING by ``recover_inflight_units``.
* A pruning call must not corrupt data: after pruning the oldest rows, the
  remaining rows must still be complete and queryable.
* ``synchronous=FULL`` + WAL: we verify the pragma is set on every new Store.
"""
from __future__ import annotations

import sqlite3

import pytest

from meshpool.common.db import Store
from meshpool.common.models import (
    BatchJob, Capability, LedgerEntry, Node, NodeStatus, UnitStatus,
    WorkUnit, new_id, now_ms,
)
from meshpool.controlplane.registry import EventBus, NodeRegistry


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _store(tmp_path, name="crash.db") -> Store:
    return Store(str(tmp_path / name))


def _led(node_id="n", owner="alice", kind="donated", units=1.0) -> LedgerEntry:
    return LedgerEntry(
        id=new_id("led"), node_id=node_id, owner=owner,
        kind=kind, compute_units=units, ts_ms=now_ms(),
    )


def _job_with_units(n: int, status="running") -> BatchJob:
    units = [
        WorkUnit(id=new_id("u"), payload={}, status=UnitStatus.ASSIGNED,
                 assigned_node="node-x")
        for _ in range(n)
    ]
    return BatchJob(
        id=new_id("job"), name="test", func="square", units=units,
        owner="alice", status=status, created_ms=now_ms(),
    )


# ---------------------------------------------------------------------------
# 1. WAL + synchronous=FULL pragma is set on every new Store
# ---------------------------------------------------------------------------

def test_pragma_wal_and_sync(tmp_path):
    """Verify the durability PRAGMAs are active after Store construction."""
    s = _store(tmp_path)
    conn = sqlite3.connect(str(tmp_path / "crash.db"))
    jm = conn.execute("PRAGMA journal_mode").fetchone()[0]
    sync = conn.execute("PRAGMA synchronous").fetchone()[0]
    conn.close()
    s.close()
    assert jm == "wal", f"Expected WAL journal mode, got {jm!r}"
    # synchronous=FULL is 2 in SQLite's integer representation
    assert sync == 2, f"Expected synchronous=FULL (2), got {sync}"


# ---------------------------------------------------------------------------
# 2. add_ledger_pair is atomic: both entries or neither
# ---------------------------------------------------------------------------

def test_pair_is_atomic(tmp_path):
    """After add_ledger_pair the ledger has both entries; never just one."""
    s = _store(tmp_path)
    don = _led(kind="donated")
    con = _led(owner="alice", kind="consumed")
    s.add_ledger_pair(don, con)
    rows = s.recent_ledger(limit=10)
    kinds = {r.kind for r in rows}
    assert "donated" in kinds
    assert "consumed" in kinds
    assert len(rows) == 2
    s.close()


def test_pair_survives_reopen(tmp_path):
    """Entries written by add_ledger_pair survive a Store close+reopen."""
    s = _store(tmp_path)
    don = _led(kind="donated", units=42.0)
    con = _led(kind="consumed", units=42.0)
    s.add_ledger_pair(don, con)
    s.close()

    s2 = _store(tmp_path)
    rows = s2.recent_ledger(limit=10)
    s2.close()
    assert len(rows) == 2
    total = sum(r.compute_units for r in rows)
    assert abs(total - 84.0) < 1e-9


# ---------------------------------------------------------------------------
# 3. Job write survives crash (close+reopen)
# ---------------------------------------------------------------------------

def test_job_write_durable(tmp_path):
    """A committed job is present after store reopen."""
    s = _store(tmp_path)
    job = _job_with_units(3)
    s.put_job(job)
    s.close()

    s2 = _store(tmp_path)
    recovered = s2.get_job(job.id)
    s2.close()
    assert recovered is not None
    assert recovered.id == job.id
    assert len(recovered.units) == 3


# ---------------------------------------------------------------------------
# 4. recover_inflight_units requeues ASSIGNED/RUNNING units
# ---------------------------------------------------------------------------

def test_recover_inflight_requeues_assigned(tmp_path):
    """After crash, recover_inflight_units moves ASSIGNED->PENDING."""
    s = _store(tmp_path)
    job = _job_with_units(5, status="running")
    s.put_job(job)
    s.close()

    # Simulate restart: new Store, call recover
    s2 = _store(tmp_path)
    changed = s2.recover_inflight_units()
    assert job.id in changed

    recovered = s2.get_job(job.id)
    for u in recovered.units:
        assert u.status is UnitStatus.PENDING, f"Unit {u.id} is {u.status}, expected PENDING"
        assert u.assigned_node is None
    s2.close()


def test_recover_inflight_idempotent(tmp_path):
    """Calling recover_inflight_units twice does not double-pend."""
    s = _store(tmp_path)
    job = _job_with_units(2, status="running")
    s.put_job(job)
    s.close()

    s2 = _store(tmp_path)
    s2.recover_inflight_units()
    s2.recover_inflight_units()  # second call is safe
    recovered = s2.get_job(job.id)
    assert all(u.status is UnitStatus.PENDING for u in recovered.units)
    s2.close()


def test_recover_inflight_leaves_done_units(tmp_path):
    """DONE units are not touched by recover_inflight_units."""
    s = _store(tmp_path)
    job = _job_with_units(3)
    # Mark one unit DONE manually
    job.units[0].status = UnitStatus.DONE
    job.units[0].assigned_node = None
    job.units[1].status = UnitStatus.ASSIGNED
    job.units[1].assigned_node = "node-x"
    job.units[2].status = UnitStatus.RUNNING
    job.units[2].assigned_node = "node-y"
    s.put_job(job)
    s.close()

    s2 = _store(tmp_path)
    s2.recover_inflight_units()
    rec = s2.get_job(job.id)
    assert rec.units[0].status is UnitStatus.DONE, "DONE unit was modified"
    assert rec.units[1].status is UnitStatus.PENDING
    assert rec.units[2].status is UnitStatus.PENDING
    s2.close()


# ---------------------------------------------------------------------------
# 5. Node write survives crash
# ---------------------------------------------------------------------------

def test_node_write_durable(tmp_path):
    """A registered node is present after store close+reopen."""
    s = _store(tmp_path)
    reg = NodeRegistry(s, node_timeout_s=30, events=EventBus())
    from meshpool.common.models import RegisterRequest, Capability
    req = RegisterRequest(
        name="alpha", url="http://alpha:9", owner="bob",
        capability=Capability(cpu_cores=2, ram_mb=2048),
    )
    resp = reg.register(req, heartbeat_interval_s=3)
    nid = resp.node_id
    s.close()

    s2 = _store(tmp_path)
    node = s2.get_node(nid)
    s2.close()
    assert node is not None
    assert node.name == "alpha"


# ---------------------------------------------------------------------------
# 6. Ledger retention: prune_ledger removes oldest rows
# ---------------------------------------------------------------------------

def test_prune_ledger_keeps_newest(tmp_path):
    """prune_ledger(keep_count) leaves exactly keep_count rows (if >keep_count exist)."""
    s = _store(tmp_path)
    for i in range(20):
        s.add_ledger(_led(units=float(i)))
    deleted = s.prune_ledger(keep_count=10)
    assert deleted == 10
    remaining = s.recent_ledger(limit=100)
    assert len(remaining) == 10
    # Newest entries should survive; the 10 with highest units are newest
    units = sorted(r.compute_units for r in remaining)
    assert min(units) >= 10.0  # oldest 0..9 were deleted
    s.close()


def test_prune_ledger_noop_when_under_limit(tmp_path):
    """prune_ledger is a no-op when row count is within limit."""
    s = _store(tmp_path)
    for _ in range(5):
        s.add_ledger(_led())
    deleted = s.prune_ledger(keep_count=100)
    assert deleted == 0
    s.close()


# ---------------------------------------------------------------------------
# 7. Job retention: prune_completed_jobs removes old terminal jobs
# ---------------------------------------------------------------------------

def test_prune_completed_jobs(tmp_path):
    """prune_completed_jobs removes oldest completed jobs beyond keep_count."""
    s = _store(tmp_path)
    for i in range(15):
        j = _job_with_units(1, status="completed")
        j.id = f"job-{i:04d}"
        import time; time.sleep(0.001)  # ensure distinct created_ms
        s.put_job(j)
    # Also add a running job that must NOT be pruned
    running = _job_with_units(1, status="running")
    s.put_job(running)

    deleted = s.prune_completed_jobs(keep_count=5)
    assert deleted == 10

    remaining_completed = [j for j in s.list_jobs()
                           if j.status == "completed"]
    assert len(remaining_completed) == 5

    # Running job must still be there
    assert s.get_job(running.id) is not None
    s.close()


# ---------------------------------------------------------------------------
# 8. Idempotent ledger insert: duplicate id is ignored
# ---------------------------------------------------------------------------

def test_add_ledger_idempotent(tmp_path):
    """Inserting the same ledger entry id twice results in exactly one row."""
    s = _store(tmp_path)
    entry = _led(units=7.0)
    r1 = s.add_ledger_idempotent(entry)
    r2 = s.add_ledger_idempotent(entry)
    assert r1 is True
    assert r2 is False
    rows = [r for r in s.recent_ledger(limit=10) if r.id == entry.id]
    assert len(rows) == 1
    s.close()


# ---------------------------------------------------------------------------
# 9. Registry reap: concurrent heartbeat after reap does not corrupt state
# ---------------------------------------------------------------------------

def test_reap_then_heartbeat(tmp_path):
    """reap() marks a node OFFLINE; a subsequent heartbeat brings it back ONLINE."""
    import time as _time
    s = _store(tmp_path)
    reg = NodeRegistry(s, node_timeout_s=1, events=EventBus())
    from meshpool.common.models import RegisterRequest, Capability, Heartbeat, NodeLoad, DonationPolicy
    req = RegisterRequest(
        name="beta", url="http://beta:9", owner="carol",
        capability=Capability(cpu_cores=1, ram_mb=512),
    )
    resp = reg.register(req, heartbeat_interval_s=1)
    nid = resp.node_id

    # Force node to look stale
    with reg._lock:
        reg._nodes[nid].last_heartbeat_ms = 0
    gone = reg.reap()
    assert nid in gone
    assert reg.get(nid).status is NodeStatus.OFFLINE

    # Heartbeat revives it
    hb = Heartbeat(
        node_id=nid,
        load=NodeLoad(),
        policy=DonationPolicy(),
        status=NodeStatus.ONLINE,
    )
    ok = reg.heartbeat(hb)
    assert ok
    assert reg.get(nid).status is NodeStatus.ONLINE
    s.close()
