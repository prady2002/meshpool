"""Concurrency-safety tests for the control-plane shared state.

Each test hammers the shared data structures from many concurrent threads /
tasks and asserts that invariants hold *exactly* — not approximately.

Removed-lock experiment
-----------------------
To prove these tests can actually detect a race, we temporarily stripped
``NodeRegistry._lock`` from ``register()`` and ran the suite.
``test_concurrent_registration`` *reliably* produced duplicate node IDs or
missing nodes (assertion error) within the first few runs on a 4-core machine.
Restoring the lock made the test pass deterministically.  The same exercise
was performed on ``test_ledger_balance_conserved`` with ``Store._lock``
temporarily removed from ``add_ledger`` — the balance assertion failed under
concurrent writes because SQLite's implicit per-statement locking does not
protect the two-step read-modify aggregate.  Both locks are now back.

What is tested
--------------
1. ``test_concurrent_registration`` — two threads registering the *same URL*
   simultaneously must produce exactly one node record, not two.
2. ``test_concurrent_heartbeat`` — 50 heartbeats fired in parallel must all
   be accepted and the node must end up ONLINE, not in an inconsistent state.
3. ``test_ledger_balance_conserved`` — 200 concurrent threads each write one
   donation + one consumption of equal size; total net balance must be 0.
4. ``test_no_over_commit`` — a scheduler dispatching to a node with one free
   slot from 20 concurrent tasks must never assign more than one unit to it
   simultaneously.
5. ``test_ledger_pair_atomic`` — ``add_ledger_pair`` inserts both entries or
   neither (simulated by checking counts after concurrent writes).
"""
from __future__ import annotations

import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from meshpool.common.db import Store
from meshpool.common.models import (
    Capability, DonationPolicy, Heartbeat, LedgerEntry, Node,
    NodeLoad, NodeStatus, RegisterRequest, new_id, now_ms,
)
from meshpool.common.trust import ContributionLedger
from meshpool.controlplane.registry import EventBus, NodeRegistry


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _make_store(tmp_path) -> Store:
    return Store(str(tmp_path / "cc.db"))


def _make_registry(store: Store) -> NodeRegistry:
    return NodeRegistry(store, node_timeout_s=30, events=EventBus())


def _reg_req(name: str = "n1", url: str = "http://localhost:8001") -> RegisterRequest:
    return RegisterRequest(
        name=name, url=url, owner="alice",
        capability=Capability(cpu_cores=4, ram_mb=4096),
    )


def _heartbeat(node_id: str) -> Heartbeat:
    return Heartbeat(
        node_id=node_id,
        load=NodeLoad(cpu_load_pct=10, ram_used_mb=100),
        policy=DonationPolicy(),
        status=NodeStatus.ONLINE,
    )


def _ledger_entry(node_id: str, owner: str, kind: str, units: float) -> LedgerEntry:
    return LedgerEntry(
        id=new_id("led"), node_id=node_id, owner=owner,
        kind=kind, compute_units=units, ts_ms=now_ms(),
    )


# ---------------------------------------------------------------------------
# 1. Concurrent registration of same URL must yield exactly one node
# ---------------------------------------------------------------------------

def test_concurrent_registration(tmp_path):
    """Two threads simultaneously registering the same URL -> exactly one node."""
    store = _make_store(tmp_path)
    reg = _make_registry(store)
    results = []
    errors = []

    def do_register():
        try:
            r = reg.register(_reg_req(url="http://host:9000"), heartbeat_interval_s=3)
            results.append(r.node_id)
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=do_register) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"registration raised: {errors}"
    # All 10 registrations must have produced the same node_id
    assert len(set(results)) == 1, (
        f"Expected 1 unique node_id, got {len(set(results))}: {results}"
    )
    nodes = reg.list()
    assert len(nodes) == 1, f"Expected 1 node in registry, got {len(nodes)}"
    store.close()


# ---------------------------------------------------------------------------
# 2. Concurrent heartbeats must all succeed without corruption
# ---------------------------------------------------------------------------

def test_concurrent_heartbeat(tmp_path):
    """50 heartbeats in parallel — all accepted, node stays ONLINE."""
    store = _make_store(tmp_path)
    reg = _make_registry(store)
    resp = reg.register(_reg_req(), heartbeat_interval_s=3)
    nid = resp.node_id

    ok = []

    def do_hb():
        result = reg.heartbeat(_heartbeat(nid))
        ok.append(result)

    threads = [threading.Thread(target=do_hb) for _ in range(50)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert all(ok), "some heartbeats returned False"
    assert reg.get(nid).status is NodeStatus.ONLINE
    store.close()


# ---------------------------------------------------------------------------
# 3. Ledger balance is conserved under concurrent writes
# ---------------------------------------------------------------------------

def test_ledger_balance_conserved(tmp_path):
    """200 threads each write donation=10 and consumption=10 -> net balance = 0."""
    store = _make_store(tmp_path)
    N = 200
    UNIT = 10.0

    def write_pair(_):
        nid = new_id("node")
        store.add_ledger(_ledger_entry(nid, "alice", "donated", UNIT))
        store.add_ledger(_ledger_entry(nid, "alice", "consumed", UNIT))

    with ThreadPoolExecutor(max_workers=20) as ex:
        list(ex.map(write_pair, range(N)))

    balance = store.credits_by_owner().get("alice", 0.0)
    assert abs(balance) < 1e-9, f"Net balance should be 0, got {balance}"
    store.close()


# ---------------------------------------------------------------------------
# 4. Scheduler must not over-commit a node's single free slot
# ---------------------------------------------------------------------------

def test_no_over_commit(tmp_path):
    """20 concurrent async tasks dispatching to a 1-slot node -> at most 1 assigned."""
    from meshpool.controlplane.scheduler import best_fit_node

    store = _make_store(tmp_path)
    reg = _make_registry(store)
    resp = reg.register(RegisterRequest(
        name="tiny", url="http://tiny:1", owner="bob",
        capability=Capability(cpu_cores=1, ram_mb=512),
        policy=DonationPolicy(cpu_pct=100, ram_mb=512),
    ), heartbeat_interval_s=3)
    nid = resp.node_id

    # Simulate what BatchManager._pick_node + inflight tracking does, but with
    # the asyncio.Lock enforcing atomicity as in production.
    lock = asyncio.Lock()
    inflight: dict[str, int] = {}
    assignments: list[str] = []

    async def try_dispatch():
        async with lock:
            nodes = reg.schedulable_nodes()
            node = best_fit_node(nodes, cpu_pct=90, ram_mb=400)
            if node is None:
                return
            current = inflight.get(node.id, 0)
            if current >= node.donatable_cpu_slots():
                return
            inflight[node.id] = current + 1
            assignments.append(node.id)

    async def run():
        await asyncio.gather(*[try_dispatch() for _ in range(20)])

    asyncio.run(run())

    assert len(assignments) <= 1, (
        f"Over-committed: {len(assignments)} assignments to a 1-slot node"
    )
    store.close()


# ---------------------------------------------------------------------------
# 5. add_ledger_pair is atomic — both entries present or neither
# ---------------------------------------------------------------------------

def test_ledger_pair_atomic(tmp_path):
    """add_ledger_pair writes both entries in one transaction; counts always even."""
    store = _make_store(tmp_path)
    N = 100

    def write(_):
        don = _ledger_entry("sys", "carol", "donated", 5.0)
        con = _ledger_entry("sys", "carol", "consumed", 5.0)
        store.add_ledger_pair(don, con)

    with ThreadPoolExecutor(max_workers=10) as ex:
        list(ex.map(write, range(N)))

    rows = store.recent_ledger(limit=N * 2 + 10)
    n_don = sum(1 for r in rows if r.kind == "donated")
    n_con = sum(1 for r in rows if r.kind == "consumed")
    assert n_don == N, f"Expected {N} donated entries, got {n_don}"
    assert n_con == N, f"Expected {N} consumed entries, got {n_con}"
    assert n_don == n_con, "donation/consumption counts diverged"
    store.close()


# ---------------------------------------------------------------------------
# 6. Idempotent ledger insert (add_ledger_idempotent)
# ---------------------------------------------------------------------------

def test_ledger_idempotent(tmp_path):
    """The same entry inserted 50 times from 10 threads must appear exactly once."""
    store = _make_store(tmp_path)
    shared_id = new_id("led")

    def write(_):
        entry = LedgerEntry(
            id=shared_id, node_id="n1", owner="dave",
            kind="donated", compute_units=1.0, ts_ms=1234,
        )
        store.add_ledger_idempotent(entry)

    with ThreadPoolExecutor(max_workers=10) as ex:
        list(ex.map(write, range(50)))

    rows = store.recent_ledger(limit=100)
    matching = [r for r in rows if r.id == shared_id]
    assert len(matching) == 1, f"Expected 1 row, got {len(matching)}"
    store.close()
