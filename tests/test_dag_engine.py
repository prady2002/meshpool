from __future__ import annotations

import asyncio
import sqlite3
import threading
import time

import meshpool.batch.engine as batch_engine
from meshpool.batch.dag import DagManager
from meshpool.batch.engine import BatchManager, NO_RUNTIME_GRACE_MS
from meshpool.common.models import (
    BatchJob,
    Capability,
    DonationPolicy,
    DagStageSpec,
    Node,
    UnitStatus,
    WorkUnit,
    new_id,
    now_ms,
)


class _DagStore:
    def __init__(self):
        self._conn = sqlite3.connect(":memory:")
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.RLock()


class _FakeBatch:
    def __init__(self):
        self.jobs = {}
        self.submits = []

    def submit(self, name, func, payloads, cpu_pct, ram_mb, owner="default", secret_names=None):
        job = BatchJob(
            id=new_id("job"),
            name=name,
            func=func,
            units=[WorkUnit(id=f"u{i}", payload=p) for i, p in enumerate(payloads)],
            cpu_pct_per_unit=cpu_pct,
            ram_mb_per_unit=ram_mb,
            owner=owner,
            secret_names=secret_names or [],
            status="running",
            created_ms=now_ms(),
        )
        self.jobs[job.id] = job
        self.submits.append((name, func, list(payloads)))
        return job

    def get(self, job_id):
        return self.jobs.get(job_id)

    def complete(self, job_id, results):
        job = self.jobs[job_id]
        for unit, result in zip(job.units, results):
            unit.status = UnitStatus.DONE
            unit.result = result
            unit.attempts = max(1, unit.attempts)
        job.status = "completed"


def test_dag_runs_ready_stages_in_parallel_then_reduces():
    store = _DagStore()
    batch = _FakeBatch()
    mgr = DagManager(store, batch)
    dag = mgr.submit("mr", [
        DagStageSpec(id="left", func="square", payloads=[{"x": 2}, {"x": 3}]),
        DagStageSpec(id="right", func="square", payloads=[{"x": 4}]),
        DagStageSpec(id="sum", func="square", depends_on=["left", "right"],
                     reduce=True, payloads=[{"tag": "reduce"}]),
    ])

    asyncio.run(mgr.dispatch_once())

    running = {s.id for s in dag.stages if s.status == "running"}
    assert running == {"left", "right"}  # non-vacuity: both independent roots launched
    assert len(batch.submits) == 2

    left = next(s for s in dag.stages if s.id == "left")
    batch.complete(left.job_id, [4, 9])
    asyncio.run(mgr.dispatch_once())
    assert next(s for s in dag.stages if s.id == "sum").status == "pending"

    right = next(s for s in dag.stages if s.id == "right")
    batch.complete(right.job_id, [16])
    asyncio.run(mgr.dispatch_once())

    reducer_payload = batch.submits[-1][2][0]
    assert reducer_payload["items"] == [4, 9, 16]
    assert reducer_payload["by_stage"] == {"left": [4, 9], "right": [16]}


def test_dag_checkpoint_resume_does_not_recompute_succeeded_stage():
    store = _DagStore()
    batch = _FakeBatch()
    mgr = DagManager(store, batch)
    dag = mgr.submit("resume", [
        DagStageSpec(id="map", func="square", payloads=[{"x": 2}, {"x": 3}]),
        DagStageSpec(id="reduce", func="square", depends_on=["map"],
                     reduce=True, payloads=[{}]),
    ])
    asyncio.run(mgr.dispatch_once())
    map_stage = next(s for s in dag.stages if s.id == "map")
    batch.complete(map_stage.job_id, [4, 9])
    asyncio.run(mgr.dispatch_once())
    before = list(batch.submits)
    assert map_stage.status == "succeeded"
    assert map_stage.execution_count == 2

    resumed = DagManager(store, batch)
    asyncio.run(resumed.dispatch_once())

    after = list(batch.submits)
    map_submits = [s for s in after if s[0].endswith(":map")]
    assert after[:len(before)] == before
    assert len(map_submits) == 1  # would be 2 without a durable completed-stage checkpoint
    resumed_map = next(s for s in resumed.get(dag.id).stages if s.id == "map")
    assert resumed_map.execution_count == 2


class _Events:
    def publish(self, kind, data):
        pass


class _BatchStore:
    def __init__(self):
        self.saved = []

    def list_jobs(self):
        return []

    def put_job(self, job):
        self.saved.append(job)

    def credits_by_owner(self):
        return {}


class _Registry:
    def schedulable_nodes(self):
        return [
            Node(id="wasm-only", name="n", url="http://n",
                 capability=Capability(runtimes=["wasm"], services=["batch"]))
        ]


class _Trust:
    tier = 0

    def record_task_result(self, node_id, ok):
        pass


class _Ledger:
    def __init__(self):
        self.donations = []
        self.consumptions = []

    def record_donation(self, node_id, owner, cost, detail=""):
        self.donations.append((node_id, owner, cost, detail))

    def record_consumption(self, node_id, owner, cost, detail=""):
        self.consumptions.append((node_id, owner, cost, detail))


class _ThreeNodeRegistry:
    def __init__(self):
        self.nodes = [
            Node(id="slow", name="slow", url="http://slow",
                 capability=Capability(cpu_cores=1, runtimes=["builtin"], services=["batch"]),
                 policy=DonationPolicy(cpu_pct=100, ram_mb=1024)),
            Node(id="fast-a", name="fast-a", url="http://fast-a",
                 capability=Capability(cpu_cores=1, runtimes=["builtin"], services=["batch"]),
                 policy=DonationPolicy(cpu_pct=100, ram_mb=1024)),
            Node(id="fast-b", name="fast-b", url="http://fast-b",
                 capability=Capability(cpu_cores=1, runtimes=["builtin"], services=["batch"]),
                 policy=DonationPolicy(cpu_pct=100, ram_mb=1024)),
        ]

    def schedulable_nodes(self):
        return self.nodes


class _DelayTransport:
    def __init__(self):
        self.calls = []

    async def call(self, node_id, path, payload=None, *, method="POST", timeout=30.0):
        self.calls.append((node_id, payload["unit_id"]))
        await asyncio.sleep(0.35 if node_id == "slow" else 0.01)
        return {"result": {"node": node_id}, "compute_ms": 10}


def test_dag_unservable_stage_fails_loudly_not_pending_forever():
    batch_store = _BatchStore()
    batch = BatchManager(_Registry(), batch_store, ledger=None, trust=_Trust(),
                         events=_Events(), transport=None)
    dag_mgr = DagManager(_DagStore(), batch)
    dag = dag_mgr.submit("bad", [
        DagStageSpec(id="builtin", func="square", payloads=[{"x": 2}]),
    ])

    asyncio.run(dag_mgr.dispatch_once())
    stage = dag.stages[0]
    job = batch.get(stage.job_id)
    job.created_ms = now_ms() - NO_RUNTIME_GRACE_MS - 1
    asyncio.run(batch.dispatch_once())
    asyncio.run(dag_mgr.dispatch_once())

    assert stage.status == "failed"
    assert "builtin" in (stage.error or "")
    assert dag.status == "completed_with_failures"


def test_speculative_backup_uses_different_node_without_double_credit(monkeypatch):
    monkeypatch.setattr(batch_engine, "SPECULATION_MIN_COMPLETED", 1)
    monkeypatch.setattr(batch_engine, "SPECULATION_FLOOR_MS", 40)
    monkeypatch.setattr(batch_engine, "SPECULATION_MULTIPLIER", 1.0)
    monkeypatch.setenv("MESHPOOL_BATCH_SPECULATION", "1")
    store = _BatchStore()
    ledger = _Ledger()
    transport = _DelayTransport()
    mgr = BatchManager(_ThreeNodeRegistry(), store, ledger=ledger, trust=_Trust(),
                       events=_Events(), transport=transport)
    job = mgr.submit("spec", "square", [{"x": 1}, {"x": 2}], 10, 64)

    async def run_until_done():
        deadline = time.time() + 3
        while time.time() < deadline and not job.status.startswith("completed"):
            await mgr.dispatch_once()
            await asyncio.sleep(0.03)

    asyncio.run(run_until_done())

    assert job.status == "completed"
    assert mgr.speculative_attempts >= 1
    assert mgr.speculative_wins >= 1
    slow_unit_id = transport.calls[0][1]
    nodes_for_slow = [n for n, u in transport.calls if u == slow_unit_id]
    assert len(set(nodes_for_slow)) >= 2
    assert "slow" in nodes_for_slow
    assert all(u.attempts == 1 for u in job.units)
    assert len(ledger.donations) == 2
    assert len(ledger.consumptions) == 2
