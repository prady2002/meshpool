"""Batch engine: a unit that exhausts its retries must finalize the job.

Regression for a stuck-job bug where `dispatch_once` marked a max-attempts unit
FAILED but never finalized/persisted, so a job whose last outstanding unit exhausted
its retries would hang forever in "running".
"""
from __future__ import annotations

import asyncio

from meshpool.batch.engine import MAX_ATTEMPTS, BatchManager
from meshpool.common.models import BatchJob, UnitStatus, WorkUnit, now_ms


class _Events:
    def __init__(self):
        self.published = []

    def publish(self, kind, data):
        self.published.append((kind, data))


class _Store:
    def __init__(self):
        self.saved = 0

    def list_jobs(self):
        return []

    def put_job(self, job):
        self.saved += 1


def _manager() -> BatchManager:
    # registry/ledger/trust/transport are only touched on the dispatch path, which the
    # max-attempts branch never reaches, so stubs are enough for this regression.
    return BatchManager(registry=None, store=_Store(), ledger=None, trust=None,
                        events=_Events(), transport=None)


def test_unit_exhausting_retries_finalizes_job():
    mgr = _manager()
    unit = WorkUnit(id="u1", status=UnitStatus.PENDING, attempts=MAX_ATTEMPTS)
    job = BatchJob(id="job1", name="j", func="square", units=[unit],
                   status="running", created_ms=now_ms())
    mgr.jobs[job.id] = job

    asyncio.run(mgr.dispatch_once())

    assert unit.status is UnitStatus.FAILED
    assert unit.error                                    # a reason is recorded
    assert job.status == "completed_with_failures"       # finalized, not stuck "running"
    assert mgr.store.saved >= 1                           # persisted


def test_partial_failure_finalizes_when_last_unit_exhausts():
    mgr = _manager()
    done = WorkUnit(id="ok", status=UnitStatus.DONE, attempts=1)
    dead = WorkUnit(id="bad", status=UnitStatus.PENDING, attempts=MAX_ATTEMPTS)
    job = BatchJob(id="job2", name="j", func="square", units=[done, dead],
                   status="running", created_ms=now_ms())
    mgr.jobs[job.id] = job

    asyncio.run(mgr.dispatch_once())

    assert dead.status is UnitStatus.FAILED
    assert job.status == "completed_with_failures"
