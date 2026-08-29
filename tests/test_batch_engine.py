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


class _Caps:
    """Minimal SpendCapManager stand-in: one switch, flipped by the test."""

    def __init__(self, exceeded: bool = True):
        self._exceeded = exceeded

    def exceeded(self, owner):  # noqa: ARG002 — cap is global in this stub
        return self._exceeded


def test_spend_capped_job_says_paused_instead_of_looking_hung():
    """A job stopped by the hard spend cap must SAY it stopped.

    Regression: the engine logged a warning, published an event and skipped
    dispatch, but left ``status == "running"`` with units pending. Over the API
    that is byte-for-byte identical to a hung scheduler, so every client -- the
    console, the CLI, and the acceptance gates -- waited forever for a job that
    had deliberately and correctly stopped.
    """
    mgr = _manager()
    mgr.spend_caps = _Caps(exceeded=True)
    pending = WorkUnit(id="u1", status=UnitStatus.PENDING, attempts=0)
    job = BatchJob(id="job3", name="j", func="square", units=[pending],
                   status="running", created_ms=now_ms())
    mgr.jobs[job.id] = job

    asyncio.run(mgr.dispatch_once())

    assert job.status == "paused", "a capped job must be distinguishable from a hung one"
    assert pending.status is UnitStatus.PENDING, "capped work must not be dispatched"
    kinds = [k for k, _ in mgr.events.published]
    assert "job_spend_capped" in kinds
    payload = dict(next(d for k, d in mgr.events.published if k == "job_spend_capped"))
    assert payload["status"] == "paused"
    assert "raise the cap" in payload["reason"]


def test_raising_the_cap_resumes_a_paused_job():
    """``paused`` is non-terminal: the job must be reconsidered once uncapped.

    The guard at the top of the dispatch loop only looks at jobs whose status is
    dispatchable, so introducing a new status without adding it there would have
    made the pause permanent -- a worse bug than the one being fixed.
    """
    mgr = _manager()
    caps = _Caps(exceeded=True)
    mgr.spend_caps = caps
    unit = WorkUnit(id="u1", status=UnitStatus.PENDING, attempts=MAX_ATTEMPTS)
    job = BatchJob(id="job4", name="j", func="square", units=[unit],
                   status="running", created_ms=now_ms())
    mgr.jobs[job.id] = job

    asyncio.run(mgr.dispatch_once())
    assert job.status == "paused"

    caps._exceeded = False
    asyncio.run(mgr.dispatch_once())

    assert job.status != "paused", "raising the cap must un-pause the job"
    assert "job_spend_resumed" in [k for k, _ in mgr.events.published]


def test_job_capped_after_its_last_unit_finished_still_finalizes():
    """The cap must not strand an already-complete job in ``paused`` forever.

    Dispatch returns early for a capped owner, so the finalize that normally
    runs on the dispatch path never fires. A job whose units had all finished
    just before the cap tripped would then never reach a terminal status.
    """
    mgr = _manager()
    mgr.spend_caps = _Caps(exceeded=True)
    done = WorkUnit(id="u1", status=UnitStatus.DONE, attempts=1)
    job = BatchJob(id="job5", name="j", func="square", units=[done],
                   status="running", created_ms=now_ms())
    mgr.jobs[job.id] = job

    asyncio.run(mgr.dispatch_once())

    assert job.status == "completed", "a finished job must finalize even while capped"
