"""Per-job timeline construction and lightweight span persistence."""
from __future__ import annotations

import functools
import json
from typing import Any, Optional

from ..common.models import BatchJob, UnitStatus, now_ms
from ..common.tracing import SpanRecord, TraceContext, current_trace_context, new_span_id


class JobTimeline:
    """Persist spans and build a console-ready per-node waterfall for a job."""

    def __init__(self, store) -> None:
        self.store = store
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        with self.store._lock:
            self.store._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS trace_spans (
                    span_id TEXT PRIMARY KEY,
                    trace_id TEXT NOT NULL,
                    parent_span_id TEXT,
                    job_id TEXT NOT NULL,
                    unit_id TEXT,
                    node_id TEXT,
                    attempt INTEGER,
                    name TEXT NOT NULL,
                    start_ms INTEGER NOT NULL,
                    end_ms INTEGER,
                    outcome TEXT,
                    error TEXT,
                    attributes TEXT NOT NULL DEFAULT '{}'
                )
                """
            )
            self.store._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_trace_spans_job "
                "ON trace_spans(job_id,start_ms)"
            )
            self.store._conn.commit()

    def record_span(self, span: SpanRecord, *, job_id: str,
                    unit_id: Optional[str] = None, node_id: Optional[str] = None,
                    attempt: Optional[int] = None) -> None:
        attrs = dict(span.attributes)
        with self.store._lock:
            self.store._conn.execute(
                "INSERT INTO trace_spans(span_id,trace_id,parent_span_id,job_id,unit_id,"
                "node_id,attempt,name,start_ms,end_ms,outcome,error,attributes) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?) "
                "ON CONFLICT(span_id) DO UPDATE SET end_ms=excluded.end_ms,"
                "outcome=excluded.outcome,error=excluded.error,attributes=excluded.attributes",
                (span.span_id, span.trace_id, span.parent_span_id, job_id, unit_id,
                 node_id, attempt, span.name, span.start_ms, span.end_ms,
                 span.outcome, span.error, json.dumps(attrs, sort_keys=True)),
            )
            self.store._conn.commit()

    def start_job(self, job: BatchJob, parent: Optional[TraceContext] = None) -> str:
        existing = self.trace_id_for_job(job.id)
        if existing:
            return existing
        ctx = parent or current_trace_context() or TraceContext.new()
        span = SpanRecord(trace_id=ctx.trace_id, span_id=new_span_id(),
                          parent_span_id=ctx.span_id, name="job",
                          start_ms=job.created_ms,
                          attributes={"job_name": job.name, "func": job.func,
                                      "owner": job.owner, "unit_count": len(job.units)})
        self.record_span(span, job_id=job.id)
        return span.trace_id

    def finish_job_if_terminal(self, job: BatchJob) -> None:
        if job.status not in ("completed", "completed_with_failures", "failed"):
            return
        with self.store._lock:
            self.store._conn.execute(
                "UPDATE trace_spans SET end_ms=COALESCE(end_ms, ?), outcome=? "
                "WHERE job_id=? AND name='job'",
                (now_ms(), job.status, job.id),
            )
            self.store._conn.commit()

    def trace_id_for_job(self, job_id: str) -> Optional[str]:
        with self.store._lock:
            row = self.store._conn.execute(
                "SELECT trace_id FROM trace_spans WHERE job_id=? ORDER BY start_ms ASC LIMIT 1",
                (job_id,),
            ).fetchone()
        return row["trace_id"] if row else None

    def spans_for_job(self, job_id: str) -> list[dict[str, Any]]:
        with self.store._lock:
            rows = self.store._conn.execute(
                "SELECT trace_id,span_id,parent_span_id,job_id,unit_id,node_id,attempt,"
                "name,start_ms,end_ms,outcome,error,attributes "
                "FROM trace_spans WHERE job_id=? ORDER BY start_ms ASC, span_id ASC",
                (job_id,),
            ).fetchall()
        out: list[dict[str, Any]] = []
        for row in rows:
            start_ms = int(row["start_ms"])
            end_ms = row["end_ms"]
            attrs = json.loads(row["attributes"] or "{}")
            out.append({
                "type": "span",
                "trace_id": row["trace_id"],
                "span_id": row["span_id"],
                "parent_span_id": row["parent_span_id"],
                "job_id": row["job_id"],
                "unit_id": row["unit_id"],
                "node_id": row["node_id"],
                "attempt": row["attempt"],
                "name": row["name"],
                "start_ms": start_ms,
                "end_ms": int(end_ms) if end_ms is not None else None,
                "duration_ms": (max(0, int(end_ms) - start_ms)
                                if end_ms is not None else None),
                "outcome": row["outcome"],
                "error": row["error"],
                "attributes": attrs,
            })
        return out

    def build(self, job: BatchJob, nodes_by_id: dict[str, Any],
              event_history: Optional[list[dict[str, Any]]] = None) -> dict[str, Any]:
        spans = self.spans_for_job(job.id)
        trace_id = spans[0]["trace_id"] if spans else self.start_job(job)
        if not spans:
            spans = self.spans_for_job(job.id)
        # Store-derived terminal/current state events keep the timeline useful even
        # for jobs created before tracing existed or paths we could not instrument.
        for unit in job.units:
            if not any(s.get("unit_id") == unit.id for s in spans):
                spans.append({
                    "type": "event",
                    "trace_id": trace_id,
                    "span_id": None,
                    "parent_span_id": None,
                    "job_id": job.id,
                    "unit_id": unit.id,
                    "node_id": unit.assigned_node,
                    "attempt": unit.attempts,
                    "name": "unit.current_state",
                    "start_ms": job.created_ms,
                    "end_ms": None,
                    "duration_ms": None,
                    "outcome": unit.status.value,
                    "error": unit.error,
                    "attributes": {"status": unit.status.value},
                })
        for event in event_history or []:
            data = event.get("data") or {}
            spans.append({
                "type": "event",
                "trace_id": trace_id,
                "span_id": None,
                "parent_span_id": None,
                "job_id": job.id,
                "unit_id": data.get("unit_id"),
                "node_id": data.get("node") or data.get("node_id"),
                "attempt": data.get("attempt"),
                "name": f"event.{event.get('kind', 'unknown')}",
                "start_ms": int(event.get("ts") or job.created_ms),
                "end_ms": int(event.get("ts") or job.created_ms),
                "duration_ms": 0,
                "outcome": data.get("status"),
                "error": data.get("error"),
                "attributes": data,
            })
        spans.sort(key=lambda item: (item["start_ms"], item.get("span_id") or ""))
        first = min((s["start_ms"] for s in spans), default=job.created_ms)
        last = max((s["end_ms"] or s["start_ms"] for s in spans), default=job.created_ms)
        lane_map: dict[str, dict[str, Any]] = {}
        for span in spans:
            node_id = span.get("node_id") or "control-plane"
            node = nodes_by_id.get(node_id)
            lane = lane_map.setdefault(node_id, {
                "node_id": node_id,
                "node_name": getattr(node, "name", None) or (
                    "Control plane" if node_id == "control-plane" else node_id),
                "spans": [],
            })
            lane["spans"].append(span)
        lanes = sorted(lane_map.values(), key=lambda lane: (
            0 if lane["node_id"] == "control-plane" else 1,
            lane["node_name"],
        ))
        return {
            "job_id": job.id,
            "trace_id": trace_id,
            "status": job.status,
            "created_ms": job.created_ms,
            "start_ms": first,
            "end_ms": last if job.status.startswith("completed") else None,
            "duration_ms": max(0, last - first),
            "units": [{
                "unit_id": u.id,
                "status": u.status.value if isinstance(u.status, UnitStatus) else str(u.status),
                "node_id": u.assigned_node,
                "attempts": u.attempts,
                "outcome": "success" if u.status is UnitStatus.DONE else (
                    "failed" if u.status is UnitStatus.FAILED else u.status.value),
                "error": u.error,
            } for u in job.units],
            "events": spans,
            "lanes": lanes,
        }

    def instrument_batch(self, batch) -> None:
        if getattr(batch, "_meshpool_timeline_instrumented", False):
            return

        original_submit = batch.submit
        original_run_unit = batch._run_unit

        @functools.wraps(original_submit)
        def submit_wrapper(*args, **kwargs):
            job = original_submit(*args, **kwargs)
            self.start_job(job, current_trace_context())
            return job

        @functools.wraps(original_run_unit)
        async def run_unit_wrapper(job, unit, node, *args, **kwargs):
            # Pass-through *args/**kwargs is load-bearing, not defensive style.
            # This wraps a private method, so any new argument on the real
            # _run_unit (speculative backups added one) would otherwise raise
            # TypeError on every dispatch and take the whole batch engine down
            # while every unit test that does not instrument still passes.
            speculative = bool(kwargs.get("speculative", args[0] if args else False))
            self.start_job(job, current_trace_context())
            parent = current_trace_context()
            if parent is None:
                parent = TraceContext(trace_id=self.trace_id_for_job(job.id) or TraceContext.new().trace_id,
                                      span_id=new_span_id())
            span = SpanRecord(trace_id=parent.trace_id, span_id=new_span_id(),
                              parent_span_id=parent.span_id, name="unit.attempt",
                              start_ms=now_ms(),
                              attributes={"func": job.func, "payload": unit.payload,
                                          "speculative": speculative})
            attempt = unit.attempts
            try:
                await original_run_unit(job, unit, node, *args, **kwargs)
            finally:
                if unit.status is UnitStatus.DONE:
                    outcome = "success"
                elif unit.status is UnitStatus.PENDING and unit.error:
                    outcome = "retry"
                elif unit.status is UnitStatus.FAILED:
                    outcome = "failed"
                else:
                    outcome = unit.status.value
                span.finish(outcome=outcome, error=unit.error)
                span.attributes.update({"status": unit.status.value, "attempt": attempt})
                self.record_span(span, job_id=job.id, unit_id=unit.id,
                                 node_id=node.id, attempt=attempt)
                self.finish_job_if_terminal(job)

        batch.submit = submit_wrapper
        batch._run_unit = run_unit_wrapper
        batch._meshpool_timeline_instrumented = True
