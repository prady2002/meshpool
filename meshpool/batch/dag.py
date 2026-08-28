"""Durable DAG/workflow scheduler layered on the batch unit runtime."""
from __future__ import annotations

import asyncio
import json
from typing import Optional

from ..common.logging import get_logger
from ..common.models import (
    DagStageSpec,
    DagStageState,
    DagWorkflow,
    UnitStatus,
    new_id,
    now_ms,
)

log = get_logger("dag")


class DagError(ValueError):
    pass


class DagManager:
    """Schedules DAG stages by submitting ordinary batch jobs.

    The durable checkpoint is the stage state in the ``dags`` table. Once a stage is
    marked ``succeeded`` its results and execution count are persisted and the stage
    is never submitted again after a control-plane restart.
    """

    def __init__(self, store, batch) -> None:
        self.store = store
        self.batch = batch
        self._lock = asyncio.Lock()
        self._ensure_table()
        self.dags: dict[str, DagWorkflow] = {d.id: d for d in self._load_all()}

    def _ensure_table(self) -> None:
        with self.store._lock:
            self.store._conn.execute(
                "CREATE TABLE IF NOT EXISTS dags ("
                "id TEXT PRIMARY KEY, owner TEXT NOT NULL, status TEXT NOT NULL, "
                "created_ms INTEGER NOT NULL, updated_ms INTEGER NOT NULL, data TEXT NOT NULL)"
            )
            self.store._conn.commit()

    def _put(self, dag: DagWorkflow) -> None:
        dag.updated_ms = now_ms()
        with self.store._lock:
            self.store._conn.execute(
                "INSERT INTO dags(id,owner,status,created_ms,updated_ms,data) "
                "VALUES(?,?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET "
                "owner=excluded.owner,status=excluded.status,"
                "updated_ms=excluded.updated_ms,data=excluded.data",
                (dag.id, dag.owner, dag.status, dag.created_ms, dag.updated_ms,
                 dag.model_dump_json()),
            )
            self.store._conn.commit()

    def _load_all(self) -> list[DagWorkflow]:
        with self.store._lock:
            rows = self.store._conn.execute(
                "SELECT data FROM dags ORDER BY created_ms DESC"
            ).fetchall()
        return [DagWorkflow.model_validate_json(r["data"]) for r in rows]

    def submit(self, name: str, stages: list[DagStageSpec | dict],
               owner: str = "default") -> DagWorkflow:
        states = [s if isinstance(s, DagStageSpec) else DagStageSpec.model_validate(s)
                  for s in stages]
        self._validate(states)
        dag = DagWorkflow(
            id=new_id("dag"),
            name=name,
            stages=[DagStageState(**s.model_dump()) for s in states],
            owner=owner,
            status="running",
            created_ms=now_ms(),
        )
        self.dags[dag.id] = dag
        self._put(dag)
        return dag

    def get(self, dag_id: str) -> Optional[DagWorkflow]:
        return self.dags.get(dag_id)

    def list(self) -> list[DagWorkflow]:
        return sorted(self.dags.values(), key=lambda d: d.created_ms, reverse=True)

    def _validate(self, stages: list[DagStageSpec]) -> None:
        ids = [s.id for s in stages]
        if len(ids) != len(set(ids)):
            raise DagError("stage ids must be unique")
        known = set(ids)
        for s in stages:
            missing = [d for d in s.depends_on if d not in known]
            if missing:
                raise DagError(f"stage {s.id} depends on unknown stage(s): {missing}")
            if not s.payloads:
                raise DagError(f"stage {s.id} must have at least one payload")
        graph = {s.id: set(s.depends_on) for s in stages}
        visiting: set[str] = set()
        seen: set[str] = set()

        def visit(node: str) -> None:
            if node in seen:
                return
            if node in visiting:
                raise DagError("DAG contains a dependency cycle")
            visiting.add(node)
            for dep in graph[node]:
                visit(dep)
            visiting.remove(node)
            seen.add(node)

        for stage_id in ids:
            visit(stage_id)

    def _stage_map(self, dag: DagWorkflow) -> dict[str, DagStageState]:
        return {s.id: s for s in dag.stages}

    def _payloads_for(self, dag: DagWorkflow, stage: DagStageState) -> list[dict]:
        if not stage.reduce:
            return stage.payloads
        stages = self._stage_map(dag)
        dep_results = {dep: stages[dep].results for dep in stage.depends_on}
        flattened = [item for dep in stage.depends_on for item in stages[dep].results]
        base = dict(stage.payloads[0]) if stage.payloads else {}
        base[stage.reduce_input_key] = flattened
        base["by_stage"] = dep_results
        return [base]

    def _mark_terminal_if_ready(self, dag: DagWorkflow) -> bool:
        if any(s.status == "failed" for s in dag.stages):
            dag.status = "completed_with_failures"
            return True
        if all(s.status == "succeeded" for s in dag.stages):
            dag.status = "completed"
            return True
        dag.status = "running"
        return False

    async def dispatch_once(self) -> None:
        async with self._lock:
            for dag in self.dags.values():
                if not dag.status.startswith("running") and dag.status != "pending":
                    continue
                dirty = False
                stages = self._stage_map(dag)

                for stage in dag.stages:
                    if stage.status != "running" or not stage.job_id:
                        continue
                    job = self.batch.get(stage.job_id)
                    if job is None:
                        stage.status = "failed"
                        stage.error = f"batch job {stage.job_id} disappeared"
                        dirty = True
                        continue
                    if not str(job.status).startswith("completed"):
                        continue
                    stage.execution_count = max(
                        stage.execution_count,
                        sum(max(1, u.attempts) for u in job.units),
                    )
                    failed = [u for u in job.units if u.status is UnitStatus.FAILED]
                    if failed:
                        stage.status = "failed"
                        stage.error = "; ".join((u.error or "unit failed") for u in failed)[:500]
                    else:
                        stage.status = "succeeded"
                        stage.results = [u.result for u in job.units]
                        stage.error = None
                    stage.completed_ms = now_ms()
                    dirty = True

                if self._mark_terminal_if_ready(dag):
                    if dirty:
                        self._put(dag)
                    continue

                for stage in dag.stages:
                    if stage.status != "pending":
                        continue
                    if not all(stages[d].status == "succeeded" for d in stage.depends_on):
                        continue
                    payloads = self._payloads_for(dag, stage)
                    job = self.batch.submit(
                        f"{dag.name}:{stage.id}", stage.func, payloads,
                        stage.cpu_pct, stage.ram_mb, dag.owner,
                        secret_names=stage.secret_names,
                    )
                    stage.job_id = job.id
                    stage.status = "running"
                    stage.started_ms = now_ms()
                    stage.execution_count += len(payloads)
                    dirty = True
                    log.info("dag stage submitted", dag=dag.id, stage=stage.id,
                             job=job.id, units=len(payloads))

                self._mark_terminal_if_ready(dag)
                if dirty:
                    self._put(dag)

    async def run_loop(self, interval: float = 0.5) -> None:
        while True:
            try:
                await self.dispatch_once()
            except Exception as exc:  # noqa: BLE001
                log.error("dag loop error", err=str(exc))
            await asyncio.sleep(interval)


def parse_stages_json(text: str) -> list[DagStageSpec]:
    raw = json.loads(text)
    if isinstance(raw, dict):
        raw = raw.get("stages", [])
    if not isinstance(raw, list):
        raise DagError("DAG JSON must be a list of stages or an object with stages")
    return [DagStageSpec.model_validate(x) for x in raw]
