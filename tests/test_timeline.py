from __future__ import annotations

from fastapi.testclient import TestClient

from meshpool.common.config import Settings
from meshpool.common.tracing import TraceContext
from meshpool.controlplane.app import ROUTE_SCOPES, build_app
from meshpool.common.auth import Scope


def _home_app(tmp_path):
    settings = Settings.load(scale="home", data_dir=str(tmp_path),
                             host="127.0.0.1", port=8301)
    return build_app(settings)


def _public_app(tmp_path):
    settings = Settings.load(scale="public", data_dir=str(tmp_path),
                             host="127.0.0.1", port=8302, auth_mode="enforced")
    return build_app(settings)


def test_job_timeline_endpoint_is_declared_jobs_read(tmp_path):
    with TestClient(_home_app(tmp_path)):
        assert ROUTE_SCOPES["GET /api/v1/jobs/{job_id}/timeline"] is Scope.JOBS_READ


def test_job_timeline_requires_auth_for_remote_public_client(tmp_path):
    app = _public_app(tmp_path)
    with TestClient(app, client=("203.0.113.7", 44444)) as client:
        r = client.get("/api/v1/jobs/job-missing/timeline")
        assert r.status_code == 401


def test_job_timeline_uses_traceparent_and_event_stream(tmp_path):
    incoming = TraceContext(trace_id="a" * 32, span_id="b" * 16)
    with TestClient(_home_app(tmp_path)) as client:
        submit = client.post(
            "/api/v1/jobs",
            headers={"traceparent": incoming.to_traceparent()},
            json={
                "name": "observability-test",
                "func": "square",
                "payloads": [{"x": 3}],
                "cpu_pct": 10,
                "ram_mb": 64,
                "owner": "default",
                "secret_names": [],
            },
        )
        assert submit.status_code == 200, submit.text
        job_id = submit.json()["job_id"]
        assert submit.json()["trace_id"] == incoming.trace_id

        r = client.get(f"/api/v1/jobs/{job_id}/timeline")
        assert r.status_code == 200, r.text
        timeline = r.json()

    assert timeline["job_id"] == job_id
    assert timeline["trace_id"] == incoming.trace_id
    assert len(timeline["events"]) >= 2, timeline
    names = {event["name"] for event in timeline["events"]}
    assert "job" in names
    assert "event.job_submitted" in names
    assert timeline["lanes"][0]["node_id"] == "control-plane"
    assert timeline["events"] == sorted(timeline["events"], key=lambda e: (e["start_ms"], e.get("span_id") or ""))

def test_timeline_instrumentation_accepts_every_run_unit_argument(tmp_path):
    """The timeline wraps a *private* batch method, so it must stay signature-transparent.

    A wrapper that drops an argument does not fail loudly in unit tests -- it fails
    at dispatch time, on every single unit, taking the entire batch engine down
    while every non-instrumented test still passes. That happened once already when
    speculative backup execution added a keyword argument.
    """
    import inspect

    from meshpool.batch.engine import BatchManager
    from meshpool.common.db import Store
    from meshpool.controlplane.timeline import JobTimeline

    batch = BatchManager.__new__(BatchManager)
    JobTimeline(Store(str(tmp_path / "timeline.db"))).instrument_batch(batch)

    # follow_wrapped=False is what makes this test real: functools.wraps sets
    # __wrapped__, and inspect.signature follows it by default, happily reporting
    # the original signature for a wrapper that cannot actually accept it.
    original = inspect.signature(BatchManager._run_unit, follow_wrapped=False)
    wrapper = inspect.signature(batch._run_unit, follow_wrapped=False)

    params = [p for name, p in original.parameters.items() if name != "self"]
    positional = [object() for p in params if p.default is p.empty]
    keyword = {p.name: p.default for p in params if p.default is not p.empty}
    assert keyword, "expected _run_unit to still have optional arguments to guard"

    wrapper.bind(*positional, **keyword)
