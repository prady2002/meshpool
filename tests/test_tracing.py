from __future__ import annotations

import re

from meshpool.common.tracing import (
    TRACEPARENT_HEADER,
    TraceContext,
    parse_traceparent,
    start_span,
)


def test_traceparent_round_trip_and_rejects_zero_ids():
    ctx = TraceContext.new()
    header = ctx.to_traceparent()
    assert re.fullmatch(r"00-[0-9a-f]{32}-[0-9a-f]{16}-01", header)

    parsed = parse_traceparent(header)
    assert parsed is not None
    assert parsed.trace_id == ctx.trace_id
    assert parsed.span_id == ctx.span_id
    assert parsed.sampled is True
    assert TraceContext.from_headers({TRACEPARENT_HEADER.upper(): header}) == parsed

    assert parse_traceparent("00-" + ("0" * 32) + "-" + ("1" * 16) + "-01") is None
    assert parse_traceparent("not-a-traceparent") is None


def test_start_span_exports_parent_linkage_and_duration():
    exported = []
    parent = TraceContext(trace_id="1" * 32, span_id="2" * 16)

    with start_span("controlplane.test", context=parent,
                    attributes={"job_id": "job-1"}, exporter=exported.append) as span:
        assert span.trace_id == parent.trace_id
        assert span.parent_span_id == parent.span_id

    assert len(exported) == 1
    assert exported[0].name == "controlplane.test"
    assert exported[0].outcome == "ok"
    assert exported[0].duration_ms is not None
    assert exported[0].attributes["job_id"] == "job-1"
