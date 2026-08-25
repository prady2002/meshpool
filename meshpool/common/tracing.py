"""Dependency-light tracing primitives for MeshPool.

This module intentionally does not depend on an OpenTelemetry SDK.  It emits
small serialisable span records and understands the W3C ``traceparent`` header
shape so an optional external exporter can bridge the records later.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
import secrets
import time
from typing import Any, Callable, Iterator, Mapping, Optional

TRACEPARENT_HEADER = "traceparent"
_VERSION = "00"
_FLAGS = "01"


def now_ms() -> int:
    return int(time.time() * 1000)


def new_trace_id() -> str:
    return secrets.token_hex(16)


def new_span_id() -> str:
    return secrets.token_hex(8)


def _is_hex(value: str, length: int) -> bool:
    return len(value) == length and all(c in "0123456789abcdef" for c in value)


@dataclass(frozen=True)
class TraceContext:
    trace_id: str
    span_id: str
    sampled: bool = True

    @classmethod
    def new(cls) -> "TraceContext":
        return cls(trace_id=new_trace_id(), span_id=new_span_id())

    @classmethod
    def from_headers(cls, headers: Mapping[str, str]) -> Optional["TraceContext"]:
        value = ""
        for key, item in headers.items():
            if key.lower() == TRACEPARENT_HEADER:
                value = item.strip()
                break
        return parse_traceparent(value) if value else None

    def child(self) -> "TraceContext":
        return TraceContext(trace_id=self.trace_id, span_id=new_span_id(),
                            sampled=self.sampled)

    def to_traceparent(self) -> str:
        return format_traceparent(self)


def parse_traceparent(value: str) -> Optional[TraceContext]:
    """Parse a W3C traceparent header.

    Invalid all-zero ids are rejected per the spec. Unknown versions are accepted
    only if the basic four-field format and id lengths are valid.
    """
    parts = value.strip().lower().split("-")
    if len(parts) != 4:
        return None
    version, trace_id, span_id, flags = parts
    if len(version) != 2 or len(flags) != 2:
        return None
    if not _is_hex(trace_id, 32) or not _is_hex(span_id, 16) or not _is_hex(flags, 2):
        return None
    if trace_id == "0" * 32 or span_id == "0" * 16:
        return None
    return TraceContext(trace_id=trace_id, span_id=span_id,
                        sampled=(int(flags, 16) & 1) == 1)


def format_traceparent(ctx: TraceContext) -> str:
    flags = _FLAGS if ctx.sampled else "00"
    return f"{_VERSION}-{ctx.trace_id}-{ctx.span_id}-{flags}"


@dataclass
class SpanRecord:
    trace_id: str
    span_id: str
    parent_span_id: Optional[str]
    name: str
    start_ms: int
    end_ms: Optional[int] = None
    attributes: dict[str, Any] = field(default_factory=dict)
    outcome: Optional[str] = None
    error: Optional[str] = None

    @property
    def duration_ms(self) -> Optional[int]:
        return None if self.end_ms is None else max(0, self.end_ms - self.start_ms)

    def finish(self, outcome: str = "ok", error: Optional[str] = None,
               end_ms: Optional[int] = None) -> "SpanRecord":
        self.end_ms = now_ms() if end_ms is None else end_ms
        self.outcome = outcome
        self.error = error
        return self

    def as_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "name": self.name,
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "duration_ms": self.duration_ms,
            "attributes": dict(self.attributes),
            "outcome": self.outcome,
            "error": self.error,
        }


_current: ContextVar[Optional[TraceContext]] = ContextVar("meshpool_trace", default=None)


def current_trace_context() -> Optional[TraceContext]:
    return _current.get()


@contextmanager
def start_span(name: str, *, context: Optional[TraceContext] = None,
               attributes: Optional[dict[str, Any]] = None,
               exporter: Optional[Callable[[SpanRecord], None]] = None) -> Iterator[SpanRecord]:
    """Create a child span, set it current, and export it on exit if requested."""
    parent = context or _current.get() or TraceContext.new()
    child = parent.child()
    span = SpanRecord(trace_id=child.trace_id, span_id=child.span_id,
                      parent_span_id=parent.span_id, name=name,
                      start_ms=now_ms(), attributes=attributes or {})
    token = _current.set(child)
    try:
        yield span
        if span.end_ms is None:
            span.finish("ok")
    except Exception as exc:
        span.finish("error", str(exc))
        raise
    finally:
        _current.reset(token)
        if exporter is not None:
            exporter(span)
