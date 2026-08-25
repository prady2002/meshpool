"""Structured logging with a consistent, greppable format.

Set ``MESHPOOL_LOG_JSON=1`` for machine-readable JSON logs (useful when shipping
to a log aggregator at enterprise scale); otherwise a compact console format.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from typing import Any


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": round(time.time(), 3),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        for key, val in getattr(record, "extra_fields", {}).items():
            payload[key] = val
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


class _ConsoleFormatter(logging.Formatter):
    COLORS = {"DEBUG": "\x1b[38;5;244m", "INFO": "\x1b[38;5;39m",
              "WARNING": "\x1b[38;5;214m", "ERROR": "\x1b[38;5;196m",
              "CRITICAL": "\x1b[48;5;196m"}
    RESET = "\x1b[0m"

    def __init__(self, color: bool = True) -> None:
        super().__init__()
        self.color = color and sys.stderr.isatty()

    def format(self, record: logging.LogRecord) -> str:
        ts = time.strftime("%H:%M:%S", time.localtime(record.created))
        lvl = record.levelname
        name = record.name.replace("meshpool.", "")
        extra = getattr(record, "extra_fields", {})
        suffix = ""
        if extra:
            suffix = " " + " ".join(f"{k}={v}" for k, v in extra.items())
        line = f"{ts} {lvl:<7} {name:<16} {record.getMessage()}{suffix}"
        if self.color:
            c = self.COLORS.get(lvl, "")
            line = f"{c}{line}{self.RESET}"
        return line


class _Adapter(logging.LoggerAdapter):
    """Allows ``log.info("msg", node="n-1", job="j-2")`` structured kwargs."""

    def process(self, msg: str, kwargs: Any) -> tuple[str, Any]:
        std = {"exc_info", "stack_info", "stacklevel", "extra"}
        fields = {k: kwargs.pop(k) for k in list(kwargs) if k not in std}
        extra = kwargs.setdefault("extra", {})
        extra["extra_fields"] = fields
        return msg, kwargs


_configured = False


def _configure_root() -> None:
    global _configured
    if _configured:
        return
    handler = logging.StreamHandler(sys.stderr)
    if os.environ.get("MESHPOOL_LOG_JSON") == "1":
        handler.setFormatter(_JsonFormatter())
    else:
        handler.setFormatter(_ConsoleFormatter())
    root = logging.getLogger("meshpool")
    root.handlers[:] = [handler]
    root.setLevel(os.environ.get("MESHPOOL_LOG_LEVEL", "INFO").upper())
    root.propagate = False
    _configured = True


def get_logger(name: str) -> _Adapter:
    _configure_root()
    logger = logging.getLogger(f"meshpool.{name}")
    return _Adapter(logger, {})
