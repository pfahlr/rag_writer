"""Tracing utilities shared across CLI scripts."""

from __future__ import annotations

import json
import os
import re
import sys
import time
import uuid
from contextlib import AbstractContextManager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

TRACE_PREFIX = "TRACE: "
MAX_TEXT_LENGTH = 3000

_SECRET_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?i)sk-[a-z0-9]{20,}"),
    re.compile(r"(?i)aws(.{0,20})?(secret|key)[^=]*=\S+"),
    re.compile(r"(?i)authorization:\s*bearer\s+\S+"),
    re.compile(r"(?i)(password|passwd|pwd)\s*[:=]\s*\S+"),
)


def _redact_string(value: str) -> str:
    truncated = value
    if len(truncated) > MAX_TEXT_LENGTH:
        truncated = truncated[:MAX_TEXT_LENGTH] + "…"
    for pattern in _SECRET_PATTERNS:
        truncated = pattern.sub("REDACTED", truncated)
    return truncated


def redact(value: Any) -> Any:
    """Recursively redact secrets and truncate long strings."""

    if isinstance(value, dict):
        return {k: redact(v) for k, v in value.items()}
    if isinstance(value, list):
        return [redact(v) for v in value]
    if isinstance(value, tuple):
        return tuple(redact(v) for v in value)
    if isinstance(value, str):
        return _redact_string(value)
    return value


def _utc_timestamp() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def _ensure_parent(event: dict[str, Any]) -> None:
    if "parent" not in event:
        event["parent"] = "root"


@dataclass
class TraceEmitter(AbstractContextManager["TraceEmitter"]):
    """Simple TRACE protocol emitter."""

    enabled: bool = False
    tee_path: str | None = None
    redact_output: bool = True
    stream: Any = sys.stderr

    def __post_init__(self) -> None:
        self._tee_handle = None
        if self.tee_path:
            path = Path(self.tee_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            self._tee_handle = path.open("a", encoding="utf-8")

    def close(self) -> None:
        if self._tee_handle:
            self._tee_handle.close()
            self._tee_handle = None

    def __exit__(self, *exc: object) -> None:  # type: ignore[override]
        self.close()

    def make_span(self, name: str, parent: str | None = None) -> str:
        span = uuid.uuid4().hex[:8]
        if parent is None:
            parent = "root"
        return span

    def emit(self, event: dict[str, Any]) -> None:
        if not self.enabled:
            return
        payload = dict(event)
        payload.setdefault("v", 1)
        payload.setdefault("ts", _utc_timestamp())
        if "span" not in payload:
            payload["span"] = uuid.uuid4().hex[:8]
        _ensure_parent(payload)
        if self.redact_output:
            payload = redact(payload)
        line = TRACE_PREFIX + json.dumps(payload, ensure_ascii=False)
        self.stream.write(line + "\n")
        self.stream.flush()
        if self._tee_handle:
            self._tee_handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
            self._tee_handle.flush()


def configure_emitter(flag: bool, *, trace_file: str | None = None) -> TraceEmitter:
    env_enabled = os.getenv("RAG_TRACE") in {"1", "true", "TRUE", "yes", "on"}
    enabled = flag or env_enabled
    return TraceEmitter(enabled=enabled, tee_path=trace_file)


def timed_span(emitter: TraceEmitter, name: str, parent: str | None = None) -> "_SpanContext":
    return _SpanContext(emitter=emitter, name=name, parent=parent)


class _SpanContext(AbstractContextManager["_SpanContext"]):
    def __init__(self, *, emitter: TraceEmitter, name: str, parent: str | None) -> None:
        self.emitter = emitter
        self.name = name
        self.parent = parent
        self.span_id = emitter.make_span(name, parent=parent) if emitter.enabled else uuid.uuid4().hex[:8]
        self._start = None

    def __enter__(self) -> "_SpanContext":
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # type: ignore[override]
        if not self.emitter.enabled:
            return
        latency_ms = None
        if self._start is not None:
            latency_ms = (time.perf_counter() - self._start) * 1000
        detail: Dict[str, Any] = {"message": f"Span {self.name} completed"}
        metrics: Dict[str, Any] = {}
        if latency_ms is not None:
            metrics["latency_ms"] = round(latency_ms, 2)
        event = {
            "span": self.span_id,
            "parent": self.parent or "root",
            "type": "info" if exc_type is None else "error",
            "name": self.name,
            "detail": detail,
            "metrics": metrics,
        }
        if exc_type is not None:
            detail["exception"] = repr(exc)
        self.emitter.emit(event)

