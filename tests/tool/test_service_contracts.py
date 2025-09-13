import json
import time

import pytest
from fastapi import HTTPException

from src.tool import service


def test_invoke_tool_idempotent(monkeypatch):
    calls = {"count": 0}

    def fake_run(tool, payload):
        calls["count"] += 1
        return {"call": calls["count"]}

    monkeypatch.setattr(service, "_run_tool", fake_run)
    first = service.invoke_tool("echo", {"a": 1, "b": 2})
    second = service.invoke_tool("echo", {"b": 2, "a": 1})
    assert first == second == {"ok": True, "data": {"call": 1}}
    assert calls["count"] == 1


def test_invoke_tool_input_limit():
    big_payload = {"data": "x" * 70000}
    with pytest.raises(HTTPException) as exc:
        service.invoke_tool("echo", big_payload)
    assert exc.value.status_code == 400
    assert exc.value.detail["error"] == "INVALID_INPUT"


def test_invoke_tool_output_limit(monkeypatch):
    def big_run(tool, payload):
        return {"data": "x" * 300000}

    monkeypatch.setattr(service, "_run_tool", big_run)
    with pytest.raises(HTTPException) as exc:
        service.invoke_tool("echo", {})
    assert exc.value.status_code == 500
    assert exc.value.detail["error"] == "INTERNAL"
    assert exc.value.detail["details"]["size"] > 256 * 1024


def test_invoke_tool_timeout(monkeypatch):
    def slow_run(tool, payload):
        time.sleep(0.2)
        return {}

    monkeypatch.setattr(service, "_run_tool", slow_run)
    with pytest.raises(HTTPException) as exc:
        service.invoke_tool("slow", {}, timeout=0.05)
    assert exc.value.detail["error"] == "TIMEOUT"
