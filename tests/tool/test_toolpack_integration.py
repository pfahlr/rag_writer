import pytest
from fastapi import HTTPException

from src.tool import service


def test_toolpack_discovery_includes_example():
    info = service.discover()
    assert "tools" in info
    assert "markdown" in info["tools"]


def test_toolpack_invocation_and_schema_validation(monkeypatch):
    res = service.invoke_tool("markdown", {"text": "hi"})
    assert res == {"ok": True, "data": {"html": "<p>hi</p>"}}

    # Ensure schema validation triggers on bad output
    import tools.example.markdown as md

    def bad_run(text: str) -> dict:  # missing required 'html'
        return {"oops": text}

    monkeypatch.setattr(md, "run", bad_run)
    with pytest.raises(HTTPException):
        service.invoke_tool("markdown", {"text": "bye"})


def test_node_toolpack_invocation(monkeypatch):
    service._CACHE.clear()
    res = service.invoke_tool("node_echo", {"text": "hi", "prefix": "--"})
    assert res == {"ok": True, "data": {"echo": "--hi"}}


def test_http_toolpack_invocation_templating_and_cache(monkeypatch):
    service._CACHE.clear()
    calls = []

    class Resp:
        def __init__(self, data):
            self._data = data

        def json(self):
            return self._data

    def fake_post(url, json, headers, timeout):
        calls.append((url, json, headers))
        return Resp({"data": {"echo": json["input"]["text"]}})

    import requests

    monkeypatch.setattr(requests, "post", fake_post)

    payload = {"path": "foo", "token": "abc", "text": "hi"}
    res1 = service.invoke_tool("http_echo", payload)
    assert res1 == {"ok": True, "data": {"echo": "hi"}}
    assert calls[0][0] == "https://example.com/foo"
    assert calls[0][2]["X-Token"] == "abc"

    payload2 = {"path": "foo", "token": "def", "text": "hi"}
    res2 = service.invoke_tool("http_echo", payload2)
    assert res2 == res1
    assert len(calls) == 1  # cache hit via templating.cacheKey

    def bad_post(url, json, headers, timeout):
        return Resp({"data": {"oops": "nope"}})

    monkeypatch.setattr(requests, "post", bad_post)
    with pytest.raises(HTTPException):
        service.invoke_tool("http_echo", {"path": "bar", "token": "xyz", "text": "hi"})
