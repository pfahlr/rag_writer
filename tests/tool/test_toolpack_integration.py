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
