import pytest
from fastapi import HTTPException


def test_invoke_tool_stub_outputs_valid():
    from src.tool.service import invoke_tool

    assert invoke_tool("web_search_query", {}) == {"ok": True, "data": {"results": []}}


def test_invoke_tool_schema_violation(monkeypatch):
    from src.tool import service

    monkeypatch.setitem(service.STUB_OUTPUTS, "web_search_query", {})
    service._CACHE.clear()
    with pytest.raises(HTTPException) as exc:
        service.invoke_tool("web_search_query", {})
    assert exc.value.status_code == 400
    assert exc.value.detail["error"] == "schema_validation_failed"


def test_invoke_tool_validates_after_runner(monkeypatch):
    from src.tool import service

    def fake_run_tool(tool, payload):
        return {}

    monkeypatch.setattr(service, "_run_tool", fake_run_tool)
    service._CACHE.clear()
    with pytest.raises(HTTPException) as exc:
        service.invoke_tool("web_search_query", {})
    assert exc.value.status_code == 400
    assert exc.value.detail["error"] == "schema_validation_failed"


def test_schemas_compile_and_validate():
    from src.tool import schemas
    import jsonschema

    schemas.validate_tool_output("web_search_query", {"results": []})
    with pytest.raises(jsonschema.ValidationError):
        schemas.validate_tool_output("web_search_query", {})
