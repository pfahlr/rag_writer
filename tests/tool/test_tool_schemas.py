import pytest
from fastapi import HTTPException


def test_invoke_tool_stub_outputs_valid():
    from src.tool.service import invoke_tool

    assert invoke_tool("web_search_query", {}) == {"results": []}


def test_invoke_tool_schema_violation(monkeypatch):
    from src.tool import service

    monkeypatch.setitem(service.STUB_OUTPUTS, "web_search_query", {})
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
