import sys
from types import ModuleType, SimpleNamespace

from fastapi.testclient import TestClient

# Stub optional MCP dependency so importing src.tool succeeds
mcp_stub = ModuleType("mcp")
mcp_stub.__path__ = []
mcp_stub.ClientSession = object
client_mod = ModuleType("mcp.client")
client_mod.stdio = SimpleNamespace(StdioServerParameters=object, stdio_client=None)
mcp_stub.client = client_mod
sys.modules.setdefault("mcp", mcp_stub)
sys.modules.setdefault("mcp.client", client_mod)

from src.tool.mcp_app import app  # noqa: E402

client = TestClient(app)


def test_discover_endpoint():
    res = client.get("/mcp/discover")
    assert res.status_code == 200
    data = res.json()
    # server metadata
    assert data["server"]["name"] == "rag-writer"
    assert "version" in data["server"]

    # tool metadata includes example markdown tool
    tools = {t["name"]: t for t in data["tools"]}
    assert "markdown" in tools
    md = tools["markdown"]
    assert md["inputSchemaRef"].endswith("markdown.input.schema.json")
    assert md["caps"] == {
        "timeoutMs": 1000,
        "maxInputBytes": 4096,
        "maxOutputBytes": 4096,
        "network": [],
    }

    # prompts are exposed with versions
    prompts = {p["id"]: p for p in data["prompts"]}
    assert "writing/sectioned_draft" in prompts
    versions = prompts["writing/sectioned_draft"]["versions"]
    assert any(v["major"] == 3 for v in versions)

    # health block present
    assert data["health"]["status"] == "ok"
    assert isinstance(data["health"]["uptimeSec"], int)


def test_prompt_endpoint():
    res = client.get("/mcp/prompt/writing/sectioned_draft/3")
    assert res.status_code == 200
    data = res.json()
    assert "Write a sectioned draft" in data["body"]
    assert data["spec"]["inputs"]["properties"]["topic"]["type"] == "string"


def test_prompt_not_found():
    res = client.get("/mcp/prompt/writing/does_not_exist/1")
    assert res.status_code == 404


def test_tool_endpoint_envelope():
    res = client.post("/mcp/tool/echo", json={"hello": "world"})
    assert res.status_code == 200
    out = res.json()
    assert "meta" in out and "traceId" in out["meta"] and "durationMs" in out["meta"]
    assert out["body"] == {
        "ok": True,
        "data": {"tool": "echo", "payload": {"hello": "world"}},
    }
    assert out["warnings"] == []


def test_tool_endpoint_size_guard():
    payload = {"data": "x" * 70000}
    res = client.post("/mcp/tool/echo", json=payload)
    assert res.status_code == 413
