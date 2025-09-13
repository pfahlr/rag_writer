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
    assert res.json()["mcp"] == "stub"


def test_prompt_endpoint():
    res = client.get("/mcp/prompt/foo/bar/1")
    assert res.status_code == 200
    data = res.json()
    assert data["spec"] == {"domain": "foo", "name": "bar", "major": "1"}
    assert data["body"] == "foo-bar-1"


def test_tool_endpoint_envelope():
    res = client.post("/mcp/tool/echo", json={"hello": "world"})
    assert res.status_code == 200
    out = res.json()
    assert "meta" in out and "traceId" in out["meta"] and "durationMs" in out["meta"]
    assert out["body"] == {"tool": "echo", "payload": {"hello": "world"}}
    assert out["warnings"] == []


def test_tool_endpoint_size_guard():
    payload = {"data": "x" * 70000}
    res = client.post("/mcp/tool/echo", json=payload)
    assert res.status_code == 413
