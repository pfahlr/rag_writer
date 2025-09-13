import json
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from src.tool.mcp_app import app
from src.tool.service import STUB_OUTPUTS


@pytest.fixture()
def tool_payload():
    path = Path(__file__).parent / "fixtures" / "tool_payload.json"
    return json.loads(path.read_text())


@pytest.mark.anyio
async def test_http_tool_determinism(tool_payload):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        bodies = []
        for _ in range(2):
            resp = await client.post(
                f"/mcp/tool/{tool_payload['tool']}", json=tool_payload["payload"]
            )
            assert resp.status_code == 200
            data = resp.json()
            assert "meta" in data and "body" in data
            assert data["body"] == STUB_OUTPUTS[tool_payload["tool"]]
            bodies.append(data["body"])
        assert bodies[0] == bodies[1]
