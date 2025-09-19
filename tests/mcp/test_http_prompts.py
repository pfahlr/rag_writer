import json
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from src.tool.mcp_app import app


@pytest.fixture()
def prompt_info():
    path = Path(__file__).parent / "fixtures" / "prompt_info.json"
    return json.loads(path.read_text())


@pytest.mark.anyio
async def test_http_prompt_determinism(prompt_info):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        results = []
        for _ in range(2):
            resp = await client.get(
                f"/mcp/prompt/{prompt_info['domain']}/{prompt_info['name']}/{prompt_info['major']}"
            )
            assert resp.status_code == 200
            data = resp.json()
            assert "body" in data and "spec" in data
            results.append(data)
        assert results[0] == results[1]
