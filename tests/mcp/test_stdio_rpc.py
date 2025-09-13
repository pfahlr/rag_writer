import json
from pathlib import Path

import anyio
import pytest

from src.tool.service import STUB_OUTPUTS


@pytest.fixture()
def tool_payload():
    path = Path(__file__).parent / "fixtures" / "tool_payload.json"
    return json.loads(path.read_text())


async def _send_request(process, message):
    data = json.dumps(message).encode("utf-8")
    header = f"Content-Length: {len(data)}\r\n\r\n".encode("utf-8")
    await process.stdin.send(header + data)
    header_bytes = b""
    while b"\r\n\r\n" not in header_bytes:
        header_bytes += await process.stdout.receive(1)
    headers = header_bytes.decode()
    content_length = 0
    for line in headers.split("\r\n"):
        if line.lower().startswith("content-length:"):
            content_length = int(line.split(":")[1].strip())
            break
    body = b""
    while len(body) < content_length:
        body += await process.stdout.receive(content_length - len(body))
    assert content_length == len(body)
    return json.loads(body.decode("utf-8"))


@pytest.mark.anyio
async def test_stdio_rpc_tool_determinism(tool_payload):
    process = await anyio.open_process(["python", "-m", "src.tool.mcp_stdio"])
    async with process:
        msg = {
            "jsonrpc": "2.0",
            "method": "mcp.tool.invoke",
            "params": {
                "tool": tool_payload["tool"],
                "payload": tool_payload["payload"],
            },
            "id": 1,
        }
        first = await _send_request(process, msg)
        msg["id"] = 2
        second = await _send_request(process, msg)
        assert first["result"] == second["result"] == STUB_OUTPUTS[tool_payload["tool"]]
