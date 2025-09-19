import json
import subprocess
import sys

import pytest
from httpx import ASGITransport, AsyncClient

from src.tool.mcp_app import app


def _send(proc, msg):
    data = json.dumps(msg).encode()
    frame = f"Content-Length: {len(data)}\r\n\r\n".encode() + data
    proc.stdin.write(frame)
    proc.stdin.flush()
    header = {}
    while True:
        line = proc.stdout.readline()
        if line == b"":
            raise RuntimeError("no response")
        if line == b"\r\n":
            break
        if line.lower().startswith(b"content-length:"):
            header["length"] = int(line.split(b":")[1].strip())
    body = proc.stdout.read(header["length"])
    return json.loads(body.decode())


@pytest.mark.anyio
async def test_http_php_tool():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/mcp/tool/util.php.echo", json={"message": "hello"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["body"] == {"ok": True, "data": {"echo": "hello"}}


def test_stdio_php_tool():
    proc = subprocess.Popen(
        [sys.executable, "-m", "src.tool.mcp_stdio"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        resp = _send(
            proc,
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "mcp.tool.invoke",
                "params": {"tool": "util.php.echo", "payload": {"message": "hello"}},
            },
        )
        assert resp["result"] == {"ok": True, "data": {"echo": "hello"}}
    finally:
        proc.terminate()
        proc.wait(timeout=3)
