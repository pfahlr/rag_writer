import json
import subprocess
import sys


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


def test_stdio_discover_and_invoke():
    proc = subprocess.Popen(
        [sys.executable, "-m", "src.tool.mcp_stdio"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        resp1 = _send(proc, {"jsonrpc": "2.0", "id": 1, "method": "mcp.discover"})
        assert resp1["result"]["server"]["name"] == "rag-writer"
        resp2 = _send(
            proc,
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "mcp.tool.invoke",
                "params": {"tool": "echo", "payload": {"hello": "world"}},
            },
        )
        assert resp2["result"] == {
            "ok": True,
            "data": {"tool": "echo", "payload": {"hello": "world"}},
        }
    finally:
        proc.terminate()
        proc.wait(timeout=3)
