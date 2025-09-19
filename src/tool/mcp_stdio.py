"""Minimal STDIO JSON-RPC server for MCP-style methods."""

from __future__ import annotations

import json
import sys
from time import perf_counter
from typing import Any, Dict

from .models import Envelope, ErrorItem, Meta
from .service import discover, get_prompt, invoke_tool


def _read_message() -> Dict[str, Any] | None:
    """Read a single JSON-RPC message framed with LSP headers."""
    content_length: int | None = None
    while True:
        line = sys.stdin.buffer.readline()
        if not line:
            return None
        if line == b"\r\n":
            break
        if line.lower().startswith(b"content-length:"):
            content_length = int(line.split(b":")[1].strip())
    if content_length is None:
        return None
    body = sys.stdin.buffer.read(content_length)
    if not body:
        return None
    return json.loads(body.decode("utf-8"))


def _write_message(msg: Dict[str, Any]) -> None:
    data = json.dumps(msg).encode("utf-8")
    header = f"Content-Length: {len(data)}\r\n\r\n".encode("utf-8")
    sys.stdout.buffer.write(header)
    sys.stdout.buffer.write(data)
    sys.stdout.buffer.flush()


def _handle_request(req: Dict[str, Any]) -> None:
    req_id = req.get("id")
    method = req.get("method")
    params: Dict[str, Any] = req.get("params", {})
    start = perf_counter()
    try:
        if method == "mcp.discover":
            result = discover()
        elif method == "mcp.prompt.get":
            result = get_prompt(params["domain"], params["name"], params["major"])
        elif method == "mcp.tool.invoke":
            result = invoke_tool(params["tool"], params.get("payload", {}))
        else:
            raise ValueError("Method not found")
        _write_message({"jsonrpc": "2.0", "id": req_id, "result": result})
    except Exception as exc:  # pragma: no cover - smoke tests cover success
        duration_ms = int((perf_counter() - start) * 1000)
        envelope = Envelope(
            meta=Meta(durationMs=duration_ms),
            error=ErrorItem(code="internal_error", message=str(exc)),
        )
        error_obj = {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {
                "code": -32603,
                "message": str(exc),
                "data": envelope.model_dump(),
            },
        }
        _write_message(error_obj)


def serve() -> None:
    """Serve requests from STDIO until EOF."""
    while True:
        req = _read_message()
        if req is None:
            break
        _handle_request(req)


def main() -> None:  # pragma: no cover - thin wrapper
    serve()


if __name__ == "__main__":  # pragma: no cover
    main()
