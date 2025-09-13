from __future__ import annotations

from time import perf_counter
from typing import Dict

from fastapi import Depends, FastAPI

from .deps import size_guard
from .models import Envelope, Meta
from .service import discover, get_prompt, invoke_tool

app = FastAPI()


@app.get("/mcp/discover")
async def mcp_discover() -> Dict[str, object]:
    return discover()


@app.get("/mcp/prompt/{domain}/{name}/{major}")
async def mcp_prompt(domain: str, name: str, major: str) -> Dict[str, object]:
    return get_prompt(domain, name, major)


@app.post("/mcp/tool/{tool_name}", response_model=Envelope)
async def mcp_tool(
    tool_name: str, payload: Dict[str, object], _: None = Depends(size_guard)
) -> Envelope:
    start = perf_counter()
    result = invoke_tool(tool_name, payload)
    duration_ms = int((perf_counter() - start) * 1000)
    meta = Meta(durationMs=duration_ms)
    return Envelope(meta=meta, body=result)
