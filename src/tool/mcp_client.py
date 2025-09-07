"""Client helpers for interacting with MCP servers."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Dict

from mcp import ClientSession
from mcp.client import stdio

from .base import ToolSpec


@asynccontextmanager
async def connect(server_url: str) -> AsyncIterator[ClientSession]:
    """Establish an MCP session using a stdio server.

    Parameters
    ----------
    server_url: str
        Path to the server executable. Only stdio servers are currently
        supported.

    Yields
    ------
    ClientSession
        An initialized MCP client session.
    """
    params = stdio.StdioServerParameters(command=server_url)
    async with stdio.stdio_client(params) as (read_stream, write_stream):
        session = ClientSession(read_stream, write_stream)
        await session.initialize()
        yield session


async def fetch_tools(session: ClientSession) -> list[ToolSpec]:
    """Fetch available tools from the MCP server.

    Parameters
    ----------
    session: ClientSession
        Active MCP session.

    Returns
    -------
    list[ToolSpec]
        Tool specifications advertised by the server.
    """
    result = await session.list_tools()
    specs: list[ToolSpec] = []
    for tool in result.tools:
        specs.append(
            ToolSpec(
                name=tool.name,
                description=tool.description or "",
                input_schema=tool.inputSchema,
                output_schema=tool.outputSchema or {"type": "object"},
            )
        )
    return specs


async def call_tool(
    session: ClientSession, name: str, args: Dict[str, Any] | None = None
) -> Dict[str, Any]:
    """Invoke a tool on the MCP server and return its structured content."""
    result = await session.call_tool(name, args or {})
    if result.isError:
        raise RuntimeError("Tool call returned an error")
    if result.structuredContent is not None:
        return result.structuredContent
    return {"content": [block.model_dump() for block in result.content]}
