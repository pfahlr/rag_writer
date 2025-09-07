"""Model Context Protocol server exposing tools from ``ToolRegistry``."""

from __future__ import annotations

from typing import Any, Dict, List

from .base import ToolRegistry

try:  # pragma: no cover - optional dependency
    from mcp.server import NotificationOptions, Server
    from mcp.server.stdio import stdio_server
    import mcp.types as types
except Exception:  # pragma: no cover - library not installed
    Server = None  # type: ignore
    stdio_server = None  # type: ignore
    types = None  # type: ignore


def _as_mcp_tools(registry: ToolRegistry) -> List["types.Tool"]:
    """Convert registry tools to MCP ``Tool`` objects."""
    tools: List["types.Tool"] = []
    for tool in registry.list_tools().values():
        spec = tool.spec
        tools.append(
            types.Tool(
                name=spec.name,
                description=spec.description,
                inputSchema=spec.input_schema,
                outputSchema=spec.output_schema,
            )
        )
    return tools


def create_server(registry: ToolRegistry) -> "Server":
    """Create an MCP ``Server`` wired to the ``ToolRegistry``."""
    if Server is None or types is None:
        raise RuntimeError("mcp library is required to run the tool server")

    server = Server("rag-writer-tool")

    @server.list_tools()
    async def list_tools() -> List["types.Tool"]:
        return _as_mcp_tools(registry)

    @server.call_tool()
    async def call_tool(name: str, arguments: Dict[str, Any] | None) -> Dict[str, Any]:
        return registry.run(name, **(arguments or {}))

    return server


def serve(registry: ToolRegistry | None = None) -> None:
    """Run the tool server using stdio transport."""
    if Server is None or stdio_server is None or types is None:
        raise RuntimeError("mcp library is required to run the tool server")

    registry = registry or ToolRegistry()
    server = create_server(registry)

    async def _run() -> None:
        async with stdio_server() as (read, write):
            options = server.create_initialization_options(NotificationOptions())
            await server.run(read, write, options)

    import anyio

    anyio.run(_run)


def main() -> None:
    """CLI entry point to start the MCP tool server."""
    serve()


if __name__ == "__main__":  # pragma: no cover
    main()
