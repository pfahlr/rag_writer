"""Generic tool interface and registry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict

try:  # pragma: no cover - optional dependency
    import mcp_client  # type: ignore
except Exception:  # pragma: no cover - MCP optional
    mcp_client = None  # type: ignore

try:
    import jsonschema
except Exception:  # pragma: no cover - jsonschema optional
    jsonschema = None  # type: ignore


@dataclass
class ToolSpec:
    """Specification for a tool."""

    name: str
    description: str
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]


class Tool:
    """Encapsulates a callable tool with schema validation."""

    def __init__(self, spec: ToolSpec, func: Callable[..., Dict[str, Any]]):
        self.spec = spec
        self._func = func

    def run(self, **kwargs: Any) -> Dict[str, Any]:
        """Execute the tool with validated input/output."""
        if jsonschema is not None:
            jsonschema.validate(kwargs, self.spec.input_schema)
        result = self._func(**kwargs)
        if jsonschema is not None:
            jsonschema.validate(result, self.spec.output_schema)
        return result


class ToolRegistry:
    """Registry for discovering and executing tools."""

    def __init__(self) -> None:
        self._tools: Dict[str, Tool] = {}
        self._mcp_tools: Dict[str, ToolSpec] = {}
        self._mcp_client: Any | None = None

    def register(self, tool: Tool) -> None:
        self._tools[tool.spec.name] = tool

    def get(self, name: str) -> Tool:
        return self._tools[name]

    def register_mcp_server(self, server_url: str) -> None:
        """Connect to an MCP server and register its tools."""

        if mcp_client is None:  # pragma: no cover - MCP optional
            raise RuntimeError("mcp_client dependency not available")

        client = mcp_client.connect(server_url)
        for t in client.fetch_tools():
            spec = ToolSpec(
                name=t["name"],
                description=t.get("description", ""),
                input_schema=t.get("input_schema", {}),
                output_schema=t.get("output_schema", {}),
            )
            self._mcp_tools[spec.name] = spec
        self._mcp_client = client

    def run(self, name: str, **kwargs: Any) -> Dict[str, Any]:
        if name in self._tools:
            tool = self._tools[name]
            return tool.run(**kwargs)

        if name in self._mcp_tools:
            spec = self._mcp_tools[name]
            if jsonschema is not None:
                jsonschema.validate(kwargs, spec.input_schema)
            if self._mcp_client is None:  # pragma: no cover - register first
                raise RuntimeError("MCP server not registered")
            result = self._mcp_client.call_tool(name, **kwargs)
            if jsonschema is not None:
                jsonschema.validate(result, spec.output_schema)
            return result

        raise KeyError(name)

    def list_tools(self) -> Dict[str, Tool]:
        return dict(self._tools)

    def describe(self) -> list[Dict[str, Any]]:
        """Return summary information for all registered tools.

        Each entry contains the tool's name, description and input schema
        so that an LLM can understand how to invoke the tool.
        """

        out: list[Dict[str, Any]] = []
        for tool in self._tools.values():
            out.append(
                {
                    "name": tool.spec.name,
                    "description": tool.spec.description,
                    "input_schema": tool.spec.input_schema,
                }
            )
        for spec in self._mcp_tools.values():
            out.append(
                {
                    "name": spec.name,
                    "description": spec.description,
                    "input_schema": spec.input_schema,
                }
            )
        return out
