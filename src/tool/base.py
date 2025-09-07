"""Generic tool interface and registry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict

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

    def register(self, tool: Tool) -> None:
        self._tools[tool.spec.name] = tool

    def get(self, name: str) -> Tool:
        return self._tools[name]

    def run(self, name: str, **kwargs: Any) -> Dict[str, Any]:
        tool = self.get(name)
        return tool.run(**kwargs)

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
        return out
