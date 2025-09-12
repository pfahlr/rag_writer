from __future__ import annotations

import json
from typing import Iterable, Mapping, Union, Any

from src.tool.base import ToolRegistry


def generate_tool_prompt(
    registry_or_tools: Union[ToolRegistry, Iterable[Mapping[str, Any]]],
) -> str:
    """Generate a system prompt describing available tools.

    Accepts either a :class:`~src.tool.base.ToolRegistry` instance or an iterable
    of tool descriptors such as the raw output from ``mcp_client.fetch_tools``.
    The returned prompt lists each tool and instructs the model to respond
    **only** with JSON in one of two forms::

        {"tool": "<name>", "args": {...}}
        {"final": "<answer>"}
    """

    if isinstance(registry_or_tools, ToolRegistry):
        tools = registry_or_tools.describe()
    else:
        tools = list(registry_or_tools)

    lines = ["You can use the following tools:"]
    for tool in tools:
        schema = json.dumps(
            tool.get("input_schema") or tool.get("inputSchema") or {},
            separators=(",", ":"),
        )
        lines.append(f"- {tool['name']}: {tool['description']} | args schema: {schema}")

    instructions = (
        "\nRespond ONLY with JSON. To call a tool, respond with\n"
        '{"tool": "<tool_name>", "args": {<tool arguments>}}\n'
        "To provide the final answer, respond with\n"
        '{"final": "<answer text>"}'
    )
    return "\n".join(lines) + "\n\n" + instructions + "\n"
