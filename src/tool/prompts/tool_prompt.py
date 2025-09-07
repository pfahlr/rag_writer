from __future__ import annotations

import json
from typing import List, Dict

from ..base import ToolRegistry


def generate_tool_prompt(registry: ToolRegistry) -> str:
    """Generate a system prompt describing available tools.

    The returned prompt lists each tool from ``registry.describe()`` and
    instructs the model to respond **only** with JSON in one of two forms::

        {"tool": "<name>", "args": {...}}
        {"final": "<answer>"}
    """

    tools: List[Dict[str, object]] = registry.describe()
    lines = ["You can use the following tools:"]
    for tool in tools:
        schema = json.dumps(tool.get("input_schema", {}), separators=(",", ":"))
        schema = schema.replace("{", "{{").replace("}", "}}")
        lines.append(
            f"- {tool['name']}: {tool['description']} | args schema: {schema}"
        )

    instructions = (
        "\nRespond ONLY with JSON. To call a tool, respond with\n"
        '{"tool": "<tool_name>", "args": {<tool arguments>}}\n'
        "To provide the final answer, respond with\n"
        '{"final": "<answer text>"}'
    ).replace("{", "{{").replace("}", "}}")
    return "\n".join(lines) + "\n\n" + instructions + "\n"
