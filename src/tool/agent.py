"""Simple LLM-powered tool agent."""

from __future__ import annotations

import json
from typing import Dict, List

from .base import ToolRegistry
from .prompts import generate_tool_prompt


def run_agent(
    llm,
    registry: ToolRegistry,
    question: str,
    max_iters: int = 5,
    system_prompt: str | None = None,
) -> str:
    """Run a simple agent loop until a final answer is produced.

    Args:
        llm: Callable LLM accepting a list of ``{"role", "content"}`` messages
            and returning a string response.
        registry: The ``ToolRegistry`` containing available tools.
        question: Initial user question to kick off the conversation.
        max_iters: Maximum number of LLM/tool iterations.
        system_prompt: Optional system prompt. If not provided, one is
            generated from ``registry`` via :func:`generate_tool_prompt`.

    Returns:
        Final answer string produced by the agent.

    The LLM is expected to return JSON of the form:
        ``{"tool": name, "args": {...}}`` to call a tool or
        ``{"final": text}`` to terminate.

    Minimal error handling is performed for invalid tool names
    or malformed schemas: errors are appended to the conversation
    allowing the LLM to recover.
    """

    if system_prompt is None:
        system_prompt = generate_tool_prompt(registry)

    messages: List[Dict[str, str]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]

    for _ in range(max_iters):
        # Call the LLM
        if hasattr(llm, "invoke"):
            response = llm.invoke(messages)  # type: ignore[attr-defined]
        else:
            response = llm(messages)  # type: ignore[call-arg]

        if isinstance(response, dict) and "content" in response:
            text = response["content"]
        else:
            text = str(response)

        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            # Append error message and continue the loop
            messages.append({"role": "assistant", "content": text})
            messages.append(
                {
                    "role": "user",
                    "content": json.dumps({"error": f"Invalid JSON: {e}"}),
                }
            )
            continue

        if "final" in data:
            return str(data["final"])
        if "tool" not in data or "args" not in data:
            messages.append({"role": "assistant", "content": text})
            messages.append(
                {
                    "role": "user",
                    "content": json.dumps(
                        {"error": "LLM output missing 'tool' or 'args'"}
                    ),
                }
            )
            continue

        tool_name = str(data["tool"])
        args = data.get("args", {})
        if not isinstance(args, dict):
            args = {}

        try:
            result = registry.run(tool_name, **args)
        except KeyError:
            result = {"error": f"Unknown tool: {tool_name}"}
        except Exception as e:  # pragma: no cover - guardrail
            result = {"error": f"Tool {tool_name} failed: {e}"}

        # Append LLM tool call and tool result to conversation
        messages.append({"role": "assistant", "content": text})
        messages.append({"role": "user", "content": json.dumps(result)})

    raise RuntimeError("Agent did not produce a final answer")
