from __future__ import annotations

from typing import Any, Dict


def discover() -> Dict[str, Any]:
    return {"mcp": "stub", "endpoints": ["discover", "prompt", "tool"]}


def get_prompt(domain: str, name: str, major: str) -> Dict[str, Any]:
    return {
        "body": f"{domain}-{name}-{major}",
        "spec": {"domain": domain, "name": name, "major": major},
    }


def invoke_tool(tool: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    return {"tool": tool, "payload": payload}
