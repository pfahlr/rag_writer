from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml
from fastapi import HTTPException

REGISTRY_PATH = Path("prompts/REGISTRY.yaml")
PACKS_DIR = Path("prompts/packs")


def _load_registry() -> Dict[str, Dict[str, list[int]]]:
    if not REGISTRY_PATH.exists():
        return {}
    data = yaml.safe_load(REGISTRY_PATH.read_text())
    return data or {}


def discover() -> Dict[str, Any]:
    registry = _load_registry()
    return {
        "mcp": "stub",
        "endpoints": ["discover", "prompt", "tool"],
        "prompts": registry,
    }


def get_prompt(domain: str, name: str, major: str) -> Dict[str, Any]:
    registry = _load_registry()
    versions = registry.get(domain, {}).get(name)
    if versions is None or int(major) not in versions:
        raise HTTPException(status_code=404, detail="Prompt not found")

    body_path = PACKS_DIR / domain / f"{name}.v{major}.md"
    spec_path = PACKS_DIR / domain / f"{name}.spec.yaml"

    if not body_path.exists() or not spec_path.exists():
        raise HTTPException(status_code=404, detail="Prompt not found")

    body = body_path.read_text()
    spec = yaml.safe_load(spec_path.read_text())

    return {"body": body, "spec": spec}


def invoke_tool(tool: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    return {"tool": tool, "payload": payload}
