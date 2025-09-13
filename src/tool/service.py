from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import json
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from time import time

import yaml
from fastapi import HTTPException
from jsonschema import ValidationError

from .schemas import validate_tool_output

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


STUB_OUTPUTS: Dict[str, Dict[str, Any]] = {
    "web_search_query": {"results": []},
    "docs_load_fetch": {"docs": []},
    "vector_query_search": {"results": []},
    "citations_audit_check": {"reports": []},
    "exports_render_markdown": {"markdown": ""},
}

MAX_IN = 64 * 1024
MAX_OUT = 256 * 1024
_CACHE: Dict[Tuple[str, str], Tuple[float, Dict[str, Any]]] = {}


def _run_tool(tool: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    if tool in STUB_OUTPUTS:
        result = STUB_OUTPUTS[tool]
        validate_tool_output(tool, result)
        return result
    return {"tool": tool, "payload": payload}


def invoke_tool(tool: str, payload: Dict[str, Any], *, timeout: float = 30.0) -> Dict[str, Any]:
    encoded = json.dumps(payload).encode("utf-8")
    if len(encoded) > MAX_IN:
        raise HTTPException(
            status_code=400,
            detail={"error": "INVALID_INPUT", "message": "payload too large", "details": {"size": len(encoded)}},
        )

    key = (tool, json.dumps(payload, sort_keys=True))
    now_ts = time()
    cached = _CACHE.get(key)
    if cached and now_ts - cached[0] < 600:
        return cached[1]

    with ThreadPoolExecutor(max_workers=1) as ex:
        future = ex.submit(_run_tool, tool, payload)
        try:
            result = future.result(timeout=timeout)
        except TimeoutError:
            raise HTTPException(
                status_code=504,
                detail={"error": "TIMEOUT", "message": f"Tool {tool} timed out"},
            )
        except ValidationError as exc:  # pragma: no cover - _run_tool validation
            raise HTTPException(
                status_code=400,
                detail={"error": "schema_validation_failed", "message": exc.message},
            )

    out_encoded = json.dumps(result).encode("utf-8")
    if len(out_encoded) > MAX_OUT:
        raise HTTPException(
            status_code=500,
            detail={"error": "INTERNAL", "message": "output too large", "details": {"size": len(out_encoded)}},
        )

    response = {"ok": True, "data": result}
    _CACHE[key] = (now_ts, response)
    return response
