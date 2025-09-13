from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple, List

import json
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from time import time

import yaml
from fastapi import HTTPException
from jsonschema import ValidationError, validate

from .. import __version__ as SERVER_VERSION
from .schemas import validate_tool_output
from .toolpack_loader import load_toolpacks
from .executor import run_toolpack, _render_template
from .toolpack_models import ToolPack

REGISTRY_PATH = Path("prompts/REGISTRY.yaml")
PACKS_DIR = Path("prompts/packs")
START_TIME = time()


def _load_registry() -> Dict[str, Dict[str, list[int]]]:
    if not REGISTRY_PATH.exists():
        return {}
    data = yaml.safe_load(REGISTRY_PATH.read_text())
    return data or {}


def _load_toolpack_meta(root: Path = Path("tools")) -> List[Dict[str, Any]]:
    meta: List[Dict[str, Any]] = []
    if not root.exists():
        return meta
    for path in root.rglob("*.tool.yaml"):
        data = yaml.safe_load(path.read_text()) or {}
        tp_id = data.get("id")
        if not tp_id:
            continue
        schema = data.get("schema", {})
        meta.append(
            {
                "name": tp_id,
                "version": data.get("version", "0.0.0"),
                "inputSchemaRef": schema.get("input", {}).get("$ref"),
                "outputSchemaRef": schema.get("output", {}).get("$ref"),
                "caps": {
                    "timeoutMs": data.get("timeoutMs"),
                    "maxInputBytes": data.get("limits", {}).get("input"),
                    "maxOutputBytes": data.get("limits", {}).get("output"),
                    "network": data.get("caps", {}).get("network", []),
                },
            }
        )
    return meta


TOOLPACK_META = _load_toolpack_meta()


def discover() -> Dict[str, Any]:
    registry = _load_registry()
    prompts: List[Dict[str, Any]] = []
    for domain, names in registry.items():
        for name, majors in names.items():
            versions = []
            for major in majors:
                spec_path = PACKS_DIR / domain / f"{name}.spec.yaml"
                spec = (
                    yaml.safe_load(spec_path.read_text()) if spec_path.exists() else {}
                )
                versions.append(
                    {
                        "major": major,
                        "description": spec.get("description", ""),
                        "specRef": str(spec_path),
                    }
                )
            prompts.append({"id": f"{domain}/{name}", "versions": versions})

    uptime_sec = int(time() - START_TIME)
    return {
        "server": {"name": "rag-writer", "version": SERVER_VERSION},
        "tools": TOOLPACK_META,
        "prompts": prompts,
        "health": {"status": "ok", "uptimeSec": uptime_sec},
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

TOOLPACKS: Dict[str, ToolPack] = load_toolpacks()


def _run_tool(tool: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    if tool in STUB_OUTPUTS:
        return STUB_OUTPUTS[tool]
    return {"tool": tool, "payload": payload}


def invoke_tool(
    tool: str, payload: Dict[str, Any], *, timeout: float = 30.0
) -> Dict[str, Any]:
    tp = TOOLPACKS.get(tool)
    limit_in = tp.limits.input if tp and tp.limits.input is not None else MAX_IN
    encoded = json.dumps(payload).encode("utf-8")
    if len(encoded) > limit_in:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "INVALID_INPUT",
                "message": "payload too large",
                "details": {"size": len(encoded)},
            },
        )
    do_cache = tp.deterministic if tp else True
    if tp and tp.templating and tp.templating.cacheKey:
        key_payload = _render_template(tp.templating.cacheKey, payload)
    else:
        key_payload = json.dumps(payload, sort_keys=True)
    key = (tool, key_payload)
    now_ts = time()
    if do_cache:
        cached = _CACHE.get(key)
        if cached and now_ts - cached[0] < 600:
            return cached[1]

    if tp:
        try:
            validate(payload, tp.schema.input)
        except ValidationError as exc:
            raise HTTPException(
                status_code=400,
                detail={"error": "schema_validation_failed", "message": exc.message},
            )

    runner = (
        (lambda: run_toolpack(tp, payload))
        if tp
        else (lambda: _run_tool(tool, payload))
    )
    run_timeout = tp.timeoutMs / 1000 if tp and tp.timeoutMs else timeout

    with ThreadPoolExecutor(max_workers=1) as ex:
        future = ex.submit(runner)
        try:
            result = future.result(timeout=run_timeout)
        except TimeoutError:
            raise HTTPException(
                status_code=504,
                detail={"error": "TIMEOUT", "message": f"Tool {tool} timed out"},
            )
        except ValidationError as exc:
            raise HTTPException(
                status_code=400,
                detail={"error": "schema_validation_failed", "message": exc.message},
            )

    if tp:
        try:
            validate(result, tp.schema.output)
        except ValidationError as exc:
            raise HTTPException(
                status_code=400,
                detail={"error": "schema_validation_failed", "message": exc.message},
            )

    try:
        validate_tool_output(tool, result)
    except KeyError:
        pass
    except ValidationError as exc:
        raise HTTPException(
            status_code=400,
            detail={"error": "schema_validation_failed", "message": exc.message},
        )

    out_encoded = json.dumps(result).encode("utf-8")
    limit_out = tp.limits.output if tp and tp.limits.output is not None else MAX_OUT
    if len(out_encoded) > limit_out:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "INTERNAL",
                "message": "output too large",
                "details": {"size": len(out_encoded)},
            },
        )

    response = {"ok": True, "data": result}
    if do_cache:
        _CACHE[key] = (now_ts, response)
    return response
