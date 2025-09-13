from __future__ import annotations

import json
import os
import subprocess
import sys
from importlib import import_module
from typing import Any, Dict, List

import requests
from jinja2 import Environment, BaseLoader

from .toolpack_models import ToolPack


_JINJA_ENV = Environment(loader=BaseLoader())


def _render_template(text: str, payload: Dict[str, Any]) -> str:
    return _JINJA_ENV.from_string(text).render(input=payload)


def _render_list(items: List[str], payload: Dict[str, Any]) -> List[str]:
    return [_render_template(i, payload) for i in items]


def _run_subprocess(
    cmd: List[str], payload: Dict[str, Any], env: Dict[str, str], timeout: float | None
) -> Dict[str, Any]:
    proc = subprocess.run(
        cmd,
        input=json.dumps(payload).encode("utf-8"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        timeout=timeout,
        check=True,
    )
    data = json.loads(proc.stdout.decode("utf-8"))
    if isinstance(data, dict) and "data" in data:
        return data["data"]
    return data


def run_toolpack(tp: ToolPack, payload: Dict[str, Any]) -> Dict[str, Any]:
    timeout = tp.timeoutMs / 1000 if tp.timeoutMs else None
    env = {"PATH": os.environ.get("PATH", "")}
    for key in tp.env:
        if key in os.environ:
            env[key] = os.environ[key]
    if tp.kind == "python":
        entry = tp.entry
        if isinstance(entry, list):
            cmd = _render_list(entry, payload)
            return _run_subprocess(cmd, payload, env, timeout)
        entry = _render_template(entry, payload)
        if ":" in entry:
            mod, func = entry.split(":", 1)
            try:
                module = import_module(mod)
                fn = getattr(module, func)
                return fn(**payload)
            except Exception:
                cmd = [sys.executable, "-m", mod]
                return _run_subprocess(cmd, payload, env, timeout)
        else:
            cmd = [sys.executable, entry]
            return _run_subprocess(cmd, payload, env, timeout)
    if tp.kind == "cli":
        if isinstance(tp.entry, list):
            cmd = _render_list(tp.entry, payload)
        else:
            cmd = [_render_template(tp.entry, payload)]
        return _run_subprocess(cmd, payload, env, timeout)
    if tp.kind == "node":
        if isinstance(tp.entry, list):
            cmd = ["node"] + _render_list(tp.entry, payload)
        else:
            cmd = ["node", _render_template(tp.entry, payload)]
        return _run_subprocess(cmd, payload, env, timeout)
    if tp.kind == "php":
        entry = tp.php if tp.php is not None else tp.entry
        php_bin = tp.phpBinary or "php"
        if isinstance(entry, list):
            cmd = [php_bin] + _render_list(entry, payload)
        else:
            cmd = [php_bin, _render_template(entry, payload)]
        return _run_subprocess(cmd, payload, env, timeout)
    # http
    url = _render_template(tp.entry, payload)
    headers = {k: _render_template(v, payload) for k, v in tp.headers.items()}
    resp = requests.post(url, json={"input": payload}, headers=headers, timeout=timeout)
    data = resp.json()
    if isinstance(data, dict) and "data" in data:
        return data["data"]
    return data
