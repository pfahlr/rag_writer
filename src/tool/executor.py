from __future__ import annotations

import json
import os
import subprocess
import sys
from importlib import import_module
from typing import Any, Dict, List

from .toolpack_models import ToolPack


def _run_subprocess(cmd: List[str], payload: Dict[str, Any], env: Dict[str, str], timeout: float | None) -> Dict[str, Any]:
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
            cmd = entry
            return _run_subprocess(cmd, payload, env, timeout)
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
    else:  # cli
        if isinstance(tp.entry, list):
            cmd = tp.entry
        else:
            cmd = [tp.entry]
        return _run_subprocess(cmd, payload, env, timeout)
