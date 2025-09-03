from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Dict, Any


def file_checksum(path: Path, algo: str = "sha256", chunk: int = 1024 * 1024) -> str:
    h = hashlib.new(algo)
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def load_manifest(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"version": 1, "entries": []}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_manifest(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

