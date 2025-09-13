from __future__ import annotations

from pathlib import Path
from typing import Dict, Any

import yaml

from .toolpack_models import ToolPack


def _resolve_refs(obj: Any, base: Path) -> Any:
    if isinstance(obj, dict):
        if "$ref" in obj:
            ref_path = (base / obj["$ref"]).resolve()
            with open(ref_path, "r", encoding="utf-8") as fh:
                data = yaml.safe_load(fh)
            return _resolve_refs(data, ref_path.parent)
        return {k: _resolve_refs(v, base) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_resolve_refs(v, base) for v in obj]
    return obj


def load_toolpacks(root: Path = Path("tools")) -> Dict[str, ToolPack]:
    packs: Dict[str, ToolPack] = {}
    if not root.exists():
        return packs
    for path in root.rglob("*.tool.yaml"):
        data = yaml.safe_load(path.read_text()) or {}
        data = _resolve_refs(data, path.parent)
        tp = ToolPack.model_validate(data)
        packs[tp.id] = tp
    return packs
