from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from jsonschema import Draft7Validator

SCHEMAS_DIR = Path(__file__).resolve().parents[2] / "schemas" / "tools"

INPUT_VALIDATORS: Dict[str, Draft7Validator] = {}
OUTPUT_VALIDATORS: Dict[str, Draft7Validator] = {}

for path in SCHEMAS_DIR.glob("*.input.schema.json"):
    tool = path.name.split(".")[0]
    with path.open("r", encoding="utf-8") as f:
        schema = json.load(f)
    INPUT_VALIDATORS[tool] = Draft7Validator(schema)

for path in SCHEMAS_DIR.glob("*.output.schema.json"):
    tool = path.name.split(".")[0]
    with path.open("r", encoding="utf-8") as f:
        schema = json.load(f)
    OUTPUT_VALIDATORS[tool] = Draft7Validator(schema)


def validate_tool_output(tool: str, data: Dict[str, Any]) -> None:
    validator = OUTPUT_VALIDATORS.get(tool)
    if validator is None:
        raise KeyError(f"No output schema for tool: {tool}")
    validator.validate(data)
