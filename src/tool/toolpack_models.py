from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class ToolSchema(BaseModel):
    input: Dict[str, Any] = Field(default_factory=dict)
    output: Dict[str, Any] = Field(default_factory=dict)


class ToolLimits(BaseModel):
    input: Optional[int] = None
    output: Optional[int] = None


class ToolPack(BaseModel):
    id: str
    kind: Literal["python", "cli"]
    entry: str | List[str]
    schema: ToolSchema
    timeoutMs: Optional[int] = None
    limits: ToolLimits = Field(default_factory=ToolLimits)
    env: List[str] = Field(default_factory=list)
    deterministic: bool = False
