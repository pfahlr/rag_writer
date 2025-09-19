from __future__ import annotations

from typing import Any, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class Meta(BaseModel):
    traceId: UUID = Field(default_factory=uuid4)
    durationMs: int = 0


class ErrorItem(BaseModel):
    code: str
    message: str


class Envelope(BaseModel):
    meta: Meta = Field(default_factory=Meta)
    body: Any | None = None
    error: Optional[ErrorItem] = None
    warnings: list[str] = Field(default_factory=list)
