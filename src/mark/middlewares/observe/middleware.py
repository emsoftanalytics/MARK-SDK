"""Observe/writeback middleware."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from mark.middlewares.base import BaseMiddleware, MiddlewareContext


@dataclass
class ObserveMiddleware(BaseMiddleware):
    """Apply observation metadata and enable automatic structuring."""

    source: str | None = None
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    auto_structure: bool = True
    name: str = "observe"

    def before(self, context: MiddlewareContext) -> None:
        if context.operation not in {"observe", "store"}:
            return
        if self.source is not None and context.payload.get("source") is None:
            context.payload["source"] = self.source
        if self.tags:
            incoming = list(context.payload.get("tags") or [])
            context.payload["tags"] = [*incoming, *self.tags]
        if self.metadata:
            incoming_meta = dict(context.payload.get("metadata") or {})
            context.payload["metadata"] = {**self.metadata, **incoming_meta}
        if context.operation == "observe":
            context.payload["auto_structure"] = self.auto_structure


__all__ = ["ObserveMiddleware"]
