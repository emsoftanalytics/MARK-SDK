"""Media and creative continuity middleware."""
from __future__ import annotations

from dataclasses import dataclass, field

from mark.middlewares.base import BaseMiddleware, MiddlewareContext


@dataclass
class MediaContinuityMiddleware(BaseMiddleware):
    """Apply local media/creative-continuity defaults."""

    memory_type: str = "scene"
    tags: list[str] = field(default_factory=list)
    retrieve_with_expansion: bool = True
    name: str = "media_continuity"

    def before(self, context: MiddlewareContext) -> None:
        if context.operation == "observe":
            context.payload.setdefault("memory_type", self.memory_type)
            incoming = list(context.payload.get("tags") or [])
            context.payload["tags"] = [*incoming, "continuity", *self.tags]
            metadata = dict(context.payload.get("metadata") or {})
            metadata.setdefault("mark_continuity", {"memory_type": self.memory_type})
            context.payload["metadata"] = metadata
            context.payload["auto_structure"] = True
            return

        if context.operation == "retrieve" and self.retrieve_with_expansion:
            context.payload["expand"] = True


__all__ = ["MediaContinuityMiddleware"]
