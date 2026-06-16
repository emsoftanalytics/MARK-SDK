"""Recall middleware for default retrieval options."""
from __future__ import annotations

from dataclasses import dataclass

from mark.middlewares.base import BaseMiddleware, MiddlewareContext


@dataclass
class RecallMiddleware(BaseMiddleware):
    """Set default retrieval behavior for downstream memory calls."""

    compress: bool | None = None
    expand: bool | None = None
    heal_gaps: bool | None = None
    escalate_on_gap: bool | None = None
    name: str = "recall"

    def before(self, context: MiddlewareContext) -> None:
        if context.operation != "retrieve":
            return
        defaults = {
            "compress": self.compress,
            "expand": self.expand,
            "heal_gaps": self.heal_gaps,
            "escalate_on_gap": self.escalate_on_gap,
        }
        for key, value in defaults.items():
            if value is not None:
                context.payload[key] = value


__all__ = ["RecallMiddleware"]
