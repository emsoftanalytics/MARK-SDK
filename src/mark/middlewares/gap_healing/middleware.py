"""Gap-healing middleware for retrieval."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mark.middlewares.base import BaseMiddleware, MiddlewareContext


@dataclass
class GapHealingMiddleware(BaseMiddleware):
    """Opt in to MARK's explicit local gap-healing path for retrieval."""

    heal_gaps: bool = True
    escalate_on_gap: bool = True
    web_skill: Any = None
    name: str = "gap_healing"

    def before(self, context: MiddlewareContext) -> None:
        if context.operation != "retrieve":
            return
        context.payload["heal_gaps"] = self.heal_gaps
        context.payload["escalate_on_gap"] = self.escalate_on_gap
        if self.web_skill is not None:
            context.payload.setdefault("web_skill", self.web_skill)


__all__ = ["GapHealingMiddleware"]
