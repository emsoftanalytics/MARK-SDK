"""Governance middleware for explicit write-time policy gates."""
from __future__ import annotations

from dataclasses import dataclass, field

from mark.middlewares.governance.pipeline import ConsolidationGate
from mark.middlewares.base import BaseMiddleware, MiddlewareContext
from mark.types import MemoryState


@dataclass
class GovernanceMiddleware(BaseMiddleware):
    """Apply an explicit local governance gate before memory writes."""

    gate: ConsolidationGate = field(default_factory=ConsolidationGate)
    quarantine_on_reject: bool = True
    name: str = "governance"

    def before(self, context: MiddlewareContext) -> None:
        if context.operation not in {"store", "observe"}:
            return
        key = "text" if context.operation == "observe" else "content"
        content = str(context.payload.get(key) or "")
        confidence = float(context.payload.get("confidence", 1.0))
        result = self.gate.check(content, confidence=confidence)
        context.payload[key] = result.content

        metadata = dict(context.payload.get("metadata") or {})
        metadata["mark_governance"] = {
            "middleware": self.name,
            "passed": result.passed,
            "reason": result.reason,
        }
        context.payload["metadata"] = metadata

        if not result.passed:
            tags = list(context.payload.get("tags") or [])
            context.payload["tags"] = [*tags, "mark:governance", "mark:quarantined"]
            if self.quarantine_on_reject and context.operation == "store":
                context.payload["state"] = MemoryState.QUARANTINED
                context.payload["importance"] = 0.0
                context.payload["confidence"] = 0.0


__all__ = ["GovernanceMiddleware"]
