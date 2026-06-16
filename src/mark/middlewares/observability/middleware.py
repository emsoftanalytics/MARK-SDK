"""Observability middleware for runtime tracing."""
from __future__ import annotations

from dataclasses import dataclass

from mark.middlewares.base import BaseMiddleware, MiddlewareContext


@dataclass
class ObservabilityMiddleware(BaseMiddleware):
    """Emit local tracer events around middleware operations."""

    event_prefix: str = "middleware"
    name: str = "observability"

    def before(self, context: MiddlewareContext) -> None:
        tracer = context.runtime.tracer() if hasattr(context.runtime, "tracer") else None
        if tracer is not None:
            tracer.emit(
                f"{self.event_prefix}.{context.operation}.start",
                agent_id=context.agent_id,
                middleware=True,
            )

    def after(self, context: MiddlewareContext) -> None:
        tracer = context.runtime.tracer() if hasattr(context.runtime, "tracer") else None
        if tracer is not None:
            tracer.emit(
                f"{self.event_prefix}.{context.operation}.finish",
                agent_id=context.agent_id,
                middleware=True,
                stopped=context.stopped,
            )


__all__ = ["ObservabilityMiddleware"]
