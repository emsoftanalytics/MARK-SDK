"""Lifecycle middleware for scheduled local maintenance."""
from __future__ import annotations

from dataclasses import dataclass

from mark.middlewares.base import BaseMiddleware, MiddlewareContext


@dataclass
class LifecycleMiddleware(BaseMiddleware):
    """Run local maintenance cycles after write-like operations."""

    every_n_writes: int = 0
    promote_threshold: float = 0.7
    name: str = "lifecycle"
    _write_count: int = 0

    def after(self, context: MiddlewareContext) -> None:
        if self.every_n_writes <= 0:
            return
        if context.operation not in {"store", "observe"}:
            return
        if context.agent_id is None or not hasattr(context.runtime, "run_cycle"):
            return
        self._write_count += 1
        if self._write_count % self.every_n_writes == 0:
            context.metadata["lifecycle"] = context.runtime.run_cycle(
                context.agent_id,
                promote_threshold=self.promote_threshold,
            )


__all__ = ["LifecycleMiddleware"]
