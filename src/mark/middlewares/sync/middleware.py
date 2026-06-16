"""Sync middleware for explicit cloud-bound delta preparation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mark.middlewares.base import BaseMiddleware, MiddlewareContext


@dataclass
class SyncMiddleware(BaseMiddleware):
    """Prepare redacted sync deltas after write-like operations."""

    options: Any = None
    cloud: Any = None
    upload: bool = False
    name: str = "sync"
    last_delta: Any = None
    last_stats: Any = None

    def after(self, context: MiddlewareContext) -> None:
        if context.operation not in {"store", "observe"}:
            return
        from mark.middlewares.sync import CloudSync

        sync = CloudSync()
        if self.upload and self.cloud is not None:
            self.last_stats = sync.sync(context.runtime, self.cloud, options=self.options)
            context.metadata["sync_stats"] = self.last_stats
            return
        self.last_delta = sync.prepare_delta(context.runtime, options=self.options)
        self.last_stats = self.last_delta.stats
        context.metadata["sync_delta"] = self.last_delta


__all__ = ["SyncMiddleware"]
