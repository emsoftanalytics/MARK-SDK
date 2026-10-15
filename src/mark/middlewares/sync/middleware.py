"""Sync middleware for explicit cloud-bound delta preparation."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from mark.middlewares.base import BaseMiddleware, MiddlewareContext


@dataclass
class SyncMiddleware(BaseMiddleware):
    """Prepare redacted sync deltas after write-like operations.

    Only the delta since the last successful preparation/upload is built: a
    high-water mark over ``updated_at`` (plus the ids already seen at that
    instant) is kept so each write does not re-package the whole store.
    Upload failures are logged and recorded on ``last_stats``; set
    ``raise_on_error=True`` to propagate them to the caller.
    """

    options: Any = None
    cloud: Any = None
    upload: bool = False
    raise_on_error: bool = False
    name: str = "sync"
    last_delta: Any = None
    last_stats: Any = None
    high_water_mark: datetime | None = None
    _ids_at_high_water: set[str] = field(default_factory=set, repr=False)

    def after(self, context: MiddlewareContext) -> None:
        if context.operation not in {"store", "observe"}:
            return
        if not context.succeeded:
            return
        from mark.middlewares.sync import CloudSync

        sync = CloudSync()
        delta = sync.prepare_delta(
            context.runtime,
            options=self.options,
            since=self.high_water_mark,
            exclude_ids=self._ids_at_high_water,
        )
        self.last_delta = delta
        self.last_stats = delta.stats

        ok = delta.stats.errors == 0
        if self.upload and self.cloud is not None:
            ok = sync.upload(delta, self.cloud, raise_on_error=self.raise_on_error) and ok
            context.metadata["sync_stats"] = delta.stats
        context.metadata["sync_delta"] = delta
        if ok:
            self._advance(delta)

    def reset(self) -> None:
        """Forget the high-water mark so the next delta covers the whole store."""
        self.high_water_mark = None
        self._ids_at_high_water = set()

    def _advance(self, delta: Any) -> None:
        mark = delta.stats.high_water_mark
        if mark is None:
            return
        ids_at_mark = set(delta.high_water_ids)
        if self.high_water_mark is not None and mark == self.high_water_mark:
            self._ids_at_high_water |= ids_at_mark
        elif self.high_water_mark is None or mark > self.high_water_mark:
            self.high_water_mark = mark
            self._ids_at_high_water = ids_at_mark


__all__ = ["SyncMiddleware"]
