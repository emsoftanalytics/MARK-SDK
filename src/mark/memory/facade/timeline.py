# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 Somtochukwu Owen Ikeanyi
"""Subject timelines on MarkMemory: ``memory.timeline(subject, attribute)``.

Unlike recall, a timeline is not capped at top-k: it returns every
retrievable record about the subject, oldest first. When a runtime
middleware stack is attached the call is routed through it as operation
``"timeline"`` (observability sees it; write-side middleware ignores it).
"""
from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from mark.memory.facade.core import MemoryCore

if TYPE_CHECKING:
    from mark.memory.timeline import Timeline


class TimelineMixin(MemoryCore):
    """Whole-history reads for one subject."""

    def timeline(
        self,
        subject: str,
        attribute: str | None = None,
        *,
        limit: int | None = None,
        since: datetime | str | None = None,
        until: datetime | str | None = None,
    ) -> "Timeline":
        """Every event about *subject* (optionally one *attribute*), oldest first.

        Merges the subject's fact history with every retrievable memory that
        names the subject as a whole word/identifier (no top-k cap).
        ``limit`` keeps the newest events; ``since``/``until`` bound event
        times (ISO strings or datetimes, inclusive). See
        :mod:`mark.memory.timeline` for ordering and rendering.
        """
        payload = {"subject": subject, "attribute": attribute, "limit": limit, "since": since, "until": until}
        if self._middleware is not None and self._runtime is not None:
            return self._middleware.run(
                "timeline", runtime=self._runtime, agent_id=self.agent_id, payload=payload,
                handler=lambda ctx: self._timeline_impl(**ctx.payload),
            )
        return self._timeline_impl(**payload)

    def _timeline_impl(
        self,
        subject: str,
        attribute: str | None = None,
        limit: int | None = None,
        since: datetime | str | None = None,
        until: datetime | str | None = None,
    ) -> "Timeline":
        # Deferred import keeps `import mark` light (the builder pulls in the
        # query-understanding helpers only when a timeline is requested).
        from mark.memory.timeline_build import build_timeline

        return build_timeline(
            self._store, self.agent_id, subject, attribute, limit=limit, since=since, until=until,
        )
