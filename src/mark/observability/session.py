# SPDX-License-Identifier: MIT
# Copyright (c) 2025 MARK Contributors
"""Session-scoped view over locally traced events."""
from __future__ import annotations

from mark.observability.events import RuntimeEvent
from mark.observability.trace import LocalTracer


class SessionTrace:
    """
    Session-scoped view of events from a LocalTracer.

    Filters events by ``session_id`` stored in the event payload.
    Provides a focused trace for one agent session without copying data.

    Usage::

        tracer = LocalTracer()
        session = SessionTrace(tracer, session_id="ep-01")
        session.events()   # only events whose payload["session_id"] == "ep-01"
    """

    def __init__(self, tracer: LocalTracer, session_id: str) -> None:
        self._tracer     = tracer
        self._session_id = session_id

    @property
    def session_id(self) -> str:
        """Return the session id."""
        return self._session_id

    def events(self, event_type: str | None = None) -> list[RuntimeEvent]:
        """Return recorded events."""
        result = [
            e for e in self._tracer.events()
            if e.payload.get("session_id") == self._session_id
        ]
        if event_type is not None:
            result = [e for e in result if e.type == event_type]
        return result

    def emit(self, event_type: str, **payload) -> RuntimeEvent:
        """Record an event."""
        return self._tracer.emit(event_type, session_id=self._session_id, **payload)

    def summary(self) -> dict:
        """Return a compact human-readable description."""
        evts    = self.events()
        by_type: dict[str, int] = {}
        for e in evts:
            by_type[e.type] = by_type.get(e.type, 0) + 1
        return {
            "session_id": self._session_id,
            "total":      len(evts),
            "by_type":    by_type,
        }
