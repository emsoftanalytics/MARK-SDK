# SPDX-License-Identifier: MIT
# Copyright (c) 2025 MARK Contributors
"""Session activity log persisted in SQLite."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mark.store.sqlite import LocalMemoryStore


class SessionActivityLog:
    """Read/write helper for the session_events table.

    Obtain via MarkRuntime.session_log(agent_id, session_id).

    Example::

        log = runtime.session_log("writer-agent", "season-01/ep-03")
        log.log("observe", fragment_id=frag_id, metadata={"text_len": 80})
        events = log.list(event_type="retrieve")
        print(log.summary())
    """

    def __init__(
        self,
        store: "LocalMemoryStore",
        agent_id: str,
        session_id: str | None = None,
    ) -> None:
        self._store      = store
        self._agent_id   = agent_id
        self._session_id = session_id

    def log(
        self,
        event_type: str,
        *,
        fragment_id: str | None = None,
        query: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Record an event. Returns the new event id."""
        return self._store.log_event(
            self._agent_id,
            event_type,
            session_id=self._session_id,
            fragment_id=fragment_id,
            query=query,
            metadata=metadata,
        )

    def list(
        self,
        *,
        event_type: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Return events for this agent/session, newest first."""
        return self._store.list_events(
            self._agent_id,
            session_id=self._session_id,
            event_type=event_type,
            limit=limit,
        )

    def summary(self) -> dict[str, Any]:
        """Return a count summary: total events and counts by event_type."""
        events  = self._store.list_events(self._agent_id, session_id=self._session_id, limit=10_000)
        by_type: dict[str, int] = {}
        for e in events:
            t = str(e.get("event_type", "unknown"))
            by_type[t] = by_type.get(t, 0) + 1
        return {
            "agent_id":   self._agent_id,
            "session_id": self._session_id,
            "total":      len(events),
            "by_type":    by_type,
        }
