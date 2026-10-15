"""Session event log: an append-only record of what happened in a session.

Events (recalls, stores, feedback, ...) are free-form rows keyed by agent and
optionally session/fragment. Logging is a non-structural write.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any

from mark.store._sqlite.codec import _optional_str


class EventMixin:
    """Append and list session events."""

    def log_event(
        self,
        agent_id: str,
        event_type: str,
        *,
        session_id: str | None = None,
        fragment_id: str | None = None,
        query: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Record a session event. Returns the new event id."""
        event_id = str(uuid.uuid4())
        now      = datetime.now(timezone.utc).isoformat()
        with self._writing(structural=False) as conn:
            conn.execute(
                "INSERT INTO session_events VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (event_id, agent_id, session_id, event_type, fragment_id, query,
                 json.dumps(metadata or {}), now),
            )
        return event_id

    def list_events(
        self,
        agent_id: str,
        *,
        session_id: str | None = None,
        event_type: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Return session events newest-first."""
        sql    = "SELECT * FROM session_events WHERE agent_id = ?"
        params: list[object] = [agent_id]
        if session_id is not None:
            sql += " AND session_id = ?"
            params.append(session_id)
        if event_type is not None:
            sql += " AND event_type = ?"
            params.append(event_type)
        sql += " ORDER BY created_at DESC, rowid DESC LIMIT ?"
        params.append(limit)
        rows = self._fetchall(sql, params)
        return [
            {
                "id":          str(row["id"]),
                "agent_id":    str(row["agent_id"]),
                "session_id":  _optional_str(row["session_id"]),
                "event_type":  str(row["event_type"]),
                "fragment_id": _optional_str(row["fragment_id"]),
                "query":       _optional_str(row["query"]),
                "metadata":    json.loads(str(row["metadata"])),
                "created_at":  str(row["created_at"]),
            }
            for row in rows
        ]
