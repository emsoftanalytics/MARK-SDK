# SPDX-License-Identifier: MIT
# Copyright (c) 2025 MARK Contributors
"""Replay of JSONL traces recorded by the local tracer."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from mark.observability.events import RuntimeEvent


class EventReplayer:
    """
    Load and replay events from a JSONL events file.

    Each line in the file must be a JSON object with ``type``,
    ``payload``, and ``created_at`` fields (the format written
    by LocalTracer).

    Usage::

        replayer = EventReplayer(".mark/events.jsonl")
        events   = replayer.load()
        replayer.replay(lambda e: print(e.type, e.payload))
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)

    def load(self) -> list[RuntimeEvent]:
        """Load persisted state."""
        if not self._path.exists():
            return []
        events: list[RuntimeEvent] = []
        with self._path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    created_at = datetime.fromisoformat(data["created_at"])
                    if created_at.tzinfo is None:
                        created_at = created_at.replace(tzinfo=timezone.utc)
                    events.append(
                        RuntimeEvent(
                            type       = data["type"],
                            payload    = data.get("payload", {}),
                            created_at = created_at,
                        )
                    )
                except (json.JSONDecodeError, KeyError):
                    continue
        return events

    def replay(self, handler: Callable[[RuntimeEvent], None]) -> int:
        """Replay recorded events through the handler."""
        events = self.load()
        for event in events:
            handler(event)
        return len(events)
