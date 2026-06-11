# SPDX-License-Identifier: MIT
# Copyright (c) 2025 MARK Contributors
"""Local JSONL tracer for runtime events."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mark.observability.events import RuntimeEvent


class LocalTracer:
    """
    Local runtime event tracer.

    Appends one JSON line per event to an optional JSONL file.
    When no path is provided, events are kept in-process only.

    Cloud Observatory ingests these events for dashboards, graph replay,
    and performance metrics. Local emits only; cloud owns retention and
    multi-agent aggregation.

    Usage::

        tracer = LocalTracer(path=".mark/events.jsonl")
        tracer.emit("memory.retrieve", query="Elena scarf", result_count=3)
    """

    def __init__(self, path: str | Path | None = None) -> None:
        self._path   = Path(path) if path else None
        self._events: list[RuntimeEvent] = []

    def emit(self, event_type: str, **payload: Any) -> RuntimeEvent:
        """Record an event."""
        event = RuntimeEvent(type=event_type, payload=payload)
        self._events.append(event)
        if self._path is not None:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._path.open("a", encoding="utf-8") as fh:
                fh.write(
                    json.dumps({
                        "type":       event.type,
                        "payload":    event.payload,
                        "created_at": event.created_at.isoformat(),
                    })
                    + "\n"
                )
        return event

    def events(self) -> list[RuntimeEvent]:
        """Return recorded events."""
        return list(self._events)

    def clear(self) -> None:
        """Delete the trace file and reset state."""
        self._events.clear()

    @property
    def path(self) -> Path | None:
        """Return the backing file path."""
        return self._path
