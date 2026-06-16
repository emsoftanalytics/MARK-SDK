"""Runtime event records and a bounded in-memory event log."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass(frozen=True)
class RuntimeEvent:
    """One recorded observability event."""
    type: str
    payload: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class RuntimeEventLog:
    """Bounded in-memory log of runtime events."""
    def __init__(self) -> None:
        self._events: list[RuntimeEvent] = []

    def append(self, event_type: str, **payload: Any) -> RuntimeEvent:
        """Append an event."""
        event = RuntimeEvent(type=event_type, payload=payload)
        self._events.append(event)
        return event

    def list(self) -> list[RuntimeEvent]:
        """Return stored entries."""
        return list(self._events)
