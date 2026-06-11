"""Backward-compatible simple memory record model."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class MemoryState(str, Enum):
    """Lifecycle states for simple records (legacy surface)."""
    RAW = "raw"
    VERIFIED = "verified"
    QUARANTINED = "quarantined"
    CONTRADICTED = "contradicted"


@dataclass(frozen=True)
class MemoryRecord:
    """Backward-compatible simple memory record."""
    id: str
    block_id: str
    content: str
    importance: float = 0.5
    confidence: float = 0.5
    state: MemoryState = MemoryState.RAW
    metadata: dict[str, Any] = field(default_factory=dict)
    source: str = "local"
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_accessed_at: datetime | None = None
    feedback_score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dictionary."""
        return {
            "id": self.id,
            "block_id": self.block_id,
            "content": self.content,
            "importance": self.importance,
            "confidence": self.confidence,
            "state": self.state.value,
            "metadata": self.metadata,
            "source": self.source,
            "created_at": self.created_at.isoformat(),
            "last_accessed_at": self.last_accessed_at.isoformat() if self.last_accessed_at else None,
            "feedback_score": self.feedback_score,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MemoryRecord":
        """Construct an instance from a plain dictionary."""
        return cls(
            id=data["id"],
            block_id=data["block_id"],
            content=data["content"],
            importance=float(data.get("importance", 0.5)),
            confidence=float(data.get("confidence", 0.5)),
            state=MemoryState(data.get("state", MemoryState.RAW.value)),
            metadata=dict(data.get("metadata") or {}),
            source=str(data.get("source", "local")),
            created_at=_parse_datetime(data.get("created_at")),
            last_accessed_at=_parse_datetime(data.get("last_accessed_at")) if data.get("last_accessed_at") else None,
            feedback_score=float(data.get("feedback_score", 0.0)),
        )


def _parse_datetime(value: str | None) -> datetime:
    if not value:
        return datetime.now(timezone.utc)
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed
