"""Simple labeled memory-block container over flat records (legacy surface)."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Protocol
from uuid import uuid4

from mark.memory.record import MemoryRecord, MemoryState


class BlockStore(Protocol):
    """Storage protocol for simple block records."""
    def write_record(self, record: MemoryRecord) -> MemoryRecord:
        """Persist a record."""
        ...
    def list_records(self, block_id: str | None = None) -> list[MemoryRecord]:
        """Return records, optionally filtered by block."""
        ...


@dataclass(frozen=True)
class MemoryBlock:
    """Simple labeled block: a governed container of MemoryRecords."""
    id: str
    label: str
    kind: str = "custom"
    description: str = ""
    active: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    _store: BlockStore | None = field(default=None, repr=False, compare=False)

    def write(
        self,
        content: str,
        *,
        importance: float = 0.5,
        confidence: float = 0.5,
        metadata: dict[str, Any] | None = None,
        state: MemoryState = MemoryState.RAW,
        source: str = "local",
    ) -> MemoryRecord:
        """Write content into memory."""
        if self._store is None:
            raise RuntimeError("MemoryBlock is detached from a store.")
        record = MemoryRecord(
            id=f"rec_{uuid4().hex}",
            block_id=self.id,
            content=content,
            importance=_clamp(importance),
            confidence=_clamp(confidence),
            metadata=metadata or {},
            state=state,
            source=source,
        )
        return self._store.write_record(record)

    def records(self) -> list[MemoryRecord]:
        """Return this block's records."""
        if self._store is None:
            raise RuntimeError("MemoryBlock is detached from a store.")
        return self._store.list_records(block_id=self.id)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dictionary."""
        return {
            "id": self.id,
            "label": self.label,
            "kind": self.kind,
            "description": self.description,
            "active": self.active,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], store: BlockStore | None = None) -> "MemoryBlock":
        """Construct an instance from a plain dictionary."""
        return cls(
            id=data["id"],
            label=data["label"],
            kind=data.get("kind", "custom"),
            description=data.get("description", ""),
            active=bool(data.get("active", True)),
            metadata=dict(data.get("metadata") or {}),
            created_at=_parse_datetime(data.get("created_at")),
            updated_at=_parse_datetime(data.get("updated_at")),
            _store=store,
        )

    def attach(self, store: BlockStore) -> "MemoryBlock":
        """Attach to a backing store."""
        return MemoryBlock(
            id=self.id,
            label=self.label,
            kind=self.kind,
            description=self.description,
            active=self.active,
            metadata=self.metadata,
            created_at=self.created_at,
            updated_at=self.updated_at,
            _store=store,
        )


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _parse_datetime(value: str | None) -> datetime:
    if not value:
        return datetime.now(timezone.utc)
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed
