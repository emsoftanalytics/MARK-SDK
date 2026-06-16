"""High-level block/record manager and its storage protocol."""
from __future__ import annotations

from typing import Protocol
from uuid import uuid4

from mark.context import ContextBuilder, ContextBundle
from mark.memory.block import MemoryBlock
from mark.memory.record import MemoryRecord
from mark.memory.retrieval import LocalRetriever


class MemoryStore(Protocol):
    """Storage protocol consumed by MemoryManager."""
    def create_block(self, block: MemoryBlock) -> MemoryBlock:
        """Create and persist a new block."""
        ...
    def get_block_by_label(self, label: str) -> MemoryBlock | None:
        """Find a block by label, or None."""
        ...
    def list_blocks(self) -> list[MemoryBlock]:
        """Return all blocks."""
        ...
    def write_record(self, record: MemoryRecord) -> MemoryRecord:
        """Persist a record."""
        ...
    def list_records(self, block_id: str | None = None) -> list[MemoryRecord]:
        """Return records, optionally filtered by block."""
        ...
    def update_record(self, record: MemoryRecord) -> MemoryRecord:
        """Persist changes to an existing record."""
        ...


class MemoryManager:
    """Compatibility facade for the original block/record memory API.

    New SDK code should prefer Mark.local(), SimpleMemory, or MarkRuntime.
    This class remains Apache-2.0-licensed local functionality for older examples and
    tests; it does not contain adaptive ranking, adaptive routing, or training
    logic.
    """

    def __init__(
        self,
        *,
        store: MemoryStore,
        context_builder: ContextBuilder | None = None,
        retriever: LocalRetriever | None = None,
    ) -> None:
        self._store = store
        self._context_builder = context_builder or ContextBuilder()
        self._retriever = retriever or LocalRetriever()

    def block(
        self,
        label: str,
        *,
        kind: str = "custom",
        description: str = "",
    ) -> MemoryBlock:
        """Return (creating if needed) the named block."""
        existing = self._store.get_block_by_label(label)
        if existing:
            return existing.attach(self._store)
        block = MemoryBlock(
            id=f"blk_{uuid4().hex}",
            label=label,
            kind=kind,
            description=description,
            _store=self._store,
        )
        return self._store.create_block(block).attach(self._store)

    def list_blocks(self) -> list[MemoryBlock]:
        """Return all blocks."""
        return [block.attach(self._store) for block in self._store.list_blocks()]

    def retrieve(
        self,
        query: str,
        *,
        blocks: list[str] | None = None,
        top_k: int = 5,
        max_chars: int = 6000,
    ) -> ContextBundle:
        """Keyword-retrieve scored records across all blocks."""
        records = self._records_for_blocks(blocks)
        scored = self._retriever.retrieve(query, records, top_k=top_k)
        return self._context_builder.build(query=query, scored_records=scored, max_chars=max_chars)

    def feedback(self, record_id: str, score: float) -> MemoryRecord:
        """Adjust a record's importance from a feedback score."""
        score = max(-1.0, min(1.0, float(score)))
        for record in self._store.list_records():
            if record.id == record_id:
                updated = MemoryRecord(
                    id=record.id,
                    block_id=record.block_id,
                    content=record.content,
                    importance=record.importance,
                    confidence=record.confidence,
                    state=record.state,
                    metadata=record.metadata,
                    source=record.source,
                    created_at=record.created_at,
                    last_accessed_at=record.last_accessed_at,
                    feedback_score=score,
                )
                return self._store.update_record(updated)
        raise KeyError(f"Memory record not found: {record_id}")

    def _records_for_blocks(self, labels: list[str] | None) -> list[MemoryRecord]:
        if not labels:
            return self._store.list_records()
        ids = {block.id for block in self._store.list_blocks() if block.label in labels}
        return [record for record in self._store.list_records() if record.block_id in ids]
