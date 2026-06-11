"""Keyword retriever over simple records (legacy surface)."""
from __future__ import annotations

from dataclasses import dataclass

from mark.memory.record import MemoryRecord, MemoryState
from mark.memory.scoring import score_record


@dataclass(frozen=True)
class ScoredRecord:
    """Simple record paired with its retrieval score."""
    record: MemoryRecord
    score: float


class LocalRetriever:
    """Keyword/importance retriever over simple records."""
    def retrieve(self, query: str, records: list[MemoryRecord], top_k: int = 5) -> list[ScoredRecord]:
        """Retrieve memory relevant to the query."""
        scored = [
            ScoredRecord(record=record, score=score_record(query, record))
            for record in records
            if record.state not in {MemoryState.QUARANTINED, MemoryState.CONTRADICTED}
        ]
        scored.sort(key=lambda item: (-item.score, item.record.created_at, item.record.id))
        return scored[: max(0, top_k)]
