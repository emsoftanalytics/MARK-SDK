# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 MARK Contributors
"""DocumentRecord: what ``memory.store_document`` returns and ``memory.documents`` lists.

A record describes one stored document: its id and title, where it came
from, and the ids of the chunk memories it was split into (in document
order). ``text`` is the full original text, loaded only by
``memory.documents.get``; listings leave it empty to stay cheap.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class DocumentRecord:
    """One stored document and its chunks."""
    id: str
    title: str
    chunk_ids: tuple[str, ...]
    created_at: datetime
    source: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    #: Full text (set by ``documents.get``; None in listings).
    text: str | None = None
    #: Facts written inline while storing (0 when extraction was off or queued).
    facts_extracted: int = 0

    @property
    def chunk_count(self) -> int:
        """Number of chunk memories."""
        return len(self.chunk_ids)

    @property
    def word_count(self) -> int:
        """Words in the original text (recorded when it was stored)."""
        return int(self.metadata.get("word_count", 0))

    def as_text(self) -> str:
        """One line for listings: title, size, source."""
        where = f" from {self.source}" if self.source else ""
        return f"[document] {self.title} ({self.word_count} words, {self.chunk_count} chunks){where} id={self.id}"
