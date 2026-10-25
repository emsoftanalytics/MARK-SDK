# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 Somtochukwu Owen Ikeanyi
"""MemoryDocument: a full source document and the fragments chunked from it."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator

from mark.types._datetime import ensure_utc
from mark.types.memory_enums import MemoryScope
from mark.types.memory_fragment import MemoryFragment


class MemoryDocument(BaseModel):
    """
    A structured knowledge source that is chunked into MemoryFragments.

    The Document retains the full original content and the provenance (source URL,
    file path, etc.) so the system can always trace a Fragment back to where it
    came from. Chunking strategy is the caller's responsibility; the Document
    just records which Fragment ids came from it.

    Typical flow:
        doc = MemoryDocument(title="FastAPI docs", content=raw_text,
                             agent_id="coder", source="https://fastapi.tiangolo.com")
        chunks = chunk_document(doc)
        fragments = [MemoryFragment(content=c, agent_id=doc.agent_id) for c in chunks]
        doc.register_chunks(fragments)
    """
    title:        str
    content:      str
    agent_id:     str
    id:           str              = Field(default_factory=lambda: str(uuid4()))
    source:       Optional[str]    = None
    scope:        MemoryScope      = MemoryScope.AGENT
    fragment_ids: List[str]        = Field(default_factory=list)
    tags:         List[str]        = Field(default_factory=list)
    metadata:     Dict[str, Any]   = Field(default_factory=dict)
    created_at:   datetime         = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("created_at")
    @classmethod
    def _utc_datetimes(cls, v: datetime) -> datetime:
        """Coerce naive datetimes to UTC."""
        return ensure_utc(v)

    @field_validator("title")
    @classmethod
    def title_not_empty(cls, v: str) -> str:
        """Validate that the title is not blank."""
        v = v.strip()
        if not v:
            raise ValueError("title must not be blank")
        return v

    @field_validator("content")
    @classmethod
    def content_not_empty(cls, v: str) -> str:
        """Validate that the content is not blank."""
        v = v.strip()
        if not v:
            raise ValueError("content must not be blank")
        return v

    def register_chunks(self, fragments: List[MemoryFragment]) -> None:
        """Record which fragment ids were produced from this document."""
        for f in fragments:
            if f.id not in self.fragment_ids:
                self.fragment_ids.append(f.id)

    def word_count(self) -> int:
        """Return the number of words in the content."""
        return len(self.content.split())

    def summary(self) -> str:
        """Return a compact human-readable description."""
        snippet = self.content[:100].replace("\n", " ")
        suffix = "…" if len(self.content) > 100 else ""
        return (
            f"[DOC|{self.scope.value}] {self.title!r} "
            f"({self.word_count()} words, {len(self.fragment_ids)} chunks) "
            f"← {snippet}{suffix}"
        )
