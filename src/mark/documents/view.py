# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 MARK Contributors
"""DocumentsView: ``memory.documents`` — list, get, chunks, count and forget documents.

Documents are addressed by id or by a :class:`DocumentRecord`. Forgetting
a document deletes its chunk memories through the memory's own ``forget``
(so whatever cleanup that performs, such as index removal, applies) and
then the parent record.
"""
from __future__ import annotations

from typing import Any

from mark.documents.ingest import _record
from mark.documents.records import DocumentRecord

DocumentRef = str | DocumentRecord


class DocumentsView:
    """Documents stored by one agent memory."""

    def __init__(self, memory: Any) -> None:
        self._memory = memory
        self._store = memory._store
        self._agent = memory.agent_id

    def list(self, *, limit: int = 100, offset: int = 0) -> list[DocumentRecord]:
        """Documents, newest first (without their full text)."""
        return [_record(row) for row in self._store.document_list(self._agent, limit=limit, offset=offset)]

    def count(self) -> int:
        """How many documents this memory holds."""
        return self._store.document_count(self._agent)

    def get(self, document: DocumentRef) -> DocumentRecord | None:
        """One document with its full text, or None."""
        row = self._store.document_fetch(self._agent, _id(document))
        return None if row is None else _record(row)

    def chunks(self, document: DocumentRef) -> list[Any]:
        """The document's chunk memories in document order (deleted ones skipped)."""
        record = document if isinstance(document, DocumentRecord) else self.get(document)
        if record is None:
            return []
        found = self._store.get_many(list(record.chunk_ids))
        return [found[cid] for cid in record.chunk_ids if found.get(cid) is not None]

    def forget(self, document: DocumentRef) -> int:
        """Delete the document and its chunks; returns how many chunk memories were deleted."""
        record = self.get(document)
        if record is None:
            return 0
        deleted = sum(1 for cid in record.chunk_ids if self._memory.forget(cid))
        self._store.document_delete(self._agent, record.id)
        return deleted


def _id(document: DocumentRef) -> str:
    return document.id if isinstance(document, DocumentRecord) else str(document)
