"""Stored documents: the parent records that chunk memories link to.

Uses the existing ``documents`` table (no schema change). Unlike the older
``store_document``/``get_document`` pair in :mod:`.graph`, these methods
encrypt the title, full text and source at rest with the store's provider,
like fragment content, and add listing, lookup by content hash, counting
and deletion. Rows written here carry ``metadata["mark_document"] = 1``.

Document writes are non-structural: the chunk fragments, not the parent
row, decide what retrieval can return.
"""
from __future__ import annotations

import json
import sqlite3
from typing import Any

from mark.store._sqlite.codec import _parse_dt

_MARKER = "mark_document"


class DocumentIndexMixin:
    """Encrypted parent records for chunked documents."""

    def document_put(self, record: dict[str, Any]) -> None:
        """Insert or replace one document record (keys: id, agent_id, title, content, source,
        scope, fragment_ids, tags, metadata, created_at)."""
        metadata = {**record.get("metadata", {}), _MARKER: 1}
        with self._writing(structural=False) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO documents (id, title, content, source, agent_id, scope, fragment_ids,"
                " tags, metadata, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (record["id"], self.encryption.encrypt_str(record["title"]),
                 self.encryption.encrypt_str(record["content"]),
                 None if record.get("source") is None else self.encryption.encrypt_str(record["source"]),
                 record["agent_id"], record.get("scope", "agent"), json.dumps(list(record["fragment_ids"])),
                 json.dumps(list(record.get("tags", []))), json.dumps(metadata), record["created_at"].isoformat()),
            )

    def document_fetch(self, agent_id: str, document_id: str, *, with_content: bool = True) -> dict[str, Any] | None:
        """One document record of *agent_id*, or None."""
        row = self._fetchone(f"SELECT * FROM documents WHERE id = ? AND agent_id = ? AND {_OURS}",
                             (document_id, agent_id))
        return None if row is None else self._document_row(row, with_content)

    def document_find(self, agent_id: str, sha256: str) -> dict[str, Any] | None:
        """The document of *agent_id* whose text has this SHA-256, or None."""
        row = self._fetchone(
            f"SELECT * FROM documents WHERE agent_id = ? AND {_OURS} AND json_extract(metadata, '$.sha256') = ?"
            " ORDER BY created_at LIMIT 1", (agent_id, sha256))
        return None if row is None else self._document_row(row, False)

    def document_list(self, agent_id: str, *, limit: int = 100, offset: int = 0) -> list[dict[str, Any]]:
        """Documents of *agent_id*, newest first, without their full text."""
        rows = self._fetchall(
            f"SELECT * FROM documents WHERE agent_id = ? AND {_OURS} ORDER BY created_at DESC, rowid DESC"
            " LIMIT ? OFFSET ?", (agent_id, int(limit), int(offset)))
        return [self._document_row(row, False) for row in rows]

    def document_count(self, agent_id: str) -> int:
        """How many documents *agent_id* has."""
        row = self._fetchone(f"SELECT COUNT(*) FROM documents WHERE agent_id = ? AND {_OURS}", (agent_id,))
        return int(row[0]) if row else 0

    def document_delete(self, agent_id: str, document_id: str) -> bool:
        """Delete one document record (its chunks are deleted by the caller)."""
        with self._writing(structural=False) as conn:
            cursor = conn.execute(f"DELETE FROM documents WHERE id = ? AND agent_id = ? AND {_OURS}",
                                  (document_id, agent_id))
            return cursor.rowcount > 0

    def _document_row(self, row: sqlite3.Row, with_content: bool) -> dict[str, Any]:
        source = row["source"]
        return {
            "id": str(row["id"]),
            "agent_id": str(row["agent_id"]),
            "title": self.encryption.decrypt_str(row["title"]),
            "content": self.encryption.decrypt_str(row["content"]) if with_content else None,
            "source": None if source is None else self.encryption.decrypt_str(source),
            "fragment_ids": json.loads(str(row["fragment_ids"] or "[]")),
            "tags": json.loads(str(row["tags"] or "[]")),
            "metadata": {k: v for k, v in json.loads(str(row["metadata"] or "{}")).items() if k != _MARKER},
            "created_at": _parse_dt(row["created_at"]),
        }


# Only rows written by this mixin (older graph documents are plaintext and untouched).
_OURS = f"json_extract(metadata, '$.{_MARKER}') = 1"
