"""Fragment writes: upserts, batch writes, state changes, deletes, plasticity.

Every write goes through ``_writing`` (one locked transaction). Writes that
can change which fragments are retrievable are *structural* and bump
``write_version`` (invalidating ``retrievable_ids`` caches); access counts,
importance and embeddings are not.

Listener events ("unretrievable", "delete") are emitted only after the
transaction commits, so listeners never observe a rolled-back change.
"""
from __future__ import annotations

import json
import logging
import sqlite3
from collections import Counter
from datetime import datetime, timezone
from typing import Iterable

from mark.security import ContentHasher
from mark.store._sqlite.codec import (
    _FTS_TABLE,
    _RETRIEVABLE_VALUES,
    _SQL_CHUNK,
    _UPSERT_FRAGMENT_SQL,
    _chunks,
    _dt_to_text,
    _pack_embedding,
)
from mark.types import MemoryFragment, MemoryState

logger = logging.getLogger("mark.store.sqlite")


class FragmentWriteMixin:
    """Insert, update and delete fragment rows (keeping FTS in sync)."""

    def store(self, fragment: MemoryFragment) -> str:
        """Insert or update a fragment row (content encrypted at rest).

        Raises ValueError when the id already belongs to a different agent.
        """
        return self.store_many([fragment])[0]

    def store_many(self, fragments: Iterable[MemoryFragment]) -> list[str]:
        """Insert or update many fragments in one transaction (all or nothing).

        Same semantics as store() per fragment. Raises ValueError (and writes
        nothing) when any id already belongs to a different agent.
        """
        unretrievable: list[str] = []
        ids: list[str] = []
        with self._writing() as conn:
            for fragment in fragments:
                fragment_id, became_unretrievable = self._store_row(conn, fragment)
                ids.append(fragment_id)
                if became_unretrievable:
                    unretrievable.append(fragment_id)
        if unretrievable:
            self._emit("unretrievable", unretrievable)
        return ids

    def existing_ids(self, ids: Iterable[str]) -> set[str]:
        """The subset of *ids* that already have a fragment row (any agent, any state).

        Cheap (primary-key lookups, no decryption): used by idempotent bulk
        writes to skip items a previous, interrupted call already stored.
        """
        wanted = list(dict.fromkeys(ids))
        found: set[str] = set()
        for chunk in _chunks(wanted, _SQL_CHUNK):
            placeholders = ",".join("?" for _ in chunk)
            found.update(str(row[0]) for row in self._fetchall(
                f"SELECT id FROM fragments WHERE id IN ({placeholders})", chunk))
        return found

    def _store_row(self, conn: sqlite3.Connection, fragment: MemoryFragment) -> tuple[str, bool]:
        """Upsert one fragment; returns (id, whether it left a retrievable state)."""
        # Hash is computed over the plaintext before encryption, so reads can
        # detect at-rest tampering (see RowDecodingMixin._decode_fragment).
        fragment = ContentHasher.attach(fragment, self.encryption)
        encrypted_content = self.encryption.encrypt_str(fragment.content)
        params = {
            "id": fragment.id,
            "content": encrypted_content,
            "block_id": fragment.block_id,
            "scope": fragment.scope.value,
            "tier": fragment.tier.value,
            "agent_id": fragment.agent_id,
            "session_id": fragment.session_id,
            "importance": fragment.importance,
            "state": fragment.state.value,
            "confidence": fragment.confidence,
            "ttl_seconds": fragment.ttl_seconds,
            "created_at": fragment.created_at.isoformat(),
            "updated_at": fragment.updated_at.isoformat(),
            "last_accessed_at": _dt_to_text(fragment.last_accessed_at),
            "tags": json.dumps(fragment.tags),
            "source": fragment.source,
            "metadata": json.dumps(fragment.metadata),
            # Embeddings live in embedding_blob; the JSON column is legacy-only.
            "embedding": None,
            "contradiction_ids": json.dumps(fragment.contradiction_ids),
            "embedding_blob": _pack_embedding(fragment.embedding),
        }
        existing = conn.execute(
            "SELECT rowid, agent_id, state FROM fragments WHERE id = ?", (fragment.id,)
        ).fetchone()
        if existing is not None and str(existing["agent_id"]) != fragment.agent_id:
            raise ValueError(
                f"Fragment id {fragment.id!r} already belongs to another agent; "
                "refusing to overwrite it."
            )
        cursor = conn.execute(_UPSERT_FRAGMENT_SQL, params)
        self._note_changed((fragment.id,))  # lets id caches apply a delta (scope_delta)
        # ON CONFLICT DO UPDATE keeps the rowid; lastrowid is only set on insert.
        rowid = int(existing["rowid"]) if existing is not None else int(cursor.lastrowid or 0)
        became_unretrievable = (
            existing is not None
            and str(existing["state"]) in _RETRIEVABLE_VALUES
            and fragment.state.value not in _RETRIEVABLE_VALUES
        )
        if self._fts_enabled:
            self._fts_upsert(conn, rowid, fragment.id, fragment.agent_id, fragment.content, fragment.tags)
        elif self._encrypting:  # we hold the plaintext now: sketch it instead of decrypting it later
            self._sketch_written(rowid, encrypted_content, fragment.content)
        return fragment.id, became_unretrievable

    def update_state(self, fragment_id: str, new_state: MemoryState) -> None:
        """Update a fragment's lifecycle state."""
        with self._writing() as conn:
            self._note_changed((fragment_id,))
            cursor = conn.execute(
                "UPDATE fragments SET state = ?, updated_at = ? WHERE id = ?",
                (new_state.value, datetime.now(timezone.utc).isoformat(), fragment_id),
            )
            changed = cursor.rowcount > 0
        if changed and new_state.value not in _RETRIEVABLE_VALUES:
            self._emit("unretrievable", [fragment_id])

    def update_embedding(self, fragment_id: str, embedding: list[float]) -> None:
        """Persist a fragment's embedding (float32 blob); a vector snapshot stops serving the old one."""
        with self._writing(structural=False) as conn:
            # No RETURNING: it needs SQLite 3.35, older than some supported Pythons ship.
            row = conn.execute("SELECT agent_id FROM fragments WHERE id = ?", (fragment_id,)).fetchone()
            conn.execute(
                "UPDATE fragments SET embedding_blob = ?, embedding = NULL WHERE id = ?",
                (_pack_embedding(embedding), fragment_id),
            )
            if row is not None:
                self._mark_vector_dirty(conn, str(row[0]), fragment_id)

    def delete(self, fragment_id: str) -> bool:
        """Delete by id; returns True when a row was removed."""
        with self._writing() as conn:
            self._note_changed((fragment_id,))
            row = conn.execute("SELECT rowid FROM fragments WHERE id = ?", (fragment_id,)).fetchone()
            if row is None:
                return False
            if self._fts_enabled:
                conn.execute(f"DELETE FROM {_FTS_TABLE} WHERE rowid = ?", (int(row[0]),))
            conn.execute("DELETE FROM fragments WHERE id = ?", (fragment_id,))
        self._emit("delete", [fragment_id])
        return True

    def clear(self, agent_id: str | None = None) -> None:
        """Delete fragments, optionally only for one agent."""
        where, params = ("", ()) if agent_id is None else (" WHERE agent_id = ?", (agent_id,))
        with self._writing() as conn:
            ids = [str(row[0]) for row in conn.execute(f"SELECT id FROM fragments{where}", params)]
            if self._fts_enabled:
                if agent_id is None:
                    conn.execute(f"DELETE FROM {_FTS_TABLE}")
                else:
                    # FTS rows must go before the fragments they mirror (rowid join).
                    conn.execute(
                        f"DELETE FROM {_FTS_TABLE} WHERE rowid IN (SELECT rowid FROM fragments{where})",
                        params,
                    )
            conn.execute(f"DELETE FROM fragments{where}", params)
        self._emit("delete", ids)

    def _persist_tampered(self, fragment_ids: list[str]) -> None:
        """Persist CONTRADICTED for fragments whose content hash check failed."""
        now = datetime.now(timezone.utc).isoformat()
        changed: list[str] = []
        with self._writing() as conn:
            self._note_changed(fragment_ids)
            for fragment_id in fragment_ids:
                cursor = conn.execute(
                    "UPDATE fragments SET state = ?, updated_at = ? WHERE id = ? AND state != ?",
                    (MemoryState.CONTRADICTED.value, now, fragment_id, MemoryState.CONTRADICTED.value),
                )
                if cursor.rowcount > 0:
                    changed.append(fragment_id)
        if changed:
            logger.warning("Content hash mismatch; marked CONTRADICTED: %s", ", ".join(changed))
            self._emit("unretrievable", changed)

    # ── plasticity (non-structural: never invalidates retrieval caches) ───────

    def update_importance(self, fragment_id: str, importance: float) -> None:
        """Update the importance of a fragment, clamped to [0, 1]."""
        self.update_importance_many({fragment_id: importance})

    def update_importance_many(self, updates: dict[str, float]) -> None:
        """Update many importances (each clamped to [0, 1]) in one transaction."""
        if not updates:
            return
        now = datetime.now(timezone.utc).isoformat()
        with self._writing(structural=False) as conn:
            conn.executemany(
                "UPDATE fragments SET importance = ?, updated_at = ? WHERE id = ?",
                [
                    (max(0.0, min(1.0, float(importance))), now, fragment_id)
                    for fragment_id, importance in updates.items()
                ],
            )

    def touch(self, fragment_id: str) -> None:
        """Record an access: update last_accessed_at and increment access_count in metadata."""
        self.touch_many([fragment_id])

    def touch_many(self, ids: Iterable[str]) -> None:
        """Record accesses for many fragments in one transaction (same semantics as touch)."""
        # Duplicate ids count as repeated accesses.
        counts = Counter(ids)
        if not counts:
            return
        now = datetime.now(timezone.utc).isoformat()
        with self._writing(structural=False) as conn:
            updates: list[tuple[str, str, str, str]] = []
            # Read-modify-write of metadata JSON happens under the write lock,
            # so concurrent touches cannot lose increments.
            for chunk in _chunks(list(counts), _SQL_CHUNK):
                placeholders = ",".join("?" for _ in chunk)
                rows = conn.execute(
                    f"SELECT id, metadata FROM fragments WHERE id IN ({placeholders})", chunk
                ).fetchall()
                for row in rows:
                    fragment_id = str(row[0])
                    meta = json.loads(str(row[1]))
                    meta["access_count"] = int(meta.get("access_count", 0)) + counts[fragment_id]
                    updates.append((now, json.dumps(meta), now, fragment_id))
            conn.executemany(
                "UPDATE fragments SET last_accessed_at = ?, metadata = ?, updated_at = ? WHERE id = ?",
                updates,
            )
