"""Fragment reads: lookups by id, filtered listings, sessions and counts.

Decoding verifies each fragment's content hash; a fragment found to be
tampered is returned as CONTRADICTED and that state is persisted (a write)
so it stops surfacing in retrieval.
"""
from __future__ import annotations

from typing import Iterable

from mark.store._sqlite.codec import _RETRIEVABLE_STATES, _SQL_CHUNK, _chunks, _decode_embedding, _escape_like
from mark.types import BlockStatus, MemoryFragment, MemoryScope, MemoryState, MemoryTier


class FragmentReadMixin:
    """Read-side fragment API plus the shared WHERE-clause builder."""

    def get(self, fragment_id: str) -> MemoryFragment | None:
        """Fetch a fragment by id, or None.

        A fragment whose content no longer matches its stored hash is returned
        as CONTRADICTED, and that state is persisted so it stops surfacing.
        """
        row = self._fetchone("SELECT * FROM fragments WHERE id = ?", (fragment_id,))
        if row is None:
            return None
        fragment, tampered = self._decode_fragment(dict(row))
        if tampered:
            self._persist_tampered([fragment.id])
        return fragment

    def get_many(self, ids: Iterable[str]) -> dict[str, MemoryFragment]:
        """Fetch many fragments by id in as few queries as possible (chunks of 900)."""
        unique = list(dict.fromkeys(ids))
        result: dict[str, MemoryFragment] = {}
        tampered_ids: list[str] = []
        for chunk in _chunks(unique, _SQL_CHUNK):
            placeholders = ",".join("?" for _ in chunk)
            rows = self._fetchall(f"SELECT * FROM fragments WHERE id IN ({placeholders})", chunk)
            for row in rows:
                fragment, tampered = self._decode_fragment(dict(row))
                result[fragment.id] = fragment
                if tampered:
                    tampered_ids.append(fragment.id)
        # One write transaction for all tampered rows found by this read.
        if tampered_ids:
            self._persist_tampered(tampered_ids)
        return result

    def list_by_agent(
        self,
        agent_id: str,
        *,
        scope: MemoryScope | None = None,
        states: Iterable[MemoryState] | None = None,
        tier: MemoryTier | None = None,
        session_id: str | None = None,
        session_prefix: str | None = None,
        tags: list[str] | None = None,
        block_id: str | None = None,
        block_ids: list[str] | None = None,
        limit: int | None = 200,
    ) -> list[MemoryFragment]:
        """Query fragments for agent_id with optional session, tag, tier, scope, and block filters.

        session_id      — exact match on session_id column
        session_prefix  — prefix match: session_id LIKE '<prefix>%' (wildcards escaped)
        tags            — all listed tags must be present (AND semantics)
        tier            — exact match on tier column
        scope           — exact match on scope column
        block_id        — restrict to fragments belonging to one block
        block_ids       — restrict to fragments belonging to any listed block
        limit           — maximum rows; None returns every match

        Fragments belonging to a QUARANTINED block are always excluded, so a
        corrupted block is isolated without affecting the rest of the agent's
        memory.
        """
        where, params = self._fragment_filters(
            agent_id,
            scope=scope,
            states=states,
            tier=tier,
            session_id=session_id,
            session_prefix=session_prefix,
            tags=tags,
            block_id=block_id,
            block_ids=block_ids,
        )
        sql = f"SELECT * FROM fragments WHERE {where} ORDER BY importance DESC, created_at DESC"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)
        rows = self._fetchall(sql, params)
        return [self._row_to_fragment(dict(row)) for row in rows]

    def embeddings_for_agent(self, agent_id: str) -> list[tuple[str, list[float] | None]]:
        """Return (id, embedding-or-None) for every retrievable-state fragment of an agent.

        Cheap projection: only the id and embedding columns are read.
        """
        state_placeholders = ",".join("?" for _ in _RETRIEVABLE_STATES)
        rows = self._fetchall(
            f"SELECT id, embedding_blob, embedding FROM fragments"
            f" WHERE agent_id = ? AND state IN ({state_placeholders}) ORDER BY rowid",
            [agent_id, *(state.value for state in _RETRIEVABLE_STATES)],
        )
        return [(str(row[0]), _decode_embedding(row[1], row[2])) for row in rows]

    def embedding_rows(self, ids: Iterable[str]) -> list[tuple]:
        """Raw (id, float32 blob, legacy JSON, rowid) rows for *ids*, like ``embedding_pages`` yields.

        Primary-key lookups in chunks: a few thousand rows written since a
        vector snapshot load without streaming the whole agent.
        """
        rows: list[tuple] = []
        for chunk in _chunks(list(ids), _SQL_CHUNK):
            with self._reading() as conn:
                cursor = conn.cursor()
                cursor.row_factory = None  # plain tuples
                rows.extend(cursor.execute(
                    "SELECT id, embedding_blob, embedding, rowid FROM fragments"
                    f" WHERE id IN ({','.join('?' for _ in chunk)})", chunk).fetchall())
        return rows

    def list_sessions(self, agent_id: str) -> list[str]:
        """Return distinct session_ids for agent_id, ordered by most recent activity."""
        rows = self._fetchall(
            """
            SELECT session_id
            FROM fragments
            WHERE agent_id = ? AND session_id IS NOT NULL
            GROUP BY session_id
            ORDER BY MAX(created_at) DESC
            """,
            (agent_id,),
        )
        return [str(row[0]) for row in rows]

    def list_all(self) -> list[MemoryFragment]:
        """Return every stored fragment."""
        rows = self._fetchall("SELECT * FROM fragments")
        return [self._row_to_fragment(dict(row)) for row in rows]

    def recent_duplicate(self, agent_id: str, text: str, since: str, *, limit: int = 64) -> str | None:
        """Id of a memory of *agent_id* written at/after *since* (ISO time) whose content is *text*.

        Used to retry a write whose outcome is unknown (the server died
        mid-request) without storing it twice. Redaction placeholders in the
        stored text (``[REDACTED]``, ``[EMAIL]``, ...) match any span, since
        the write path may have replaced secrets. Only the newest *limit*
        rows are decrypted.
        """
        import re

        from mark.security.redaction import PLACEHOLDER_RE
        from mark.store._sqlite.codec import _as_bytes

        wanted = " ".join(text.split())
        rows = self._fetchall(
            "SELECT id, content FROM fragments WHERE agent_id = ? AND updated_at >= ? ORDER BY rowid DESC LIMIT ?",
            (agent_id, since, limit),
        )
        for fragment_id, content in rows:
            try:
                stored = " ".join(self.encryption.decrypt_str(_as_bytes(content)).split())
            except Exception:  # noqa: BLE001 - unreadable rows are never a match
                continue
            if stored == wanted:
                return str(fragment_id)
            parts = PLACEHOLDER_RE.split(stored)
            if len(parts) > 1 and re.fullmatch(".+?".join(map(re.escape, parts)), wanted, re.S):
                return str(fragment_id)
        return None

    def count(self, agent_id: str) -> int:
        """Return the number of fragments stored for an agent."""
        row = self._fetchone("SELECT COUNT(*) FROM fragments WHERE agent_id = ?", (agent_id,))
        return int(row[0]) if row else 0

    def _fragment_filters(
        self,
        agent_id: str,
        *,
        scope: MemoryScope | None,
        states: Iterable[MemoryState] | None,
        tier: MemoryTier | None,
        session_id: str | None,
        session_prefix: str | None,
        tags: list[str] | None,
        block_id: str | None,
        block_ids: list[str] | None,
    ) -> tuple[str, list[object]]:
        """Build the WHERE clause (and params) shared by listings and id queries.

        The (sql, params) pair is also the retrievable_ids() cache key, so the
        same filters must always produce the same text.
        """
        sql = "agent_id = ?"
        params: list[object] = [agent_id]
        # Block-level quarantine: a quarantined block's members never surface.
        sql += (
            " AND (block_id IS NULL OR block_id NOT IN"
            " (SELECT id FROM blocks WHERE status = ?))"
        )
        params.append(BlockStatus.QUARANTINED.value)
        if block_id is not None:
            sql += " AND block_id = ?"
            params.append(block_id)
        elif block_ids:
            placeholders = ",".join("?" for _ in block_ids)
            sql += f" AND block_id IN ({placeholders})"
            params.extend(block_ids)
        if scope is not None:
            sql += " AND scope = ?"
            params.append(scope.value)
        if tier is not None:
            sql += " AND tier = ?"
            params.append(tier.value)
        if session_id is not None:
            sql += " AND session_id = ?"
            params.append(session_id)
        elif session_prefix is not None:
            sql += " AND session_id LIKE ? ESCAPE '\\'"
            params.append(f"{_escape_like(session_prefix)}%")
        if tags:
            for tag in tags:
                sql += " AND EXISTS (SELECT 1 FROM json_each(tags) WHERE value = ?)"
                params.append(tag)
        if states:
            state_list = list(states)
            placeholders = ",".join("?" for _ in state_list)
            sql += f" AND state IN ({placeholders})"
            params.extend(state.value for state in state_list)
        return sql, params
