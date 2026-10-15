"""Retrieval-facing queries: retrievable id sets and FTS5 lexical search.

``retrievable_ids`` is on the hot path of every recall, so its result is
cached per filter set. A cache entry is valid only while all hold:
  - the persisted ``write_version`` is unchanged (bumped by every structural
    write, in this or any other process sharing the database file);
  - the next TTL expiry among the agent's live fragments has not passed;
  - it is younger than ``_IDS_CACHE_MAX_AGE`` seconds.
"""
from __future__ import annotations

import json
import logging
import re
import sqlite3
import time

from mark.store._sqlite.codec import (
    _FTS_TABLE,
    _IDS_CACHE_MAX_AGE,
    _IDS_CACHE_SIZE,
    _RETRIEVABLE_STATES,
    _RETRIEVABLE_VALUES,
    _TTL_LIVE_SQL,
    _fts_match_expression,
)
from mark.types import MemoryScope, MemoryTier

logger = logging.getLogger("mark.store.sqlite")


class SearchMixin:
    """Cached retrievable-id sets, lexical (BM25) search and phrase lookup."""

    _ids_cache: dict[tuple, tuple[frozenset[str], int, float]]
    _fts_enabled: bool

    def retrievable_ids(
        self,
        agent_id: str,
        *,
        session_id: str | None = None,
        session_prefix: str | None = None,
        tags: list[str] | None = None,
        tier: MemoryTier | None = None,
        scope: MemoryScope | None = None,
        block_id: str | None = None,
        block_ids: list[str] | None = None,
    ) -> set[str]:
        """Return ids of every retrievable fragment matching the filters (no row limit).

        Retrievable means: state in {UNVERIFIED, VERIFIED, PROMOTED}, not a
        member of a QUARANTINED block, and not TTL-expired. Evaluated entirely
        in SQL; no fragment rows are decoded.
        """
        where, params = self._fragment_filters(
            agent_id,
            scope=scope,
            states=_RETRIEVABLE_STATES,
            tier=tier,
            session_id=session_id,
            session_prefix=session_prefix,
            tags=tags,
            block_id=block_id,
            block_ids=block_ids,
        )
        key = (where, tuple(params))
        # Read the version *before* the query: a write racing with the query
        # then leaves a stale version in the entry, so it is simply refetched.
        version = self._write_version()
        cached = self._ids_cache.get(key)
        if cached is not None and cached[1] == version and time.monotonic() < cached[2]:
            return set(cached[0])
        with self._reading() as conn:
            cursor = conn.cursor()
            cursor.row_factory = None  # plain tuples: ~3x faster than sqlite3.Row
            ids = frozenset(
                row[0] for row in cursor.execute(
                    f"SELECT id FROM fragments WHERE {where}{_TTL_LIVE_SQL}", params
                )
            )
            # Seconds until the soonest live fragment of this agent expires;
            # the cached set must not outlive that moment.
            next_expiry = cursor.execute(
                "SELECT MIN((julianday(created_at) - julianday('now')) * 86400.0 + ttl_seconds)"
                " FROM fragments WHERE agent_id = ? AND ttl_seconds IS NOT NULL"
                " AND julianday(created_at) IS NOT NULL"
                " AND (julianday('now') - julianday(created_at)) * 86400.0 <= ttl_seconds",
                (agent_id,),
            ).fetchone()[0]
        ttl_left = _IDS_CACHE_MAX_AGE if next_expiry is None else min(_IDS_CACHE_MAX_AGE, max(0.0, next_expiry))
        with self._lock:
            # Bounded FIFO eviction (dicts preserve insertion order).
            if len(self._ids_cache) >= _IDS_CACHE_SIZE:
                self._ids_cache.pop(next(iter(self._ids_cache)))
            self._ids_cache[key] = (ids, version, time.monotonic() + ttl_left)
        return set(ids)

    def _write_version(self) -> int:
        """Persisted structural-write counter; -1 if it has not been seeded."""
        row = self._fetchone("SELECT value FROM schema_info WHERE key = 'write_version'")
        return int(row[0]) if row is not None else -1

    def lexical_search(
        self,
        agent_id: str,
        query: str,
        *,
        limit: int = 50,
        include_ids: set[str] | None = None,
    ) -> list[tuple[str, float]]:
        """BM25 keyword search over fragment content and tags.

        Returns (fragment_id, score) pairs, best first; higher score is better.
        Returns [] when FTS5 is unavailable (or the store is encrypted), when
        the query has no usable tokens, or when SQLite rejects the query.
        Results are not state-filtered: intersect with retrievable_ids() (or
        pass it as include_ids) to restrict to retrievable fragments.
        """
        if not self._fts_enabled or limit <= 0:
            return []
        if include_ids is not None and not include_ids:
            return []
        match = _fts_match_expression(query)
        if match is None:
            return []
        # bm25() is lower-is-better; negate so callers get higher-is-better.
        sql = (
            f"SELECT fragment_id, -bm25({_FTS_TABLE}) AS score FROM {_FTS_TABLE}"
            f" WHERE {_FTS_TABLE} MATCH ? AND agent_id = ?"
        )
        params: list[object] = [match, agent_id]
        if include_ids is not None:
            # One JSON parameter filters exactly at any size. Post-filtering a
            # truncated hit list would silently drop in-scope matches whenever
            # many out-of-scope documents rank higher.
            sql += " AND fragment_id IN (SELECT value FROM json_each(?))"
            params.append(json.dumps(sorted(include_ids)))
        sql += f" ORDER BY bm25({_FTS_TABLE}) LIMIT ?"
        params.append(limit)
        try:
            rows = self._fetchall(sql, params)
        except sqlite3.OperationalError:
            logger.debug("FTS query rejected: %r", match, exc_info=True)
            return []
        return [(str(row[0]), float(row[1])) for row in rows]

    def phrase_exists(self, agent_id: str, phrase: str) -> bool | None:
        """Whether *phrase* (as consecutive tokens) occurs in any of the agent's memories.

        Returns None when it cannot be answered (FTS5 unavailable, encrypted
        store, or no usable tokens), so callers can fall back to other evidence.
        """
        if not self._fts_enabled:
            return None
        tokens = re.findall(r"\w+", phrase.lower())
        if not tokens:
            return None
        match = '"' + " ".join(token.replace('"', '""') for token in tokens) + '"'
        # Only retrievable, unexpired memories count: a quarantined injection
        # attempt or an expired note must not make an unknown subject look known.
        states = ",".join("?" for _ in _RETRIEVABLE_VALUES)
        try:
            row = self._fetchone(
                f"SELECT 1 FROM {_FTS_TABLE} JOIN fragments f ON f.id = {_FTS_TABLE}.fragment_id"
                f" WHERE {_FTS_TABLE} MATCH ? AND {_FTS_TABLE}.agent_id = ?"
                f" AND f.state IN ({states}){_TTL_LIVE_SQL.replace('created_at', 'f.created_at').replace('ttl_seconds', 'f.ttl_seconds')}"
                " LIMIT 1",
                (match, agent_id, *sorted(_RETRIEVABLE_VALUES)),
            )
        except sqlite3.OperationalError:
            logger.debug("FTS phrase query rejected: %r", match, exc_info=True)
            return None
        return row is not None
