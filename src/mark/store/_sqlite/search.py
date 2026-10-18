"""Retrieval-facing queries: embedding pages for bulk loads and FTS5 lexical search.

The cached retrievable-id sets live in :mod:`mark.store._sqlite.scope_delta`.
"""
from __future__ import annotations

import heapq
import logging
import re
import sqlite3
from typing import Iterator

from mark.store._sqlite import fts_query
from mark.store._sqlite.codec import (
    _FTS_ML_TABLE,
    _FTS_TABLE,
    _RETRIEVABLE_STATES,
    _RETRIEVABLE_VALUES,
    _TTL_LIVE_SQL,
)
from mark.store._sqlite.fts_query import DocFreqCache
from mark.store._sqlite.fts_text import (
    ascii_query_terms,
    fts_quote,
    multilingual_phrase,
    multilingual_query_chunks,
)
from mark.store._sqlite.scope_delta import ScopeCacheMixin
from mark.store._sqlite.vector_snapshot import VectorSnapshotMixin

logger = logging.getLogger("mark.store.sqlite")


class SearchMixin(ScopeCacheMixin, VectorSnapshotMixin):
    """Cached retrievable-id sets, lexical (BM25) search and phrase lookup."""

    _fts_enabled: bool

    def embedding_pages(self, agent_id: str, page_size: int = 20_000) -> Iterator[list[tuple]]:
        """Yield pages of (id, float32 blob, legacy JSON, rowid) for the agent's retrievable rows.

        Raw column values in rowid order, so a vector index can be loaded in
        bulk without building a Python list per vector. Keyset pagination
        keeps each read short (no transaction spans the whole load).
        """
        states = [state.value for state in _RETRIEVABLE_STATES]
        sql = (
            "SELECT id, embedding_blob, embedding, rowid FROM fragments"
            f" WHERE agent_id = ? AND state IN ({','.join('?' for _ in states)}) AND rowid > ?"
            " ORDER BY rowid LIMIT ?"
        )
        last = 0
        with self._reading() as conn:
            # Memory-mapped reads cut the cost of streaming every row (each holds
            # a ~1 KB vector) by about a third; restored when the load ends.
            previous = conn.execute("PRAGMA mmap_size").fetchone()[0]
            conn.execute("PRAGMA mmap_size = 2147418112")
        try:
            while True:
                with self._reading() as conn:
                    cursor = conn.cursor()
                    cursor.row_factory = None  # plain tuples
                    rows = cursor.execute(sql, (agent_id, *states, last, page_size)).fetchall()
                if not rows:
                    return
                last = int(rows[-1][3])
                yield rows  # raw tuples: no per-row copy at a million rows
                if len(rows) < page_size:
                    return
        finally:
            with self._reading() as conn:
                conn.execute(f"PRAGMA mmap_size = {int(previous)}")

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
        include_ids: set[str] | frozenset[str] | None = None,
    ) -> list[tuple[str, float]]:
        """BM25 keyword search over fragment content and tags.

        Returns (fragment_id, score) pairs, best first; higher score is better.
        An encrypted store (no FTS) answers identifier-like tokens only, from
        its keyed token sketches. Returns [] when FTS5 is unavailable, when
        the query has no usable tokens, or when SQLite rejects the query.
        Results are not state-filtered: intersect with retrievable_ids() (or
        pass it as include_ids) to restrict to retrievable fragments.

        ASCII words are searched in the English (porter) table and other
        scripts, CJK and emoji in the multilingual table; a fragment matching
        in both gets the sum of its scores. Query cost is bounded at any store
        size (see :mod:`mark.store._sqlite.fts_query`).
        """
        if limit <= 0 or (include_ids is not None and not include_ids):
            return []
        if not self._fts_enabled:
            # Encrypted: identifier tokens only, from the keyed sketches (sketch_search).
            sketch = getattr(self, "_sketch_lexical", None) if getattr(self, "_encrypting", False) else None
            return sketch(agent_id, query, limit, include_ids) if sketch is not None else []
        channels = [(_FTS_TABLE, ascii_query_terms(query))]
        if getattr(self, "_fts_ml_enabled", False):
            channels.append((_FTS_ML_TABLE, multilingual_query_chunks(query)))
        cache = self.__dict__.setdefault("_fts_df_cache", DocFreqCache())
        scores: dict[str, float] = {}
        for table, phrases in channels:
            try:
                hits = fts_query.search(self._fetchall, table, phrases, agent_id, limit, include_ids, cache)
            except sqlite3.OperationalError:
                logger.debug("FTS query rejected: %r", phrases, exc_info=True)
                continue
            for fragment_id, score in hits:
                scores[fragment_id] = scores.get(fragment_id, 0.0) + score
        return heapq.nlargest(limit, scores.items(), key=lambda item: item[1])

    def phrase_exists(self, agent_id: str, phrase: str) -> bool | None:
        """Whether *phrase* (as consecutive tokens) occurs in any of the agent's memories.

        Returns None when it cannot be answered (FTS5 unavailable, encrypted
        store, or no usable tokens), so callers can fall back to other evidence.
        Non-ASCII phrases (CJK, accented words, emoji) are looked up in the
        multilingual table, where CJK is segmented and nothing is stemmed.
        """
        if not self._fts_enabled:
            return None
        table, match = _FTS_TABLE, None
        ml_phrase = multilingual_phrase(phrase) if getattr(self, "_fts_ml_enabled", False) else None
        if ml_phrase is not None:
            table, match = _FTS_ML_TABLE, fts_quote(ml_phrase)
        else:
            tokens = re.findall(r"\w+", phrase.lower())
            if tokens:
                match = fts_quote(" ".join(tokens))
        if match is None:
            return None
        # Only retrievable, unexpired memories count: a quarantined injection
        # attempt or an expired note must not make an unknown subject look known.
        states = ",".join("?" for _ in _RETRIEVABLE_VALUES)
        live = _TTL_LIVE_SQL.replace("created_at", "f.created_at").replace("ttl_seconds", "f.ttl_seconds")
        try:
            row = self._fetchone(
                f"SELECT 1 FROM {table} JOIN fragments f ON f.id = {table}.fragment_id"
                f" WHERE {table} MATCH ? AND {table}.agent_id = ?"
                f" AND f.state IN ({states}){live} LIMIT 1",
                (match, agent_id, *sorted(_RETRIEVABLE_VALUES)),
            )
        except sqlite3.OperationalError:
            logger.debug("FTS phrase query rejected: %r", match, exc_info=True)
            return None
        return row is not None
