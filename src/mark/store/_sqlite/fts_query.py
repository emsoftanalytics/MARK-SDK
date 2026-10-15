"""Bounded-cost FTS5 queries: which terms to require, which only to rank.

A plain ``t1 OR t2 OR ...`` query makes SQLite score every document that
contains any term. At a million memories one common word ("svc", "after")
matches most rows and a single recall spends seconds in ``bm25()``.

For indexes above ``SMALL_INDEX`` rows the cost is kept proportional to the
*rare* evidence instead:

1. document frequencies come from ``count(*) ... MATCH`` (fast doclist scans,
   cached per term until the index grows by 10%);
2. the rarest terms are *required* (any of them) while together they cover
   at most ``RARE_BUDGET`` documents; all their matches are fetched with
   their exact ``bm25()`` (so scope filtering in Python is exact);
3. every other term only *ranks* those matches: a cheap rowid-only
   ``(required) AND term`` query finds which candidates contain it, and each
   adds the term's BM25 IDF (its weight for a typical-length memory with
   one occurrence). Asking SQLite to score such terms itself would make it
   walk their whole doclists to count them;
4. terms in more than half of all documents are dropped: BM25 gives them no
   weight anyway;
5. when even the rarest term is common, only the newest window of rows that
   is expected to hold ``RARE_BUDGET`` matches is searched (rowids grow
   with insertion order, and FTS5 seeks by rowid).

Small indexes keep the exact ranked OR query, so their results are
identical to earlier releases.
"""
from __future__ import annotations

import heapq
import json
import math
from typing import Callable, Iterable

from mark.store._sqlite.fts_text import fts_quote

SMALL_INDEX = 5_000
RARE_BUDGET = 1_000
# A scoped search with more ids than this post-filters instead of shipping them.
_INLINE_IDS = 2_000
_DF_CACHE_SIZE = 4_096

Fetch = Callable[[str, list], list]


def total_rows(fetch: Fetch, table: str) -> int:
    """Rows in an FTS5 table, read from its averages record (O(1)).

    The record (``<table>_data`` id 1) starts with the total row count as an
    SQLite varint; ``max(rowid)`` is the fallback if the layout is unexpected.
    """
    try:
        rows = fetch(f"SELECT block FROM {table}_data WHERE id = 1", [])
        if rows and rows[0][0]:
            return _varint(bytes(rows[0][0]))
    except Exception:  # noqa: BLE001 - fall back to a slower but exact query
        pass
    rows = fetch(f"SELECT max(rowid) FROM {table}", [])
    return int(rows[0][0] or 0) if rows else 0


def _varint(data: bytes) -> int:
    value = 0
    for i, byte in enumerate(data[:9]):
        if i == 8:
            return (value << 8) | byte
        value = (value << 7) | (byte & 0x7F)
        if not byte & 0x80:
            return value
    return value


class DocFreqCache:
    """Approximate document frequencies per (table, phrase), refreshed as the index grows."""

    def __init__(self) -> None:
        self._entries: dict[tuple[str, str], tuple[int, int]] = {}

    def get(self, fetch: Fetch, table: str, phrase: str, total: int) -> int:
        key = (table, phrase)
        cached = self._entries.get(key)
        if cached is not None and abs(total - cached[1]) * 10 <= max(cached[1], 1):
            return cached[0]
        rows = fetch(f"SELECT count(*) FROM {table} WHERE {table} MATCH ?", [fts_quote(phrase)])
        df = int(rows[0][0]) if rows else 0
        if len(self._entries) >= _DF_CACHE_SIZE:
            self._entries.clear()
        self._entries[key] = (df, total)
        return df


def search(
    fetch: Fetch,
    table: str,
    phrases: list[str],
    agent_id: str,
    limit: int,
    include_ids: set[str] | frozenset[str] | None,
    cache: DocFreqCache,
) -> list[tuple[str, float]]:
    """(fragment_id, score) pairs for *phrases*, best first; higher is better."""
    if not phrases or limit <= 0:
        return []
    total = total_rows(fetch, table)
    if total <= SMALL_INDEX:
        return _exact(fetch, table, _any(phrases), agent_id, limit, include_ids)
    df = {phrase: cache.get(fetch, table, phrase, total) for phrase in phrases}
    present = sorted((p for p in phrases if df[p] > 0), key=lambda p: df[p])
    if not present:
        return []
    required, covered = [present[0]], df[present[0]]
    for phrase in present[1:]:
        if covered + df[phrase] > RARE_BUDGET:
            break
        required.append(phrase)
        covered += df[phrase]
    window = ""
    if covered > RARE_BUDGET:
        rows = fetch(f"SELECT max(rowid) FROM {table}", [])
        newest = int(rows[0][0] or 0) if rows else 0
        window = f" AND rowid > {newest - int(RARE_BUDGET * total / covered)}"
    must = _any(required)
    # bm25() is lower-is-better; negate so callers get higher-is-better.
    rows = fetch(
        f"SELECT rowid, fragment_id, -bm25({table}) FROM {table}"
        f" WHERE {table} MATCH ? AND agent_id = ?{window}",
        [must, agent_id],
    )
    scores = {
        int(rowid): [str(fid), float(score)] for rowid, fid, score in rows
        if include_ids is None or fid in include_ids
    }
    if not scores:
        return []
    for phrase in present:
        if phrase in required or df[phrase] * 2 > total:
            continue
        idf = math.log((total - df[phrase] + 0.5) / (df[phrase] + 0.5))
        for (rowid,) in fetch(f"SELECT rowid FROM {table} WHERE {table} MATCH ?{window}",
                              [f"({must}) AND {fts_quote(phrase)}"]):
            entry = scores.get(int(rowid))
            if entry is not None:
                entry[1] += idf
    best = heapq.nlargest(limit, scores.values(), key=lambda entry: entry[1])
    return [(fid, score) for fid, score in best]


def _any(phrases: Iterable[str]) -> str:
    return " OR ".join(fts_quote(phrase) for phrase in phrases)


def _exact(
    fetch: Fetch,
    table: str,
    expression: str,
    agent_id: str,
    limit: int,
    include_ids: set[str] | frozenset[str] | None,
) -> list[tuple[str, float]]:
    """One ranked MATCH; scope filtering never ships a large id list to SQLite.

    A small ``include_ids`` set is passed as one JSON parameter. A large one
    is applied in Python to a growing ranked prefix, so in-scope matches are
    never lost behind out-of-scope ones (the result is exact either way).
    """
    sql = f"SELECT fragment_id, -bm25({table}) FROM {table} WHERE {table} MATCH ? AND agent_id = ?"
    params: list[object] = [expression, agent_id]
    if include_ids is not None and len(include_ids) <= _INLINE_IDS:
        sql += " AND fragment_id IN (SELECT value FROM json_each(?))"
        params.append(json.dumps(sorted(include_ids)))
        include_ids = None
    sql += f" ORDER BY bm25({table}) LIMIT ?"
    if include_ids is None:
        return [(str(row[0]), float(row[1])) for row in fetch(sql, [*params, limit])]
    batch = max(limit * 4, 64)
    while True:
        rows = fetch(sql, [*params, batch])
        hits = [(str(row[0]), float(row[1])) for row in rows if row[0] in include_ids]
        if len(hits) >= limit or len(rows) < batch:
            return hits[:limit]
        batch *= 4
