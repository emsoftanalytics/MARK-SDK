"""Retrievable-id scopes kept up to date incrementally.

``retrievable_id_set`` is on the hot path of every recall. Computing it is
one indexed SQL scan of the agent's rows (0.15 s at 100k memories, 2-3 s at
1M), so its result is cached per filter set; this module keeps that cache
valid across writes without rescanning.

Write log: every structural write (``_writing``) bumps ``write_version``
and, in the same transaction, records in ``write_log`` the fragment ids it
changed, or NULL when the writer did not say (a safe "unknown"). Writers
that know their ids call ``_note_changed`` inside the transaction. Since
the log lives in the database, writes from other processes are seen too;
processes running older code log nothing, which reads as unknown.

On a cache hit with an older version, the logged ids since that version are
re-evaluated against the cached filter (one primary-key query for a handful
of rows) and applied to the cached set as a delta. A full recompute happens
only when the change set is unknown (a NULL row, pruned or missing rows) or
larger than ``MAX_DELTA_IDS``, and at most every ``_IDS_CACHE_MAX_AGE``
seconds as a safety net.

TTL: an entry also records when the soonest live TTL memory in it expires.
At that moment the expired ids are read through the partial TTL index and
removed, again without rescanning the agent.
"""
from __future__ import annotations

import json
import math
import sqlite3
import time
from operator import itemgetter
from typing import Iterable, NamedTuple

from mark.store._sqlite.codec import _IDS_CACHE_MAX_AGE, _IDS_CACHE_SIZE, _RETRIEVABLE_STATES, _SQL_CHUNK, _TTL_LIVE_SQL
from mark.store._sqlite.scope_ids import ScopeIds
from mark.types import MemoryScope, MemoryTier

WRITE_LOG_DDL = "CREATE TABLE IF NOT EXISTS write_log (version INTEGER PRIMARY KEY, ids TEXT)"
# Above this many changed ids (summed over the versions to catch up on) a
# rescan is cheaper than per-id lookups.
MAX_DELTA_IDS = 5_000
# A single write logging more ids is logged as unknown: bulk imports would
# otherwise make the log large (~40 bytes per id), and they are rare and
# already cost far more than one rescan.
MAX_LOGGED_IDS = 512
_LOG_KEEP = 1_024  # versions kept in write_log (at most ~20 MB); older caches simply rescan
_PRUNE_EVERY = 64

# Seconds until a row expires; NULL when it never does (no TTL, unparsable date).
_TTL_LEFT_SQL = (
    "CASE WHEN ttl_seconds IS NULL OR julianday(created_at) IS NULL THEN NULL"
    " ELSE (julianday(created_at) - julianday('now')) * 86400.0 + ttl_seconds END"
)


class _Entry(NamedTuple):
    ids: ScopeIds
    version: int
    expiry: float  # monotonic time the soonest live TTL member expires (inf: none)
    refresh: float  # monotonic time of the next full recompute (safety net)


class ScopeCacheMixin:
    """Cached, incrementally maintained retrievable-id sets (see module docstring)."""

    _ids_cache: dict[tuple, _Entry]

    # ── write side ────────────────────────────────────────────────────────────

    def _note_changed(self, fragment_ids: Iterable[str]) -> None:
        """Record fragment ids changed by the current structural transaction (may be empty)."""
        notes = self._local.__dict__.get("write_notes")
        if notes is None:
            notes = self._local.write_notes = []
        notes.extend(fragment_ids)

    def _log_structural_write(self, conn: sqlite3.Connection) -> None:
        """Log the version just bumped with the noted ids (NULL: unknown); called by ``_writing``."""
        notes = self._local.__dict__.pop("write_notes", None)
        row = conn.execute("SELECT value FROM schema_info WHERE key = 'write_version'").fetchone()
        if row is None:
            return
        version = int(row[0])
        ids = None if notes is None or len(notes) > MAX_LOGGED_IDS else json.dumps(list(dict.fromkeys(notes)))
        try:
            conn.execute("INSERT OR REPLACE INTO write_log (version, ids) VALUES (?, ?)", (version, ids))
            if version % _PRUNE_EVERY == 0:
                conn.execute("DELETE FROM write_log WHERE version <= ?", (version - _LOG_KEEP,))
        except sqlite3.OperationalError:  # table not created yet (first schema init): rows read as unknown
            pass

    def _reset_write_notes(self) -> None:
        """Start a structural transaction with an unknown change set."""
        self._local.__dict__.pop("write_notes", None)

    # ── read side ─────────────────────────────────────────────────────────────

    def retrievable_ids(
        self, agent_id: str, *, session_id: str | None = None, session_prefix: str | None = None,
        tags: list[str] | None = None, tier: MemoryTier | None = None, scope: MemoryScope | None = None,
        block_id: str | None = None, block_ids: list[str] | None = None,
    ) -> set[str]:
        """Return ids of every retrievable fragment matching the filters (no row limit).

        Retrievable means: state in {UNVERIFIED, VERIFIED, PROMOTED}, not a
        member of a QUARANTINED block, and not TTL-expired. Evaluated entirely
        in SQL; no fragment rows are decoded. Returns a copy the caller owns.
        """
        return set(self.retrievable_id_set(
            agent_id, session_id=session_id, session_prefix=session_prefix, tags=tags,
            tier=tier, scope=scope, block_id=block_id, block_ids=block_ids,
        ))

    def retrievable_id_set(
        self, agent_id: str, *, session_id: str | None = None, session_prefix: str | None = None,
        tags: list[str] | None = None, tier: MemoryTier | None = None, scope: MemoryScope | None = None,
        block_id: str | None = None, block_ids: list[str] | None = None,
    ) -> ScopeIds:
        """Like :meth:`retrievable_ids`, but returns the cached, immutable set itself.

        The same object comes back while nothing changed, so callers can
        recognise an unchanged scope by identity (``is``); after a small
        change the new set carries its delta from the previous one.
        """
        where, params = self._fragment_filters(
            agent_id, scope=scope, states=_RETRIEVABLE_STATES, tier=tier, session_id=session_id,
            session_prefix=session_prefix, tags=tags, block_id=block_id, block_ids=block_ids,
        )
        key = (where, tuple(params))
        # Read the version *before* any query: a write racing with the queries
        # leaves an older version in the entry, so it is applied again next time
        # (re-applying a delta is idempotent: rows are re-read, not replayed).
        version = self._write_version()
        entry = self._ids_cache.get(key)
        now = time.monotonic()
        if entry is not None and now < entry.refresh:
            if entry.version != version:
                entry = self._apply_log(entry, version, where, params, now)
            if entry is not None and now >= entry.expiry:
                entry = self._expire(entry, agent_id, now)
        else:
            entry = None
        if entry is None:
            entry = self._full_scope(version, where, params, agent_id, now)
        with self._lock:
            # Bounded FIFO eviction (dicts preserve insertion order).
            if key not in self._ids_cache and len(self._ids_cache) >= _IDS_CACHE_SIZE:
                self._ids_cache.pop(next(iter(self._ids_cache)))
            self._ids_cache[key] = entry
        return entry.ids

    def _full_scope(self, version: int, where: str, params: list, agent_id: str, now: float) -> _Entry:
        """Scan the agent's rows once (the O(N) path)."""
        with self._reading() as conn:
            cursor = conn.cursor()
            cursor.row_factory = None  # plain tuples: ~3x faster than sqlite3.Row
            rows = cursor.execute(f"SELECT id FROM fragments WHERE {where}{_TTL_LIVE_SQL}", params).fetchall()
            ids = ScopeIds(map(itemgetter(0), rows))
        return _Entry(ids, version, now + self._next_expiry(agent_id), now + _IDS_CACHE_MAX_AGE)

    def _next_expiry(self, agent_id: str) -> float:
        """Seconds until the soonest live TTL fragment of the agent expires (inf: none)."""
        left = self._fetchone(
            f"SELECT MIN({_TTL_LEFT_SQL}) FROM fragments WHERE agent_id = ? AND ttl_seconds IS NOT NULL"
            " AND julianday(created_at) IS NOT NULL"
            " AND (julianday('now') - julianday(created_at)) * 86400.0 <= ttl_seconds",
            (agent_id,),
        )[0]
        return math.inf if left is None else max(0.0, float(left))

    def _apply_log(self, entry: _Entry, version: int, where: str, params: list, now: float) -> _Entry | None:
        """Apply the logged changes since ``entry.version``; None when they are unknown or too many."""
        touched = self._changed_since(entry.version, version)
        if touched is None:
            return None
        present: set[str] = set()
        soonest = math.inf
        ordered = sorted(touched)
        for start in range(0, len(ordered), _SQL_CHUNK):
            chunk = ordered[start:start + _SQL_CHUNK]
            marks = ",".join("?" for _ in chunk)
            # The id IN (...) primary-key lookup leads; the filter checks each row.
            for fragment_id, left in self._fetchall(
                f"SELECT id, {_TTL_LEFT_SQL} FROM fragments WHERE id IN ({marks})"
                f" AND {where}{_TTL_LIVE_SQL}", [*chunk, *params],
            ):
                present.add(str(fragment_id))
                if left is not None:
                    soonest = min(soonest, max(0.0, float(left)))
        ids = entry.ids
        if not present.issubset(ids) or not ids.isdisjoint(touched.difference(present)):
            ids = ScopeIds.derive(ids, present, touched.difference(present))
        return _Entry(ids, version, min(entry.expiry, now + soonest), entry.refresh)

    def _changed_since(self, old: int, new: int) -> set[str] | None:
        """Fragment ids changed by versions (old, new], or None when not known exactly."""
        if new < old or new - old > _LOG_KEEP:
            return None
        try:
            rows = self._fetchall(
                "SELECT ids FROM write_log WHERE version > ? AND version <= ?", (old, new),
            )
        except sqlite3.OperationalError:
            return None
        if len(rows) != new - old:  # pruned, or written by code that does not log
            return None
        touched: set[str] = set()
        for (ids,) in rows:
            if ids is None:
                return None
            touched.update(json.loads(ids))
            if len(touched) > MAX_DELTA_IDS:
                return None
        return touched

    def _expire(self, entry: _Entry, agent_id: str, now: float) -> _Entry:
        """Drop members whose TTL elapsed (read through the partial TTL index)."""
        expired = self._fetchall(
            "SELECT id FROM fragments WHERE agent_id = ? AND ttl_seconds IS NOT NULL"
            " AND julianday(created_at) IS NOT NULL"
            " AND (julianday('now') - julianday(created_at)) * 86400.0 > ttl_seconds",
            (agent_id,),
        )
        gone = {str(row[0]) for row in expired}
        ids = entry.ids if entry.ids.isdisjoint(gone) else ScopeIds.derive(entry.ids, (), gone)
        return _Entry(ids, entry.version, now + self._next_expiry(agent_id), entry.refresh)
