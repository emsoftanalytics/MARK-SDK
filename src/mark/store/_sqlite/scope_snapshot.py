"""Scope snapshots: a cold start rebuilds a large agent's retrievable-id set without a scan.

The first recall of a process needs the agent's retrievable ids. Computing
them scans every row of the agent (4-20 s at a million memories, most of a
cold start). A scope snapshot stores those ids (about 37 bytes each, in
parts of ``PART_IDS``) with the ``write_version`` they are exact at, a
hash of the filter they answer, and a digest. A cold start then loads the
ids and applies the ``write_log`` delta since that version (see
``scope_delta``), which yields exactly what the scan would.

It is written:
  - when a vector snapshot is saved whose ids are exactly the scope at that
    version (the store's cached scope says so);
  - after a cold start had to scan an agent of at least ``MIN_ROWS``
    memories (no snapshot yet, or the log since it was incomplete: pruned,
    or written by an older build that logs nothing), so the next cold
    start does not scan again.

Any doubt (digest or count mismatch, an unknown change in the log) means
the snapshot is not used and the scan runs, as before. Only ids are stored:
no content, no vectors.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from typing import Iterable

SCOPE_META_PREFIX = "scope_snapshot:"
PART_IDS = 65_536
SCOPE_DDL = (
    "CREATE TABLE IF NOT EXISTS scope_snapshots (agent_id TEXT NOT NULL, part INTEGER NOT NULL,"
    " ids TEXT NOT NULL, PRIMARY KEY (agent_id, part))"
)


class ScopeSnapshotMixin:
    """Save and load per-agent scope snapshots (see module docstring)."""

    def _default_scope_key(self, agent_id: str) -> tuple[str, list[object]]:
        """(where, params) of the agent's unfiltered retrievable scope (the recall default)."""
        from mark.store._sqlite.codec import _RETRIEVABLE_STATES

        return self._fragment_filters(
            agent_id, scope=None, states=_RETRIEVABLE_STATES, tier=None, session_id=None,
            session_prefix=None, tags=None, block_id=None, block_ids=None,
        )

    def scope_snapshot_wanted(self, agent_id: str, where: str, params: list[object], size: int) -> bool:
        """Whether a freshly scanned scope is worth saving (default filter, large agent)."""
        from mark.store._sqlite import vector_snapshot

        return size >= vector_snapshot.MIN_ROWS and (where, list(params)) == self._default_scope_key(agent_id)

    def save_scope_snapshot(
        self, agent_id: str, where: str, params: list[object], ids: Iterable[str], version: int,
        conn: sqlite3.Connection | None = None,
    ) -> None:
        """Replace the agent's scope snapshot with *ids*, exact at *version* (inside *conn* if given)."""
        if conn is None:
            with self._writing(structural=False) as own:
                return self.save_scope_snapshot(agent_id, where, params, ids, version, own)
        ordered = list(ids)
        texts = ["\n".join(ordered[i:i + PART_IDS]) for i in range(0, len(ordered), PART_IDS)]
        conn.execute(SCOPE_DDL)  # cheap no-op; tolerates stores opened by older code since
        conn.execute("DELETE FROM scope_snapshots WHERE agent_id = ?", (agent_id,))
        conn.executemany("INSERT INTO scope_snapshots (agent_id, part, ids) VALUES (?, ?, ?)",
                         [(agent_id, part, text) for part, text in enumerate(texts)])
        info = {"version": int(version), "key": _key_hash(where, params), "count": len(ordered),
                "digest": _ids_digest(texts)}
        conn.execute(
            "INSERT INTO schema_info (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (SCOPE_META_PREFIX + agent_id, json.dumps(info)),
        )
        return None

    def _record_scope_if_exact(self, conn: sqlite3.Connection, agent_id: str, ids: list[str], version: int) -> None:
        """Save *ids* as the scope snapshot when they equal the cached default scope at *version*."""
        where, params = self._default_scope_key(agent_id)
        entry = getattr(self, "_ids_cache", {}).get((where, tuple(params)))
        if entry is None or entry.version != version or len(entry.ids) != len(ids):
            return
        if all(map(entry.ids.__contains__, ids)):  # ids are unique, so equal sizes => equal sets
            self.save_scope_snapshot(agent_id, where, params, ids, version, conn)

    def scope_snapshot_version(self, agent_id: str) -> int | None:
        """The ``write_version`` the agent's scope snapshot is exact at, or None."""
        raw = self.get_meta(SCOPE_META_PREFIX + agent_id)
        try:
            return int(json.loads(raw)["version"]) if raw else None
        except (ValueError, KeyError, TypeError):
            return None

    def snapshot_scope_ids(self, agent_id: str, where: str, params: list[object]) -> tuple[int, list[str]] | None:
        """(write_version, ids) of the agent's scope snapshot for this filter, or None."""
        raw = self.get_meta(SCOPE_META_PREFIX + agent_id)
        try:
            info = json.loads(raw) if raw else None
            if not info or info["key"] != _key_hash(where, params):
                return None
            texts = [str(row[0]) for row in self._fetchall(
                "SELECT ids FROM scope_snapshots WHERE agent_id = ? ORDER BY part", (agent_id,))]
            if not texts or _ids_digest(texts) != info["digest"]:
                return None
            ids = [fid for text in texts for fid in text.split("\n")]
            return (int(info["version"]), ids) if len(ids) == int(info["count"]) else None
        except (ValueError, KeyError, TypeError, sqlite3.Error):
            return None


def _key_hash(where: str, params: list[object]) -> str:
    return hashlib.sha256(json.dumps([where, [str(p) for p in params]]).encode("utf-8")).hexdigest()[:32]


def _ids_digest(texts: Iterable[str]) -> str:
    digest = hashlib.blake2b(digest_size=16)
    for text in texts:
        digest.update(text.encode("utf-8"))
        digest.update(b"\x00")
    return digest.hexdigest()
