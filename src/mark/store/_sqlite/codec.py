"""Shared constants and pure helpers for the SQLite store.

Nothing here touches store state: these are SQL fragments, tuning limits and
value codecs (embeddings, datetimes, LIKE escaping, fact keys) used by the
store mixins. ``mark.store.sqlite`` re-exports every name for compatibility.
"""
from __future__ import annotations

import json
import re
import sqlite3
import sys
from array import array
from datetime import datetime
from typing import Callable, Iterator

from mark.types import MemoryState

# States a fragment must be in to surface in retrieval.
_RETRIEVABLE_STATES: tuple[MemoryState, ...] = (
    MemoryState.UNVERIFIED,
    MemoryState.VERIFIED,
    MemoryState.PROMOTED,
)
_RETRIEVABLE_VALUES = frozenset(state.value for state in _RETRIEVABLE_STATES)

# Stay well below SQLite's default host-parameter limit (999 on old builds).
_SQL_CHUNK = 900

# retrievable_ids() cache bounds: entries are also invalidated by write_version
# and by the next TTL expiry among the agent's live fragments.
_IDS_CACHE_SIZE = 64
_IDS_CACHE_MAX_AGE = 300.0

# A fragment row is live unless a TTL is set and has elapsed. Rows whose
# created_at cannot be parsed by SQLite are kept (fail open, like is_expired).
_TTL_LIVE_SQL = (
    " AND (ttl_seconds IS NULL OR julianday(created_at) IS NULL"
    " OR (julianday('now') - julianday(created_at)) * 86400.0 <= ttl_seconds)"
)

_FRAGMENT_COLUMNS = (
    "id", "content", "block_id", "scope", "tier", "agent_id", "session_id",
    "importance", "state", "confidence", "ttl_seconds", "created_at",
    "updated_at", "last_accessed_at", "tags", "source", "metadata",
    "embedding", "contradiction_ids", "embedding_blob",
)

# Upsert keeps the fragment's rowid stable on update, which matters because
# the FTS index is keyed by fragments.rowid.
_UPSERT_FRAGMENT_SQL = (
    f"INSERT INTO fragments ({', '.join(_FRAGMENT_COLUMNS)}) VALUES ("
    + ", ".join(f":{name}" for name in _FRAGMENT_COLUMNS)
    + ") ON CONFLICT(id) DO UPDATE SET "
    + ", ".join(f"{name} = excluded.{name}" for name in _FRAGMENT_COLUMNS if name != "id")
)

_FTS_TABLE = "fragments_fts"
# schema_info key marking that the FTS index has been backfilled from fragments.
_FTS_META_KEY = "fts_backfilled"

ListenerCallback = Callable[[str, str], None]


def _pack_embedding(embedding: list[float] | None) -> bytes | None:
    """Encode an embedding as float32 little-endian bytes (no numpy needed)."""
    if embedding is None:
        return None
    packed = array("f", (float(value) for value in embedding))
    if sys.byteorder != "little":  # pragma: no cover - big-endian hosts only
        packed.byteswap()
    return packed.tobytes()


def _unpack_embedding(blob: bytes) -> list[float]:
    """Inverse of :func:`_pack_embedding`."""
    unpacked = array("f")
    unpacked.frombytes(bytes(blob))
    if sys.byteorder != "little":  # pragma: no cover - big-endian hosts only
        unpacked.byteswap()
    return unpacked.tolist()


def _decode_embedding(blob: object, legacy_json: object) -> list[float] | None:
    """Prefer the float32 blob; fall back to the legacy JSON column."""
    if blob is not None:
        return _unpack_embedding(blob)  # type: ignore[arg-type]
    if legacy_json:
        return [float(value) for value in json.loads(str(legacy_json))]
    return None


def _fts_match_expression(query: str) -> str | None:
    """Build a safe FTS5 MATCH expression: quoted tokens OR-joined."""
    tokens = [
        token
        for token in re.findall(r"\w+", query.lower())
        if len(token) >= 2 or token.isdigit()
    ]
    if not tokens:
        return None
    return " OR ".join('"' + token.replace('"', '""') + '"' for token in dict.fromkeys(tokens))


def _probe_fts5(conn: sqlite3.Connection) -> bool:
    """True when this SQLite build supports FTS5 (probed with a temp table)."""
    try:
        conn.execute("CREATE VIRTUAL TABLE IF NOT EXISTS temp._mark_fts5_probe USING fts5(x)")
        conn.execute("DROP TABLE IF EXISTS temp._mark_fts5_probe")
        return True
    except sqlite3.OperationalError:
        return False


def _table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    """Column names of *table*, read from PRAGMA table_info."""
    return {str(row[1]) for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}


def _escape_like(value: str) -> str:
    """Escape LIKE wildcards; pair with ``ESCAPE '\\'`` in the SQL."""
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _chunks(items: list[str], size: int) -> Iterator[list[str]]:
    """Yield consecutive slices of at most *size* items."""
    for start in range(0, len(items), size):
        yield items[start:start + size]


def _as_bytes(value: object) -> bytes:
    """Content column may hold bytes (ciphertext) or text (legacy rows)."""
    return value if isinstance(value, bytes) else str(value).encode("utf-8")


def _dt_to_text(value: datetime | None) -> str | None:
    return None if value is None else value.isoformat()


def _parse_dt(value: object) -> datetime:
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(str(value))


def _parse_optional_dt(value: object) -> datetime | None:
    if value is None:
        return None
    return _parse_dt(value)


def _optional_str(value: object) -> str | None:
    return None if value is None else str(value)


def _fact_key(text: str) -> str:
    """Normalise a fact subject/attribute key.

    Case, whitespace, hyphens and underscores are folded ('Atlas-Core' ==
    'atlas core' == 'atlas_core'); every other symbol is kept, so 'C++',
    'C#' and 'C' remain distinct subjects.
    """
    folded = re.sub(r"[\s_\-]+", " ", str(text).lower().replace("’", "'"))
    return folded.strip(" \"'()[]{}").rstrip(" .,;:!?")
