"""Every retrievable memory that may mention a subject: uncapped, paged reads.

Recall returns a relevance top-k; a subject's *timeline* needs every record
that mentions it, however many there are. ``mention_candidates`` pages
through them in rowid order (keyset pagination, so each page is an index
range scan and memory use is bounded by the page size).

Invariants:
  - only retrievable rows are returned: state in {UNVERIFIED, VERIFIED,
    PROMOTED}, not in a QUARANTINED block, and not TTL-expired — the same
    definition ``retrievable_ids`` uses;
  - results are a *superset* of whole-phrase mentions. FTS5 stems words
    (porter) and folds diacritics, and without FTS5 (encrypted stores) every
    retrievable row is a candidate, so callers must confirm the match on the
    decrypted content (``mark.memory.timeline`` does, with whole-anchor
    matching, so "titan" never matches "titanium");
  - embeddings are not decoded (a timeline never needs them), but the content
    hash is still verified: a tampered row is persisted as CONTRADICTED and
    skipped, exactly as on every other read path.
"""
from __future__ import annotations

import re
import sqlite3
from typing import Callable, Iterator

from mark.store._sqlite.codec import _FRAGMENT_COLUMNS, _as_bytes, _FTS_TABLE, _RETRIEVABLE_STATES, _TTL_LIVE_SQL
from mark.types import MemoryFragment

# Embedding columns are replaced by NULL: decoding thousands of vectors would
# dominate the cost of a timeline and nothing downstream reads them.
_LIGHT_COLUMNS = ", ".join(
    f"NULL AS {name}" if name in ("embedding", "embedding_blob") else name
    for name in _FRAGMENT_COLUMNS
)


class MentionMixin:
    """Paged, uncapped lookup of the memories that mention a phrase."""

    _fts_enabled: bool

    def mention_candidates(
        self,
        agent_id: str,
        phrase: str,
        *,
        page_size: int = 1000,
        accept: Callable[[str], bool] | None = None,
    ) -> Iterator[list[MemoryFragment]]:
        """Yield pages of retrievable fragments that may contain *phrase*.

        Pages come in storage (rowid) order. With FTS5 the phrase is matched
        as consecutive tokens in the content column only (never in tags);
        without it every retrievable fragment is yielded and the caller does
        the matching. A phrase without word characters yields nothing.

        ``accept`` (optional) is called with each row's decrypted content
        before the row is decoded; rejected rows are skipped cheaply. This
        matters most without FTS5, where every retrievable row is a candidate.
        """
        tokens = re.findall(r"\w+", phrase.lower())
        if not tokens or page_size <= 0:
            return
        where, params = self._fragment_filters(
            agent_id, scope=None, states=_RETRIEVABLE_STATES, tier=None, session_id=None,
            session_prefix=None, tags=None, block_id=None, block_ids=None,
        )
        match = None
        if self._fts_enabled:
            match = 'content : "' + " ".join(token.replace('"', '""') for token in tokens) + '"'
        last_rowid = 0
        while True:
            rows = self._mention_page(where, params, match, last_rowid, page_size)
            if rows is None:  # FTS rejected the expression: fall back to a full scan
                match, rows = None, self._mention_page(where, params, None, last_rowid, page_size)
            if not rows:
                return
            last_rowid = int(rows[-1]["rowid"])
            page: list[MemoryFragment] = []
            tampered: list[str] = []
            for row in rows:
                if accept is not None and not accept(self.encryption.decrypt_str(_as_bytes(row["content"]))):
                    continue
                fragment, is_tampered = self._decode_fragment(dict(row))
                if is_tampered:
                    tampered.append(fragment.id)
                else:
                    page.append(fragment)
            if tampered:
                self._persist_tampered(tampered)
            if page:
                yield page
            if len(rows) < page_size:
                return

    def _mention_page(
        self,
        where: str,
        params: list[object],
        match: str | None,
        after_rowid: int,
        page_size: int,
    ) -> list[sqlite3.Row] | None:
        """One page of candidate rows after *after_rowid*; None if FTS rejects *match*."""
        sql = f"SELECT rowid, {_LIGHT_COLUMNS} FROM fragments WHERE {where}{_TTL_LIVE_SQL} AND rowid > ?"
        args: list[object] = [*params, after_rowid]
        if match is not None:
            # The FTS rowid equals the fragments rowid; the rowid bound is pushed
            # into the FTS scan too so each page reads only its own range.
            sql += f" AND rowid IN (SELECT rowid FROM {_FTS_TABLE} WHERE {_FTS_TABLE} MATCH ? AND rowid > ?)"
            args += [match, after_rowid]
        sql += " ORDER BY rowid LIMIT ?"
        args.append(page_size)
        try:
            return self._fetchall(sql, args)
        except sqlite3.OperationalError:
            if match is None:
                raise
            return None
