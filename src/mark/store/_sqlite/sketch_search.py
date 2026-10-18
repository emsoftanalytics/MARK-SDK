"""Exact identifier search for encrypted stores, from the keyed token sketches.

An encrypted store has no FTS index, so recall only had the vector
channel, and a hash or neural embedder places "uniq5" next to "uniq5911".
For query tokens that look like identifiers (they contain a digit: ids,
ticket numbers, ports, versions) this module provides the lexical channel:

1. the in-process sketches (see :mod:`.sketches`) give the rows that may
   contain the token: one bit test per row;
2. tokens matching more than ``MAX_CANDIDATES`` rows are skipped (they are
   not rare, and the vector channel ranks them fine);
3. the candidates are decrypted and kept only when the token is one of
   their words (whole-token equality, so "uniq5" never matches "uniq5911");
4. a row scores the sum of ``log(1 + N / matches)`` over its tokens.

Plain-word queries return nothing here and cost nothing. The sketches are
built once per process (decrypting each memory once): new memories are
sketched when they are written, and ``warm_token_sketches`` lets a
long-running host (``mark serve``) build them at start.
"""
from __future__ import annotations

import math

from mark.store._sqlite.codec import _RETRIEVABLE_STATES, _SQL_CHUNK, _as_bytes, _chunks
from mark.store._sqlite.sketches import TokenSketcher, sketch_tokens

MAX_CANDIDATES = 512


class SketchSearchMixin:
    """``_sketch_lexical`` (lexical channel of encrypted stores) and sketch upkeep at write time."""

    def _agent_scan_filter(self, agent_id: str) -> tuple[str, list[object]]:
        return self._fragment_filters(
            agent_id, scope=None, states=_RETRIEVABLE_STATES, tier=None, session_id=None,
            session_prefix=None, tags=None, block_id=None, block_ids=None,
        )

    def warm_token_sketches(self, agent_id: str) -> int:
        """Sketch every retrievable memory of *agent_id* now (encrypted stores); returns the row count."""
        if not getattr(self, "_encrypting", False):
            return 0
        where, params = self._agent_scan_filter(agent_id)
        self._sketch_candidates(where, params, ())
        return len(self.__dict__.get("_token_sketches", {}))

    def _sketch_lexical(
        self, agent_id: str, query: str, limit: int, include_ids: set[str] | frozenset[str] | None,
    ) -> list[tuple[str, float]]:
        """(fragment id, score) of memories containing the query's identifier tokens as whole words."""
        tokens = sorted(t for t in sketch_tokens(query) if any(ch.isdigit() for ch in t))
        if not tokens:
            return []
        where, params = self._agent_scan_filter(agent_id)
        total = max(1, len(self._sketch_candidates(where, params, ())))  # also brings sketches up to date
        scores: dict[str, float] = {}
        order: dict[str, int] = {}
        for token in tokens:
            rowids = self._sketch_candidates(where, params, [token])
            if not rowids or len(rowids) > MAX_CANDIDATES:
                continue
            matched = self._rows_with_token(rowids, token)
            weight = math.log(1.0 + total / max(1, len(matched)))
            for rowid, fragment_id in matched:
                if include_ids is None or fragment_id in include_ids:
                    scores[fragment_id] = scores.get(fragment_id, 0.0) + weight
                    order.setdefault(fragment_id, rowid)
        # Deterministic: best score first, then storage order.
        return sorted(scores.items(), key=lambda item: (-item[1], order[item[0]]))[:limit]

    def _rows_with_token(self, rowids: list[int], token: str) -> list[tuple[int, str]]:
        """(rowid, id) of the given rows whose decrypted text has *token* as a whole word."""
        found: list[tuple[int, str]] = []
        for chunk in _chunks(rowids, _SQL_CHUNK):  # type: ignore[arg-type]
            marks = ",".join("?" for _ in chunk)
            for rowid, fragment_id, content in self._fetchall(
                f"SELECT rowid, id, content FROM fragments WHERE rowid IN ({marks}) ORDER BY rowid", chunk,
            ):
                try:
                    text = self.encryption.decrypt_str(_as_bytes(content))
                except Exception:  # noqa: BLE001 - unreadable rows are never a lexical hit
                    continue
                if token in sketch_tokens(text):
                    found.append((int(rowid), str(fragment_id)))
        return found

    def _sketch_written(self, rowid: int, ciphertext: bytes, plaintext: str) -> None:
        """Keep this process's sketches current for a row just written (no decrypt later)."""
        known = self.__dict__.get("_token_sketches")
        if known is None:  # sketches not in use in this process: nothing to keep current
            return
        sketcher = self.__dict__.get("_token_sketcher")
        if sketcher is None:
            sketcher = self.__dict__["_token_sketcher"] = TokenSketcher.for_store(self.encryption.content_digest)
        # The stamp is what STAMP_SQL reads back: the ciphertext's last 16 bytes.
        known[rowid] = (bytes(ciphertext[-16:]), sketcher.sketch(plaintext))

