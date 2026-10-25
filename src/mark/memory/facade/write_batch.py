# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 Somtochukwu Owen Ikeanyi
"""Bulk writes for MarkMemory: ``store_many_sync``.

Contract (the "one bad item never kills the batch" rule):

1. **Validate everything first.** Shared options (metadata, tags, importance,
   confidence) are checked once and raise :class:`InvalidMemoryError` for the
   whole call. Each item is checked on its own; invalid items are skipped
   with a reason (``strict=True`` raises on the first one instead).
2. **Embed before writing.** All accepted items are embedded before any row
   is written, so an embedder failure (e.g. a network timeout after its
   retries) leaves the store untouched.
3. **One transaction.** Without middleware every row of the call is written
   in a single SQLite transaction: a crash or kill mid-call stores nothing.
   (Very large calls therefore hold the write lock for their duration; send
   tens of thousands of items in chunks.)
4. **Idempotent retries.** With ``idempotency_key`` each item's id is derived
   from (agent, key, position, content hash). Items an earlier attempt
   already stored are skipped, so retrying an interrupted import — including
   several chunked calls — never duplicates memories.

With middleware registered every item goes through the full middleware path
(governance, redaction, ...) like ``store_sync``, one write per item; steps 1
and 4 still apply, steps 2 and 3 are per item.
"""
from __future__ import annotations

import hashlib
import logging
import uuid
from datetime import datetime
from typing import Any, Iterable

from pydantic import ValidationError

from mark.errors import InvalidMemoryError
from mark.memory.facade.write_guard import _with_occurred_at
from mark.memory.facade.write_validation import (
    SkippedItem,
    StoreReport,
    check_content,
    check_metadata,
    check_number,
    check_tags,
)
from mark.types import MemoryFragment, MemoryScope, MemoryState

logger = logging.getLogger("mark.memory")

# Namespace for idempotent ids (fixed forever: changing it would break retries).
_IDEMPOTENCY_NS = uuid.UUID("5b0c0e9e-6b1a-4d55-9a57-0d1b5e3c7a10")
_EMBED_BATCH = 64


def idempotent_id(agent_id: str, key: str, index: int, content: str) -> str:
    """Stable fragment id for item *index* of an idempotent bulk write."""
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
    return str(uuid.uuid5(_IDEMPOTENCY_NS, "\x1f".join((agent_id, key, str(index), digest))))


class BatchWriteMixin:
    """``store_many_sync`` (validated, embedded first, atomic, optionally idempotent)."""

    def store_many_sync(
        self,
        contents: Iterable[str],
        *,
        importance: float = 0.5,
        scope: MemoryScope = MemoryScope.AGENT,
        tags: list[str] | None = None,
        source: str | None = None,
        state: MemoryState = MemoryState.UNVERIFIED,
        confidence: float = 1.0,
        session_id: str | None = None,
        ttl_seconds: int | None = None,
        metadata: dict[str, Any] | None = None,
        occurred_at: "list[datetime | str | None] | None" = None,
        extract_facts: bool | None = None,
        idempotency_key: str | None = None,
        strict: bool = False,
    ) -> StoreReport:
        """Store many memories; returns a :class:`StoreReport` (ids in input order).

        Invalid items (empty, ``None``, non-text, bytes, invalid Unicode, over
        ``max_memory_bytes``, bad ``occurred_at``) are skipped: their slot in
        the result is ``None`` and ``report.skipped`` says why. ``strict=True``
        raises :class:`InvalidMemoryError` instead, before anything is written.

        Transactions: without middleware all accepted items are embedded
        first and then written in **one transaction** (all or nothing). With
        middleware each item is written separately through the middleware.
        ``idempotency_key`` makes retries safe: the same key, position and
        text map to the same id, and already-stored items are not written
        again (``report.existing`` counts them). ``extract_facts`` works as in
        ``store_sync``, once for the newly stored items.
        """
        if isinstance(contents, (str, bytes)):
            raise InvalidMemoryError("store_many_sync expects a list of texts; use store_sync for one",
                                     reason="not_text")
        items = list(contents)
        if occurred_at is not None and len(occurred_at) != len(items):
            raise ValueError("occurred_at must have one entry per content item")
        # Shared options: one bad value would fail every item, so fail the call.
        metadata, tags = check_metadata(metadata), check_tags(tags)
        check_number("importance", importance)
        check_number("confidence", confidence)
        valid, skipped = self._validate_items(items, occurred_at, metadata, strict)
        if skipped:
            reasons = ", ".join(sorted({item.reason for item in skipped}))
            logger.warning("store_many_sync skipped %d of %d item(s) (%s); see the returned report",
                           len(skipped), len(items), reasons)
        ids: dict[int, str] = {}
        if idempotency_key is not None:
            for index, content, _ in valid:
                ids[index] = idempotent_id(self.agent_id, str(idempotency_key), index, content)
        existing = self._already_stored(ids.values()) if ids else set()
        todo = [item for item in valid if ids.get(item[0]) not in existing]
        options = dict(importance=importance, scope=scope, tags=tags, source=source, state=state,
                       confidence=confidence, session_id=session_id, ttl_seconds=ttl_seconds)
        # Any registered middleware disables the batch fast path
        # (checked on the stack contents, not on whether a runtime is bound).
        if self._middleware is not None and self._middleware.list():
            for index, content, meta in todo:
                ids[index] = self._store_sync_routed(content, metadata=meta, occurred_at=None,
                                                     fragment_id=ids.get(index), **options)
        else:
            ids.update(self._write_batch(todo, ids, options))
        new_ids = [ids[index] for index, _, _ in todo]
        self._extract_stored(new_ids, extract_facts)
        report = StoreReport((ids.get(index) for index in range(len(items))), skipped,
                             existing=len(valid) - len(todo))
        return report

    # ── helpers ───────────────────────────────────────────────────────────────

    def _validate_items(self, items: list[Any], occurred_at: list[Any] | None, metadata: dict | None,
                        strict: bool) -> tuple[list[tuple[int, str, dict | None]], list[SkippedItem]]:
        limit = self._max_memory_bytes()
        valid: list[tuple[int, str, dict | None]] = []
        skipped: list[SkippedItem] = []
        for index, content in enumerate(items):
            try:
                text = check_content(content, limit)
                at = occurred_at[index] if occurred_at is not None else None
                try:
                    meta = _with_occurred_at(metadata, at)
                except ValueError as exc:
                    raise InvalidMemoryError(str(exc), reason="occurred_at") from None
            except InvalidMemoryError as exc:
                if strict:
                    raise InvalidMemoryError(f"item {index}: {exc}", reason=exc.reason) from None
                skipped.append(SkippedItem(index, exc.reason, str(exc)))
                continue
            valid.append((index, text, meta))
        return valid, skipped

    def _already_stored(self, ids: Iterable[str]) -> set[str]:
        existing_ids = getattr(self._store, "existing_ids", None)
        if existing_ids is not None:
            return existing_ids(ids)
        found = getattr(self._store, "get_many", lambda wanted: {})(list(ids))
        return set(found)

    def _write_batch(self, todo: list[tuple[int, str, dict | None]], ids: dict[int, str],
                     options: dict[str, Any]) -> dict[int, str]:
        """Fast path: build and gate fragments, embed all, then write them in one transaction."""
        accepted: list[MemoryFragment] = []
        rejected: list[MemoryFragment] = []
        written: dict[int, str] = {}
        for index, content, meta in todo:
            try:
                fragment, ok = self._guarded_fragment(content, metadata=meta, **options)
            except ValidationError as exc:
                raise InvalidMemoryError(f"invalid memory options: {exc.errors()[0].get('msg', exc)}",
                                         reason="invalid_option") from None
            if index in ids:
                fragment = fragment.model_copy(update={"id": ids[index]})
            written[index] = fragment.id
            (accepted if ok else rejected).append(fragment)
        # Rejected/quarantined fragments are kept for audit but never indexed.
        # Pipelines that support it store unembedded rows during an embedder
        # outage (embedded later); otherwise embed everything before writing.
        embed_for_write = getattr(self._pipeline, "embed_for_write", None)
        embedded = embed_for_write(accepted) if embed_for_write else self._embed_all(accepted)
        rows = rejected + embedded
        if not rows:
            return written
        store_many = getattr(self._store, "store_many", None)
        if store_many is not None:
            store_many(rows)  # one transaction: all or nothing
        else:  # legacy/custom store: no batch API
            for fragment in rows:
                self._store.store(fragment)
        self._index_embedded(embedded)
        return written

    def _embed_all(self, fragments: list[MemoryFragment]) -> list[MemoryFragment]:
        """Embed every fragment before anything is written (embedder errors write nothing)."""
        out: list[MemoryFragment] = []
        for start in range(0, len(fragments), _EMBED_BATCH):
            chunk = fragments[start:start + _EMBED_BATCH]
            vectors = self._embedder.embed_batch([fragment.content for fragment in chunk])
            if len(vectors) != len(chunk):
                raise RuntimeError(f"embedder returned {len(vectors)} vectors for {len(chunk)} texts")
            out.extend(f.model_copy(update={"embedding": v}) for f, v in zip(chunk, vectors))
        return out

    def _index_embedded(self, fragments: list[MemoryFragment]) -> None:
        """Add already-stored, already-embedded fragments to the vector index."""
        add = getattr(self._pipeline, "_index_add", None)
        if add is None:  # custom pipeline: let it index (it may re-embed and re-upsert)
            if fragments:
                self._pipeline.index_fragments(fragments)
            return
        for fragment in fragments:
            add(fragment.id, fragment.embedding, fragment.content, fragment.agent_id)
