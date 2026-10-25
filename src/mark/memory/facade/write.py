# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 Somtochukwu Owen Ikeanyi
"""Write path of MarkMemory: store, store_sync (and store_many_sync via write_batch).

Routing invariant: when a middleware stack *and* a runtime are attached,
public writes go through ``middleware.run("store", ...)`` and only the
handler at the bottom of the stack calls :meth:`WriteMixin._store_sync_impl`.
The async :meth:`WriteMixin.store` simply runs :meth:`store_sync` on the
executor, so async callers get exactly the same governance/redaction path.

Input is validated before anything else runs (see ``write_validation``): bad
content, metadata, tags or options raise :class:`mark.errors.InvalidMemoryError`
— never a raw pydantic/JSON/Unicode error — and nothing is written.
"""
from __future__ import annotations

import functools
from datetime import datetime
from typing import TYPE_CHECKING, Any

from pydantic import ValidationError

from mark.errors import InvalidMemoryError
from mark.memory.facade.fact_extraction import FactExtractionMixin
from mark.memory.facade.write_batch import BatchWriteMixin
from mark.memory.facade.write_guard import GuardMixin, _with_occurred_at
from mark.memory.facade.write_validation import (
    DEFAULT_MAX_MEMORY_BYTES,
    check_content,
    check_metadata,
    check_number,
    check_tags,
)
from mark.types import MemoryScope, MemoryState

if TYPE_CHECKING:
    from mark.memory.facts import FactsView


class WriteMixin(BatchWriteMixin, FactExtractionMixin, GuardMixin):
    """Persisting new memories, singly or in batches."""

    async def store(
        self,
        content: str,
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
        extract_facts: bool | None = None,
    ) -> str:
        """Persist the given value.

        Runs the exact same path as :meth:`store_sync` (middleware stack,
        governance, redaction, consolidation gate) on the executor, so async
        callers can never bypass write-time policy.
        """
        import asyncio  # only async callers pay for it (CLI start-up stays lean)

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor,
            functools.partial(
                self.store_sync,
                content,
                importance=importance,
                scope=scope,
                tags=tags,
                source=source,
                state=state,
                confidence=confidence,
                session_id=session_id,
                ttl_seconds=ttl_seconds,
                metadata=metadata,
                extract_facts=extract_facts,
            ),
        )

    def store_sync(
        self,
        content: str,
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
        occurred_at: "datetime | str | None" = None,
        extract_facts: bool | None = None,
    ) -> str:
        """Synchronous variant of store().

        occurred_at    when the remembered event happened (datetime or ISO
                       string); stored as ``metadata["occurred_at"]`` and used by
                       temporal ("current"/"original") retrieval ordering.
        extract_facts  None: use the runtime's fact extractor (if configured);
                       True: extract (built-in patterns when none configured);
                       False: don't. Runs on the memory as stored and never
                       makes the write fail (see facade.fact_extraction).

        Raises :class:`mark.errors.InvalidMemoryError` (also a ValueError) for
        content that is not non-blank text, invalid Unicode, larger than
        ``max_memory_bytes`` (``MemoryTooLargeError``), non-JSON metadata,
        non-string tags or out-of-range importance/confidence.
        """
        content = check_content(content, self._max_memory_bytes())
        metadata, tags = check_metadata(metadata), check_tags(tags)
        check_number("importance", importance)
        check_number("confidence", confidence)
        tags = _with_label_kind(content, tags)
        fragment_id = self._store_sync_routed(
            content, importance=importance, scope=scope, tags=tags, source=source, state=state,
            confidence=confidence, session_id=session_id, ttl_seconds=ttl_seconds,
            metadata=metadata, occurred_at=occurred_at,
        )
        self._extract_stored([fragment_id], extract_facts)
        return fragment_id

    def _store_sync_routed(
        self,
        content: str,
        *,
        importance: float,
        scope: MemoryScope,
        tags: list[str] | None,
        source: str | None,
        state: MemoryState,
        confidence: float,
        session_id: str | None,
        ttl_seconds: int | None,
        metadata: dict[str, Any] | None,
        occurred_at: "datetime | str | None",
        fragment_id: str | None = None,
    ) -> str:
        """store_sync without fact extraction: middleware routing, then the guarded write.

        ``fragment_id`` (idempotent bulk writes) fixes the new memory's id; it
        is applied below the middleware, which never sees it.
        """
        # Normalise before middleware sees the payload, so middleware works
        # on the same metadata that ends up persisted.
        try:
            metadata = _with_occurred_at(metadata, occurred_at)
        except ValueError as exc:
            raise InvalidMemoryError(str(exc), reason="occurred_at") from None
        if self._middleware is not None and self._runtime is not None:
            return self._middleware.run(
                "store",
                runtime=self._runtime,
                agent_id=self.agent_id,
                payload={
                    "content": content,
                    "importance": importance,
                    "scope": scope,
                    "tags": tags,
                    "source": source,
                    "state": state,
                    "confidence": confidence,
                    "session_id": session_id,
                    "ttl_seconds": ttl_seconds,
                    "metadata": metadata,
                },
                handler=lambda ctx: self._store_sync_impl(**ctx.payload, fragment_id=fragment_id),
            )
        return self._store_sync_impl(
            content,
            importance=importance,
            scope=scope,
            tags=tags,
            source=source,
            state=state,
            confidence=confidence,
            session_id=session_id,
            ttl_seconds=ttl_seconds,
            metadata=metadata,
            fragment_id=fragment_id,
        )

    def _max_memory_bytes(self) -> int | None:
        """Per-memory size limit: the runtime's ``max_memory_bytes`` (None = unlimited)."""
        return getattr(self._runtime, "max_memory_bytes", DEFAULT_MAX_MEMORY_BYTES)

    @property
    def facts(self) -> "FactsView":
        """Bi-temporal facts for this agent: current values with full history."""
        from mark.memory.facts import FactsView

        return FactsView(self._store, self.agent_id, extraction=self._fact_service(None))

    def _store_sync_impl(
        self,
        content: str,
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
        fragment_id: str | None = None,
    ) -> str:
        """Middleware-free write: guard, then store (audit) or index (accepted)."""
        try:
            fragment, accepted = self._guarded_fragment(
                content,
                importance=importance,
                scope=scope,
                tags=tags,
                source=source,
                state=state,
                confidence=confidence,
                session_id=session_id,
                ttl_seconds=ttl_seconds,
                metadata=metadata,
            )
        except ValidationError as exc:
            # e.g. middleware blanked the content, or an invalid scope/state value.
            raise InvalidMemoryError(f"invalid memory: {exc.errors()[0].get('msg', exc)}",
                                     reason="invalid_option") from None
        if fragment_id is not None:
            fragment = fragment.model_copy(update={"id": fragment_id})
        if not accepted:
            return self._store.store(fragment)
        return self._pipeline.index_fragment(fragment).id


def _with_label_kind(content: str, tags: list[str] | None) -> list[str] | None:
    """Add ``kind:<k>`` for a leading label ("TODO: ...") unless a kind tag is already given.

    remember(), remember_many(), the CLI and MCP do the same, so a note is
    the same kind whichever entry point stored it.
    """
    from mark.kinds import KIND_TAG_PREFIX, kind_from_text, kind_tag

    if any(tag.startswith(KIND_TAG_PREFIX) for tag in tags or ()):
        return tags
    kind = kind_from_text(content)
    return [*(tags or []), kind_tag(kind)] if kind else tags
