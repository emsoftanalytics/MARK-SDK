# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 MARK Contributors
"""Write path of MarkMemory: store, store_sync, store_many_sync.

Routing invariant: when a middleware stack *and* a runtime are attached,
public writes go through ``middleware.run("store", ...)`` and only the
handler at the bottom of the stack calls :meth:`WriteMixin._store_sync_impl`.
The async :meth:`WriteMixin.store` simply runs :meth:`store_sync` on the
executor, so async callers get exactly the same governance/redaction path.
"""
from __future__ import annotations

import asyncio
import functools
from datetime import datetime
from typing import TYPE_CHECKING, Any

from mark.memory.facade.write_guard import GuardMixin, _with_occurred_at
from mark.types import MemoryFragment, MemoryScope, MemoryState

if TYPE_CHECKING:
    from mark.memory.facts import FactsView


class WriteMixin(GuardMixin):
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
    ) -> str:
        """Persist the given value.

        Runs the exact same path as :meth:`store_sync` (middleware stack,
        governance, redaction, consolidation gate) on the executor, so async
        callers can never bypass write-time policy.
        """
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
    ) -> str:
        """Synchronous variant of store().

        occurred_at  when the remembered event happened (datetime or ISO
                     string); stored as ``metadata["occurred_at"]`` and used by
                     temporal ("current"/"original") retrieval ordering.
        """
        # Normalise before middleware sees the payload, so middleware works
        # on the same metadata that ends up persisted.
        metadata = _with_occurred_at(metadata, occurred_at)
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
                handler=lambda ctx: self._store_sync_impl(**ctx.payload),
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
        )

    @property
    def facts(self) -> "FactsView":
        """Bi-temporal facts for this agent: current values with full history."""
        from mark.memory.facts import FactsView

        return FactsView(self._store, self.agent_id)

    def store_many_sync(
        self,
        contents: list[str],
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
    ) -> list[str]:
        """Store many memories; returns their ids in input order.

        Without registered middleware, embeddings are computed in one batch and
        rows are written in one transaction. With middleware, every item goes
        through the full middleware path (governance, redaction, ...) exactly
        like store_sync, because batching must never skip those checks.
        """
        if occurred_at is not None and len(occurred_at) != len(contents):
            raise ValueError("occurred_at must have one entry per content item")
        when = occurred_at or [None] * len(contents)
        options = dict(
            importance=importance, scope=scope, tags=tags, source=source, state=state,
            confidence=confidence, session_id=session_id, ttl_seconds=ttl_seconds,
        )
        # Fallback: any registered middleware disables the batch fast path
        # (checked on the stack contents, not on whether a runtime is bound).
        if self._middleware is not None and self._middleware.list():
            return [
                self.store_sync(content, metadata=metadata, occurred_at=at, **options)
                for content, at in zip(contents, when)
            ]
        accepted: list[MemoryFragment] = []
        rejected: list[MemoryFragment] = []
        order: list[MemoryFragment] = []
        for content, at in zip(contents, when):
            fragment, ok = self._guarded_fragment(
                content, metadata=_with_occurred_at(metadata, at), **options,
            )
            (accepted if ok else rejected).append(fragment)
            order.append(fragment)
        # Rejected/quarantined fragments are persisted for audit but never
        # indexed; accepted ones are embedded and written in one batch.
        if rejected:
            self._store.store_many(rejected) if hasattr(self._store, "store_many") else [
                self._store.store(fragment) for fragment in rejected
            ]
        if accepted:
            self._pipeline.index_fragments(accepted)
        return [fragment.id for fragment in order]

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
    ) -> str:
        """Middleware-free write: guard, then store (audit) or index (accepted)."""
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
        if not accepted:
            return self._store.store(fragment)
        return self._pipeline.index_fragment(fragment).id
