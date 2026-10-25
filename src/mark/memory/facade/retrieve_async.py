# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 Somtochukwu Owen Ikeanyi
"""Async batch read path of MarkMemory: retrieve_many.

Kept apart from :mod:`mark.memory.facade.retrieve` only to respect the
300-line module limit. Like ``retrieve`` relative to ``retrieve_sync``, it
runs the synchronous implementation on the runtime executor, so middleware,
filters and result order are identical to ``retrieve_many_sync``.
"""
from __future__ import annotations

import asyncio

from mark.intelligence.retrieval import RetrievalPolicy, RetrievalResult
from mark.types import MemoryScope, MemoryTier


class BatchRetrieveMixin:
    """Adds ``await memory.retrieve_many([...])`` on top of RetrieveMixin."""

    async def retrieve_many(
        self,
        queries: list[str],
        policy: RetrievalPolicy = RetrievalPolicy.BALANCED,
        *,
        session_id: str | None = None,
        session_prefix: str | None = None,
        tags: list[str] | None = None,
        tier: MemoryTier | None = None,
        scope: MemoryScope | None = None,
        block_id: str | None = None,
        block_ids: list[str] | None = None,
        compress: bool = False,
        expand: bool = False,
    ) -> list[RetrievalResult]:
        """Answer several queries in one pass without blocking the event loop.

        Runs :meth:`retrieve_many_sync` on the runtime executor; results come
        back in query order.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor,  # type: ignore[attr-defined]
            lambda: self.retrieve_many_sync(  # type: ignore[attr-defined]
                queries,
                policy,
                session_id=session_id,
                session_prefix=session_prefix,
                tags=tags,
                tier=tier,
                scope=scope,
                block_id=block_id,
                block_ids=block_ids,
                compress=compress,
                expand=expand,
            ),
        )
