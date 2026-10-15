# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 MARK Contributors
"""Read path of MarkMemory: retrieve, retrieve_sync, retrieve_many_sync.

Retrieval starts in the narrowest namespace the caller asked for (a session
or session prefix) and records each widening step in
``RetrievalResult.escalation_path``:

    session -> agent_namespace -> external_lookup -> *_retry

Escalation (``escalate_on_gap``/``heal_gaps``) only ever drops the session
dimension; every other caller filter (tags, tier, scope, block ids) and the
``compress``/``expand`` flags are preserved at every step.
"""
from __future__ import annotations

import asyncio
from dataclasses import replace

from mark.intelligence.retrieval import RetrievalPolicy, RetrievalResult, SessionFilter
from mark.memory.facade.healing import HealingMixin
from mark.middlewares.skills.base import Skill
from mark.types import MemoryScope, MemoryTier


class RetrieveMixin(HealingMixin):
    """Query-time retrieval with optional escalation and gap healing."""

    async def retrieve(
        self,
        query: str,
        policy: RetrievalPolicy = RetrievalPolicy.BALANCED,
        *,
        session_id: str | None = None,
        session_prefix: str | None = None,
        tags: list[str] | None = None,
        tier: MemoryTier | None = None,
        scope: MemoryScope | None = None,
        block_id: str | None = None,
        block_ids: list[str] | None = None,
        escalate_on_gap: bool = False,
        heal_gaps: bool = False,
        web_skill: Skill | None = None,
        compress: bool = False,
        order: str = "relevance",
    ) -> RetrievalResult:
        """Retrieve memory relevant to the query.

        Runs :meth:`retrieve_sync` (and therefore the middleware stack) on
        the runtime executor.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor,
            lambda: self.retrieve_sync(
                query,
                policy,
                session_id=session_id,
                session_prefix=session_prefix,
                tags=tags,
                tier=tier,
                scope=scope,
                block_id=block_id,
                block_ids=block_ids,
                escalate_on_gap=escalate_on_gap,
                heal_gaps=heal_gaps,
                web_skill=web_skill,
                compress=compress,
                order=order,
            ),
        )

    def retrieve_sync(
        self,
        query: str,
        policy: RetrievalPolicy = RetrievalPolicy.BALANCED,
        *,
        session_id: str | None = None,
        session_prefix: str | None = None,
        tags: list[str] | None = None,
        tier: MemoryTier | None = None,
        scope: MemoryScope | None = None,
        block_id: str | None = None,
        block_ids: list[str] | None = None,
        escalate_on_gap: bool = False,
        heal_gaps: bool = False,
        web_skill: Skill | None = None,
        compress: bool = False,
        expand: bool = False,
        order: str = "relevance",
        since: object = None,
        until: object = None,
    ) -> RetrievalResult:
        """Synchronous variant of retrieve().

        ``order="chronological"`` re-sorts the *same* results oldest first by
        event time (two-clock rule, see :mod:`mark.memory.timeline_order`);
        the default ``"relevance"`` leaves ranking untouched.
        """
        from mark.memory.timeline_order import ORDERS, apply_order

        if order not in ORDERS:  # validate before doing any work
            raise ValueError(f"order must be one of {ORDERS}, got {order!r}")
        if self._middleware is not None and self._runtime is not None:
            return apply_order(self._middleware.run(
                "retrieve",
                runtime=self._runtime,
                agent_id=self.agent_id,
                payload={
                    "query": query,
                    "policy": policy,
                    "session_id": session_id,
                    "session_prefix": session_prefix,
                    "tags": tags,
                    "tier": tier,
                    "scope": scope,
                    "block_id": block_id,
                    "block_ids": block_ids,
                    "escalate_on_gap": escalate_on_gap,
                    "heal_gaps": heal_gaps,
                    "web_skill": web_skill,
                    "compress": compress,
                    "expand": expand,
                    "since": since,
                    "until": until,
                },
                handler=lambda ctx: self._retrieve_sync_impl(**ctx.payload),
            ), order)
        return apply_order(self._retrieve_sync_impl(
            query,
            policy,
            session_id=session_id,
            session_prefix=session_prefix,
            tags=tags,
            tier=tier,
            scope=scope,
            block_id=block_id,
            block_ids=block_ids,
            escalate_on_gap=escalate_on_gap,
            heal_gaps=heal_gaps,
            web_skill=web_skill,
            compress=compress,
            expand=expand,
            since=since,
            until=until,
        ), order)

    def retrieve_many_sync(
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
        """Answer several queries in one pass (results in query order).

        Shares scope filtering and index sync across the batch. With
        registered middleware each query runs through retrieve_sync instead.
        """
        filters = dict(
            session_id=session_id, session_prefix=session_prefix, tags=tags, tier=tier,
            scope=scope, block_id=block_id, block_ids=block_ids,
        )
        if self._middleware is not None and self._middleware.list():
            return [
                self.retrieve_sync(query, policy, compress=compress, expand=expand, **filters)
                for query in queries
            ]
        path = ["session" if (session_id or session_prefix) else "agent_namespace"]
        results = self._pipeline.retrieve_many(
            queries, self.agent_id, policy,
            session_filter=SessionFilter(**filters), compress=compress, expand=expand,
        )
        return [replace(result, escalation_path=path) for result in results]

    def _retrieve_sync_impl(
        self,
        query: str,
        policy: RetrievalPolicy = RetrievalPolicy.BALANCED,
        *,
        session_id: str | None = None,
        session_prefix: str | None = None,
        tags: list[str] | None = None,
        tier: MemoryTier | None = None,
        scope: MemoryScope | None = None,
        block_id: str | None = None,
        block_ids: list[str] | None = None,
        escalate_on_gap: bool = False,
        heal_gaps: bool = False,
        web_skill: Skill | None = None,
        compress: bool = False,
        expand: bool = False,
        since: object = None,
        until: object = None,
    ) -> RetrievalResult:
        """Middleware-free retrieval with escalation and optional healing."""
        sf = SessionFilter(
            session_id=session_id, session_prefix=session_prefix,
            tags=tags, tier=tier, scope=scope,
            block_id=block_id, block_ids=block_ids, since=since, until=until,
        )
        path = ["session" if (session_id or session_prefix) else "agent_namespace"]
        result = replace(
            self._pipeline.retrieve(query, self.agent_id, policy,
                                    session_filter=sf, compress=compress, expand=expand),
            escalation_path=path,
        )

        # Step 1: widen a session-scoped miss to the whole agent namespace.
        # Without a session filter there is nothing wider to escalate to.
        should_escalate = escalate_on_gap or heal_gaps
        if (
            should_escalate
            and result.gap_report.should_search
            and (session_id or session_prefix)
        ):
            # Escalation widens only the session dimension; every other caller
            # filter (tags, tier, scope, blocks, time window) and expand stay.
            agent_filter = SessionFilter(
                tags=tags, tier=tier, scope=scope,
                block_id=block_id, block_ids=block_ids, since=since, until=until,
            )
            agent_result = replace(
                self._pipeline.retrieve(
                    query,
                    self.agent_id,
                    policy,
                    session_filter=agent_filter,
                    compress=compress,
                    expand=expand,
                ),
                escalation_path=[*path, "agent_namespace"],
            )
            if not agent_result.gap_report.should_search:
                return agent_result
            result = agent_result

        # Step 2 (heal_gaps only): fetch external evidence, then retry the
        # original filter first and the agent namespace second.
        if heal_gaps and result.gap_report.should_search:
            healed = self._lookup_and_store_evidence(
                query,
                result,
                session_id=session_id,
                web_skill=web_skill,
            )
            if healed:
                retry = replace(
                    self._pipeline.retrieve(
                        query,
                        self.agent_id,
                        policy,
                        session_filter=sf,
                        compress=compress,
                        expand=expand,
                    ),
                    escalation_path=[*result.escalation_path, "external_lookup", "session_retry"],
                    external_lookup_used=True,
                )
                if not retry.gap_report.should_search:
                    return retry

                agent_retry = replace(
                    self._pipeline.retrieve(
                        query,
                        self.agent_id,
                        policy,
                        session_filter=SessionFilter(
                            tags=tags, tier=tier, scope=scope,
                            block_id=block_id, block_ids=block_ids,
                        ),
                        compress=compress,
                        expand=expand,
                    ),
                    escalation_path=[*result.escalation_path, "external_lookup", "agent_namespace_retry"],
                    external_lookup_used=True,
                )
                return agent_retry

        return result
