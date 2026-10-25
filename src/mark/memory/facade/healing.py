# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 Somtochukwu Owen Ikeanyi
"""Gap healing for MarkMemory: explicit external lookup and evidence storage.

Used by ``retrieve_sync(heal_gaps=True)`` when local memory cannot answer a
query. The lookup is opt-in twice over: the caller must ask for healing, and
the implicit default web skill additionally needs a ``"web"`` permission
grant on the runtime. Everything fetched is written through the normal
:meth:`store_sync` path, so governance and redaction still apply.
"""
from __future__ import annotations

import logging

from mark.intelligence.retrieval import RetrievalResult
from mark.memory.facade.write import WriteMixin
from mark.middlewares.skills.base import Skill
from mark.types import MemoryFragment, MemoryScope, MemoryState

# Keep the historical logger name so existing log filters keep matching.
logger = logging.getLogger("mark.memory.mark_memory")


class HealingMixin(WriteMixin):
    """External evidence acquisition for retrieval gaps."""

    def _lookup_and_store_evidence(
        self,
        query: str,
        result: RetrievalResult,
        *,
        session_id: str | None,
        web_skill: Skill | None,
    ) -> bool:
        """Run an explicit local web lookup and persist attributed evidence.

        Local lookup is intentionally basic. Smart browser automation belongs
        to a registered plugin where an LLM, source policy, and observability can guide it.

        Returns True only if at least one fragment was stored *and* survived
        governance (i.e. was not quarantined), meaning a retry can succeed.
        """
        skill = web_skill
        if skill is None:
            # The implicit default web skill needs an explicit "web" grant.
            # Passing a web_skill object (directly or via GapHealingMiddleware)
            # is itself the explicit opt-in.
            if not self._web_permission_granted():
                logger.info("gap healing skipped: 'web' permission not granted")
                return False
            try:
                from mark.middlewares.skills.web_search import WebSearchSkill
                skill = WebSearchSkill()
            except Exception as exc:
                logger.warning("gap healing web skill unavailable: %s: %s", type(exc).__name__, exc)
                return False

        search_query = result.gap_report.search_query_hint or query
        lookup = skill.run_sync(
            {"query": search_query, "mode": "search"},
            {"agent_id": self.agent_id},
        )
        if not lookup.success or not lookup.fragments:
            return False

        stored = False
        for fragment in lookup.fragments:
            if not isinstance(fragment, MemoryFragment):
                continue
            # External content is tagged and capped (importance/confidence
            # <= 0.5) so it never outranks the agent's own verified memory.
            metadata = dict(fragment.metadata)
            metadata.update(
                {
                    "mark_acquired_via": "gap_healing",
                    "mark_original_query": query,
                    "mark_search_query": search_query,
                    "mark_trust": "external",
                    "source_url": fragment.source,
                }
            )
            if fragment.attribution is not None:
                metadata["attribution"] = fragment.attribution.model_dump(mode="json")
            tags = sorted(set([*fragment.tags, "external_evidence", "gap_healing", "source:web"]))
            # Route through the normal write path so governance, redaction, and
            # every other middleware see externally acquired content.
            fragment_id = self.store_sync(
                fragment.content,
                importance=min(fragment.importance, 0.5),
                scope=MemoryScope.AGENT,
                tags=tags,
                source="web",
                state=MemoryState.UNVERIFIED,
                confidence=min(fragment.confidence, 0.5),
                session_id=session_id or fragment.session_id,
                metadata=metadata,
            )
            written = self._store.get(fragment_id)
            if written is not None and written.state != MemoryState.QUARANTINED:
                stored = True
        return stored

    def _web_permission_granted(self) -> bool:
        """True when the runtime granted the ``web`` (or ``*``) permission.

        Prefers the skill registry's grant set; falls back to a plain
        ``granted_permissions`` attribute on the runtime.
        """
        registry = getattr(self._runtime, "skills", None)
        granted = getattr(registry, "_granted_permissions", None)
        if granted is None:
            granted = set(getattr(self._runtime, "granted_permissions", ()) or ())
        return "web" in granted or "*" in granted
