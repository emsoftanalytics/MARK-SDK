# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 MARK Contributors
#
# ConsolidationManager — promotes working memory to long-term memory.
#
# Biological analogy: hippocampal consolidation — important short-term
# experiences are transferred to the cortex (semantic/episodic LTM)
# during rest periods or when importance exceeds threshold.
#
# Hook: HOOK_CONSOLIDATION can provide LLM-backed consolidation with
# quality scoring, deduplication, and semantic summarisation.
"""Working-memory consolidation into long-term memory."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, List, Optional

from mark.middlewares.governance import ConsolidationGate, GovernanceAuditLog
from mark.types import MemoryFragment, MemoryState, MemoryTier


@dataclass
class ConsolidationResult:
    """Summary of one consolidation pass (promoted, expired, deleted)."""
    promoted:  int = 0
    expired:   int = 0
    skipped:   int = 0
    blocked:   int = 0
    sanitized: int = 0


class ConsolidationManager:
    """
    Promotes high-importance working memory fragments to long-term memory.

    Promotion criteria (local SDK, basic):
    - Fragment is working memory (MemoryTier.WORKING or ttl_seconds set)
    - importance >= promote_threshold
    - Fragment has not expired

    Promoted fragments:
    - tier changed to EPISODIC (conversation-scoped LTM)
    - state advanced to VERIFIED
    - ttl_seconds cleared (becomes permanent LTM)

    Expired fragments are deleted.

    Register HOOK_CONSOLIDATION to score content quality, deduplicate similar
    fragments, or produce summarised representations before promotion.

    Media agent usage:
        After each episode:
            consolidator.run("director-agent", promote_threshold=0.7)
        → Important scene decisions → episodic LTM
        → Trivial notes → expired and deleted
        → Character/world rules already PROMOTED (exempt from decay)
    """

    def __init__(self, store: Any, agent_id: str,
                 promote_threshold: float = 0.7,
                 gate: ConsolidationGate | None = None,
                 audit_log: GovernanceAuditLog | None = None) -> None:
        self._store     = store
        self._agent_id  = agent_id
        self._threshold = promote_threshold
        self._gate      = gate or ConsolidationGate()
        self._audit_log = audit_log

    def run(self, *, promote_threshold: Optional[float] = None) -> ConsolidationResult:
        """Run a full consolidation pass. Returns ConsolidationResult."""
        threshold = promote_threshold if promote_threshold is not None else self._threshold
        result    = ConsolidationResult()
        now       = datetime.now(timezone.utc)

        fragments: List[MemoryFragment] = self._store.list_by_agent(self._agent_id)

        for frag in fragments:
            if frag.tier != MemoryTier.WORKING and frag.ttl_seconds is None:
                result.skipped += 1
                continue

            if frag.is_expired(now=now):
                self._store.delete(frag.id)
                result.expired += 1
                continue

            if frag.importance >= threshold:
                gate_result = self._gate.check(frag.content, confidence=frag.confidence)
                if self._audit_log is not None:
                    self._audit_log.record(
                        gate_result.content,
                        passed=gate_result.passed,
                        reason=gate_result.reason,
                        gate="ConsolidationGate",
                        agent_id=self._agent_id,
                    )
                if not gate_result.passed:
                    result.blocked += 1
                    continue
                if gate_result.content != frag.content:
                    result.sanitized += 1
                promoted = frag.model_copy(update={
                    "content":     gate_result.content,
                    "tier":        MemoryTier.EPISODIC,
                    "state":       MemoryState.VERIFIED,
                    "ttl_seconds": None,
                    "updated_at":  now,
                })
                self._store.store(promoted)
                result.promoted += 1
            else:
                result.skipped += 1

        return result
