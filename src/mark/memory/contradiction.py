# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 MARK Contributors
#
# ContradictionDetector — detects conflicting memory fragments.
#
# Biological analogy: the hippocampus flags conflicting memories for
# review rather than silently accepting contradictions.
#
# HOOK_CONTRADICTION_RESOLVER can arbitrate conflicts and produce a resolved,
# authoritative fragment.
"""Heuristic contradiction detection between fragments."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

from mark.types import MemoryFragment, MemoryState


@dataclass
class ContradictionReport:
    """Detected conflict between a new fragment and existing memory."""
    fragment_id:      str
    conflicts_with:   List[str]
    reason:           str = "keyword_overlap_with_negation"
    confidence:       float = 0.5


class ContradictionDetector:
    """
    Basic contradiction detector using keyword overlap and negation signals.

    Detection strategy (local SDK, heuristic):
    - Two fragments share significant keyword overlap AND
    - One contains a negation pattern ("not", "never", "no longer") relative
      to the other, OR their content has high similarity but conflicting facts

    When a contradiction is detected:
    - Both fragments are flagged via fragment.mark_contradicted(other_id)
    - Their state transitions to CONTRADICTED (excluded from retrieval)
    - A ContradictionReport is returned for developer inspection

    Register HOOK_CONTRADICTION_RESOLVER to use semantic similarity
    + LLM arbitration to determine which fragment is correct and resolves
    the conflict automatically.

    Media agent usage:
        Check before writing: if new scene contradicts established character
        facts, block the write and raise the conflict for resolution.
        Character bible facts (PROMOTED state) block all contradicting writes.
    """

    def __init__(self, store: Any, agent_id: str,
                 overlap_threshold: float = 0.4) -> None:
        self._store     = store
        self._agent_id  = agent_id
        self._threshold = overlap_threshold

    def check(self, fragment: MemoryFragment,
              candidates: Optional[List[MemoryFragment]] = None) -> List[ContradictionReport]:
        """
        Check if fragment contradicts any existing fragments.
        Returns a list of ContradictionReports (empty = no contradictions).
        """
        if candidates is None:
            candidates = self._store.list_by_agent(
                self._agent_id,
                states=[MemoryState.VERIFIED, MemoryState.PROMOTED],
            )

        reports: List[ContradictionReport] = []
        frag_words = _keywords(fragment.content)

        for existing in candidates:
            if existing.id == fragment.id:
                continue
            # PROMOTED fragments are authoritative — always check against them
            if self._is_contradiction(fragment.content, existing.content, frag_words):
                reports.append(ContradictionReport(
                    fragment_id    = fragment.id,
                    conflicts_with = [existing.id],
                    reason         = "content_conflict",
                    confidence     = 0.6,
                ))

        return reports

    def mark_conflict(self, frag_a_id: str, frag_b_id: str) -> Tuple[Optional[MemoryFragment], Optional[MemoryFragment]]:
        """
        Mark two fragments as contradicting each other.
        Returns the updated (a, b) tuple, or (None, None) if not found.
        """
        frag_a = self._store.get(frag_a_id)
        frag_b = self._store.get(frag_b_id)
        if frag_a and frag_b:
            updated_a = frag_a.mark_contradicted(frag_b_id)
            updated_b = frag_b.mark_contradicted(frag_a_id)
            self._store.store(updated_a)
            self._store.store(updated_b)
            return updated_a, updated_b
        return None, None

    def _is_contradiction(self, content_a: str, content_b: str,
                          words_a: Optional[set] = None) -> bool:
        """Heuristic: significant overlap + negation signal in either content."""
        wa = words_a or _keywords(content_a)
        wb = _keywords(content_b)
        if not wa or not wb:
            return False
        overlap = len(wa & wb) / min(len(wa), len(wb))
        if overlap < self._threshold:
            return False
        _NEGATION = {"not", "never", "no", "without", "except", "neither", "nor", "removed",
                     "deleted", "changed", "different", "incorrect", "wrong", "false"}
        has_neg_a = bool(wa & _NEGATION)
        has_neg_b = bool(wb & _NEGATION)
        return has_neg_a or has_neg_b


def _keywords(text: str) -> set:
    _STOP = {"a", "an", "the", "is", "are", "was", "were", "be", "been",
             "and", "or", "but", "in", "on", "at", "to", "for", "of", "it"}
    return {w.lower().strip(".,!?;:") for w in text.split()
            if len(w) >= 3 and w.lower() not in _STOP}
