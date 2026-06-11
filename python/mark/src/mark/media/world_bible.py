# SPDX-License-Identifier: MIT
# Copyright (c) 2025 MARK Contributors
#
# WorldBibleMemory — canonical facts that never decay.
#
# "World bible" comes from long-form production (film, TV, games, novels)
# where a reference document defines immutable truths: character traits,
# physical laws, historical facts, established settings. The same concept
# applies to any long-running agent project: a research agent has baseline
# facts about the domain; a legal case agent has established precedents; a
# game world agent has world-state axioms. This helper is the SDK surface
# for those immutable ground truths.
"""World-bible helper for promoted canonical facts."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mark.memory.mark_memory import MarkMemory
    from mark.types import MemoryFragment

from mark.types import MemoryState

_BIBLE_TAG = "world-bible"
_BIBLE_AGENT = "__world_bible__"

_SHORT_WORDS = frozenset({
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "is", "was", "are", "were", "be", "has", "had",
    "it", "he", "she", "they", "we", "you", "i",
})


class WorldBibleMemory:
    """Canonical facts for long-running agents.

    Facts written here are immediately PROMOTED — they never decay, are never
    pruned, and survive any consolidation cycle. They serve as ground truth
    for continuity checks in the current session.

    For any project with persistent ground-truth facts:
        - Media: character traits, physical laws, world history
        - Research: established domain facts, invariants
        - Games: world-state axioms, player stats
        - Legal: precedents, established facts of the case
        - Engineering: system invariants, architectural decisions

    Usage::

        mark.world_bible.remember("Elena is left-handed.", tags=["character:elena"])
        mark.world_bible.remember("The North Warehouse was destroyed in Episode 4.",
                                   tags=["location:north-warehouse"])

        # Check for potential conflicts before writing a scene:
        conflicts = mark.world_bible.check("Elena picks up the rifle with her right hand.")
        if conflicts:
            print("Continuity warning:", conflicts[0].content)
    """

    def __init__(self, mark_memory: "MarkMemory") -> None:
        self._mem = mark_memory

    def remember(
        self,
        fact:       str,
        *,
        tags:       list[str] | None = None,
        importance: float = 1.0,
        metadata:   dict[str, Any] | None = None,
    ) -> str:
        """Write a canonical fact and immediately promote it (never decays).

        Returns the fragment_id.
        """
        return self._mem.store_sync(
            fact,
            importance = importance,
            state      = MemoryState.PROMOTED,
            tags       = [_BIBLE_TAG, *(tags or [])],
            source     = "world-bible",
            metadata   = metadata or {},
        )

    def list_facts(
        self,
        *,
        tags: list[str] | None = None,
        limit: int = 500,
    ) -> list["MemoryFragment"]:
        """Return all canonical facts (optionally filtered by tags)."""
        facts = [
            f for f in self._mem.list(states=[MemoryState.PROMOTED], limit=limit)
            if _BIBLE_TAG in f.tags
        ]
        if tags:
            facts = [f for f in facts if all(t in f.tags for t in tags)]
        return facts

    def check(self, claim: str) -> list["MemoryFragment"]:
        """Return world-bible facts that share significant keywords with claim.

        This is a lightweight local check — keyword overlap, not semantic
        reasoning. It surfaces facts worth reviewing before writing a new
        observation. Cloud continuity guard does full semantic contradiction
        detection; this is the local pre-check.
        """
        claim_words = {
            w.lower().rstrip(".,!?;:\"'")
            for w in claim.split()
            if len(w) > 3 and w.lower() not in _SHORT_WORDS
        }
        if not claim_words:
            return []
        return [
            f for f in self.list_facts()
            if bool(
                {w.lower().rstrip(".,!?;:\"'") for w in f.content.split()
                 if len(w) > 3 and w.lower() not in _SHORT_WORDS}
                & claim_words
            )
        ]

    def forget(self, fragment_id: str) -> bool:
        """Remove a canonical fact by fragment_id."""
        return self._mem.forget(fragment_id)
