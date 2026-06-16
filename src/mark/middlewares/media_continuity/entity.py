# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 MARK Contributors
#
# Scoped memory helpers for named entities in long-running agents.
#
# "Character", "Object", "Location" are media production terms, but the
# underlying pattern is general: any long-running project that tracks named
# concepts across many sessions benefits from scoped helpers.
#
#   Media:       character:elena, object:red-scarf, location:north-warehouse
#   Research:    entity:watson, object:spectrometer, location:lab-3b
#   Engineering: entity:service-a, object:database, location:us-east-1
#   Game:        entity:player, object:sword, location:dungeon-level-2
"""Character, object, and location memory helpers over the agent graph."""
from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mark.intelligence import RetrievalResult
    from mark.memory.mark_memory import MarkMemory
    from mark.types import MemoryFragment, MemoryNode
    from mark.types.graph import NodeType


def _slug(label: str) -> str:
    return re.sub(r'[\s_]+', '-', label.strip().lower())


class _ScopedEntityMemory:
    """Base helper for memories scoped to a named entity.

    Writes are tagged with the entity tag (e.g. "character:elena") so
    they can be retrieved efficiently without searching all agent memory.
    """

    def __init__(
        self,
        label:       str,
        entity_type: str,
        mark_memory: "MarkMemory",
    ) -> None:
        self._label      = label
        self._type       = entity_type
        self._tag        = f"{entity_type}:{_slug(label)}"
        self._mem        = mark_memory

    @property
    def label(self) -> str:
        """Return the display label."""
        return self._label

    @property
    def entity_type(self) -> str:
        """Return the entity's node type."""
        return self._type

    @property
    def tag(self) -> str:
        """Auto-generated tag: "character:elena", "object:red-scarf", etc."""
        return self._tag

    def remember(
        self,
        fact:       str,
        *,
        importance: float = 0.8,
        tags:       list[str] | None = None,
        source:     str | None = None,
        metadata:   dict[str, Any] | None = None,
    ) -> str:
        """Write a fact about this entity. Returns fragment_id."""
        return self._mem.store_sync(
            fact,
            importance = importance,
            tags       = [self._tag, *(tags or [])],
            source     = source or self._type,
            metadata   = metadata or {},
        )

    def recall(
        self,
        query:          str,
        *,
        session_id:     str | None = None,
        session_prefix: str | None = None,
        top_k:          int = 5,
    ) -> "RetrievalResult":
        """Retrieve facts about this entity relevant to query."""
        from mark.intelligence import RetrievalPolicy
        return self._mem.retrieve_sync(
            query,
            policy         = RetrievalPolicy.BALANCED,
            session_id     = session_id,
            session_prefix = session_prefix,
            tags           = [self._tag],
        )

    def facts(self, *, limit: int = 100) -> list["MemoryFragment"]:
        """Return all stored facts for this entity (un-ranked)."""
        return [
            f for f in self._mem.list(limit=limit)
            if self._tag in f.tags
        ]

    def node(self) -> "MemoryNode":
        """Get or create the graph node for this entity."""
        return self._mem.node(self._label, self._type)


class CharacterMemory(_ScopedEntityMemory):
    """Scoped memory for a named person or agent.

    Works for any named actor in a long-running project: a film character,
    a research subject, a user in a multi-session app, a game player.

    Usage::

        elena = mark.runtime.memory("agent").character("Elena")
        elena.remember("Elena is left-handed.")
        results = elena.recall("What are Elena's skills?")
    """

    def __init__(self, name: str, mark_memory: "MarkMemory") -> None:
        super().__init__(name, "character", mark_memory)


class ObjectMemory(_ScopedEntityMemory):
    """Scoped memory for a named object, tool, or artifact.

    Works for any significant tracked item: a prop in a story, a piece of
    equipment in a lab, a database in a system, a weapon in a game.

    Usage::

        scarf = mark.runtime.memory("agent").object("Red Scarf")
        scarf.remember("The red scarf belonged to Elena's mother.")
    """

    def __init__(self, name: str, mark_memory: "MarkMemory") -> None:
        super().__init__(name, "object", mark_memory)


class LocationMemory(_ScopedEntityMemory):
    """Scoped memory for a named place or spatial context.

    Works for any significant place: a film location, a research facility,
    a geographic region, or a game zone.

    Usage::

        warehouse = mark.runtime.memory("agent").location("North Warehouse")
        warehouse.remember("The North Warehouse is abandoned since Episode 2.")
    """

    def __init__(self, name: str, mark_memory: "MarkMemory") -> None:
        super().__init__(name, "location", mark_memory)
