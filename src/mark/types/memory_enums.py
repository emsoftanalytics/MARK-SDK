# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 MARK Contributors
"""Lifecycle and scoping enums for MARK memory types.

``MemoryState.TRANSITIONS`` is the single source of truth for which state
changes ``MemoryFragment.transition_to`` allows.
"""
from __future__ import annotations

from enum import Enum
from typing import ClassVar, Dict, Set


class MemoryState(str, Enum):
    """
    Lifecycle of a MemoryFragment.

    RAW          → just written, not yet evaluated
    UNVERIFIED   → passed basic sanitization but not LLM-validated
    VERIFIED     → LLM or human confirmed it is accurate
    PROMOTED     → high-confidence, surfaced globally
    CONTRADICTED → conflicts with another fragment; retrieval penalised
    QUARANTINED  → flagged for review; excluded from retrieval
    """
    RAW          = "raw"
    UNVERIFIED   = "unverified"
    VERIFIED     = "verified"
    PROMOTED     = "promoted"
    CONTRADICTED = "contradicted"
    QUARANTINED  = "quarantined"

    TRANSITIONS: ClassVar[Dict["MemoryState", Set["MemoryState"]]]

    def can_transition_to(self, new_state: "MemoryState") -> bool:
        """Return True when the state transition is allowed."""
        return new_state in MemoryState.TRANSITIONS.get(self, set())


# Assigned after class creation: enum members cannot reference each other
# inside the class body, and a ClassVar keeps it out of the member list.
MemoryState.TRANSITIONS = {
    MemoryState.RAW:          {MemoryState.UNVERIFIED, MemoryState.CONTRADICTED},
    MemoryState.UNVERIFIED:   {MemoryState.VERIFIED,   MemoryState.CONTRADICTED},
    MemoryState.VERIFIED:     {MemoryState.PROMOTED,   MemoryState.CONTRADICTED, MemoryState.QUARANTINED},
    MemoryState.PROMOTED:     {MemoryState.CONTRADICTED, MemoryState.QUARANTINED},
    MemoryState.CONTRADICTED: {MemoryState.UNVERIFIED},
    MemoryState.QUARANTINED:  {MemoryState.UNVERIFIED},
}


class MemoryScope(str, Enum):
    """
    Who can see a fragment.

    SESSION  → only this conversation turn
    AGENT    → any session for this agent_id
    USER     → any agent belonging to this user
    GLOBAL   → team-wide / world-wide (requires PROMOTED state)
    """
    SESSION = "session"
    AGENT   = "agent"
    USER    = "user"
    GLOBAL  = "global"


class MemoryTier(str, Enum):
    """
    Cognitive memory tier — mirrors the human memory system.

    WORKING     short-term scratch space; expires via ttl_seconds
    EPISODIC    event-based, conversation-scoped memories
    SEMANTIC    general factual knowledge
    PROCEDURAL  how-to / skill knowledge
    GLOBAL      shared across all agents (requires scope=GLOBAL)
    """
    WORKING    = "working"
    EPISODIC   = "episodic"
    SEMANTIC   = "semantic"
    PROCEDURAL = "procedural"
    GLOBAL     = "global"


class BlockStatus(str, Enum):
    """
    Lifecycle of a MemoryBlock.

    OPEN        → accepting new fragments, nodes, and edges (default)
    SEALED      → content-hashed and chained; tamper-evident, append no more
    QUARANTINED → excluded from retrieval; members are isolated until released
    """
    OPEN        = "open"
    SEALED      = "sealed"
    QUARANTINED = "quarantined"
