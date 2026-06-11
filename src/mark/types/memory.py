# SPDX-License-Identifier: MIT
# Copyright (c) 2025 MARK Contributors
"""Core memory types: fragments, blocks, documents, and lifecycle enums."""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, ClassVar, Dict, List, Optional, Set
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator

from mark.types.attribution import SourceAttribution


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


class MemoryFragment(BaseModel):
    """
    Atomic unit of memory in MARK.

    A fragment BELONGS TO a MemoryBlock (block_id). The block is the governance
    boundary — it owns the policy that decides which fragments are accepted,
    promoted, or expired. A fragment without a block_id is a "free fragment"
    (created directly via MarkMemory.store); it still works but has no block-level
    policy applied to it.

    Memory layer is determined by tier + ttl_seconds:
    - tier=WORKING or ttl_seconds set  → Working memory  (short-term, expires)
    - tier=EPISODIC/SEMANTIC, no ttl   → Long-term memory (persists, subject to decay)
    - scope=GLOBAL                     → Shared global memory
    """
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    content:           str
    agent_id:          str
    id:                str                         = Field(default_factory=lambda: str(uuid4()))
    block_id:          Optional[str]               = None
    scope:             MemoryScope                 = MemoryScope.AGENT
    tier:              MemoryTier                  = MemoryTier.EPISODIC
    session_id:        Optional[str]               = None
    importance:        float                       = Field(default=0.5, ge=0.0, le=1.0)
    state:             MemoryState                 = MemoryState.RAW
    confidence:        float                       = Field(default=1.0, ge=0.0, le=1.0)
    ttl_seconds:       Optional[int]               = None
    created_at:        datetime                    = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at:        datetime                    = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_accessed_at:  Optional[datetime]          = None
    tags:              List[str]                   = Field(default_factory=list)
    source:            Optional[str]               = None
    metadata:          Dict[str, Any]              = Field(default_factory=dict)
    embedding:         Optional[List[float]]       = None
    contradiction_ids: List[str]                   = Field(default_factory=list)
    attribution:       Optional[SourceAttribution] = None

    @field_validator("content")
    @classmethod
    def content_not_empty(cls, v: str) -> str:
        """Validate that the content is not blank."""
        v = v.strip()
        if not v:
            raise ValueError("content must not be blank")
        return v

    @property
    def is_working_memory(self) -> bool:
        """True if this fragment is short-term (has a TTL)."""
        return self.ttl_seconds is not None or self.tier == MemoryTier.WORKING

    def is_expired(self, *, now: Optional[datetime] = None) -> bool:
        """True if a TTL was set and has elapsed."""
        if self.ttl_seconds is None:
            return False
        current = now or datetime.now(timezone.utc)
        return (current - self.created_at).total_seconds() > self.ttl_seconds

    def transition_to(self, new_state: MemoryState) -> "MemoryFragment":
        """Return a copy of this fragment with updated state, or raise ValueError."""
        if not self.state.can_transition_to(new_state):
            allowed = sorted(s.value for s in MemoryState.TRANSITIONS.get(self.state, set()))
            raise ValueError(
                f"Invalid transition: {self.state.value} → {new_state.value}. "
                f"Allowed: {allowed}"
            )
        return self.model_copy(update={"state": new_state, "updated_at": datetime.now(timezone.utc)})

    def mark_contradicted(self, other_id: str) -> "MemoryFragment":
        """Return a copy flagged as contradicting another fragment."""
        ids = sorted(set([*self.contradiction_ids, other_id]))
        return self.model_copy(update={
            "state": MemoryState.CONTRADICTED,
            "contradiction_ids": ids,
            "updated_at": datetime.now(timezone.utc),
        })

    def touch(self) -> "MemoryFragment":
        """Return a copy with last_accessed_at updated to now."""
        return self.model_copy(update={"last_accessed_at": datetime.now(timezone.utc)})

    def summary(self) -> str:
        """Return a compact human-readable description."""
        suffix = "…" if len(self.content) > 80 else ""
        return f"[{self.state.value.upper()}|{self.scope.value}|imp={self.importance:.2f}] {self.content[:80]}{suffix}"


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


class MemoryBlock(BaseModel):
    """
    A named, scoped collection of MemoryFragments owned by one agent.

    The block is the governance boundary for a group of related fragments —
    e.g. "project-facts", "conversation-history", "web-research".
    Every fragment that enters a block has its block_id stamped by block.add().

    Blocks are also graph containers: nodes join a block through the block
    membership table and edges may be scoped to a block. A block can represent
    a topic within a session, an entire session, or a world-bible scope.

    block_type hints at the memory layer:
      "ltm"         → long-term memory block  (default; persistent, subject to decay)
      "working"     → working memory block    (fragments carry ttl_seconds; auto-expire)
      "global"      → shared global bus block (scope=GLOBAL)
      "world-bible" → promoted canonical facts for a story/project scope

    Provenance fields (managed by mark.memory.block_chain.BlockChain):
      status          OPEN while accepting writes; SEALED once hashed;
                      QUARANTINED to exclude the whole block from retrieval
      content_hash    deterministic SHA-256 over immutable member fields
      prev_block_id / prev_block_hash
                      link to the previously sealed block in this agent's
                      chain, making local memory history tamper-evident
      sealed_at       when the block was sealed
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name:            str
    agent_id:        str
    id:              str                  = Field(default_factory=lambda: str(uuid4()))
    scope:           MemoryScope          = MemoryScope.AGENT
    block_type:      str                  = "ltm"
    session_id:      Optional[str]        = None
    status:          BlockStatus          = BlockStatus.OPEN
    content_hash:    Optional[str]        = None
    prev_block_id:   Optional[str]        = None
    prev_block_hash: Optional[str]        = None
    sealed_at:       Optional[datetime]   = None
    fragments:       List[MemoryFragment] = Field(default_factory=list)
    created_at:      datetime             = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata:        Dict[str, Any]       = Field(default_factory=dict)

    @property
    def is_sealed(self) -> bool:
        """True once the block has been sealed into the provenance chain."""
        return self.status == BlockStatus.SEALED

    @property
    def is_quarantined(self) -> bool:
        """True while the block is excluded from retrieval."""
        return self.status == BlockStatus.QUARANTINED

    def add(self, fragment: MemoryFragment) -> None:
        """Add a fragment to this block, stamping its block_id."""
        if fragment.agent_id != self.agent_id:
            raise ValueError("Fragment agent_id must match block agent_id")
        self.fragments.append(fragment.model_copy(update={"block_id": self.id}))

    @property
    def working_fragments(self) -> List[MemoryFragment]:
        """Return fragments with a TTL (working memory)."""
        return [f for f in self.fragments if f.is_working_memory]

    @property
    def ltm_fragments(self) -> List[MemoryFragment]:
        """Return fragments without a TTL (long-term memory)."""
        return [f for f in self.fragments if not f.is_working_memory]

    def expired_fragments(self) -> List[MemoryFragment]:
        """Return fragments whose TTL has elapsed."""
        return [f for f in self.fragments if f.is_expired()]

    def by_state(self, state: MemoryState) -> List[MemoryFragment]:
        """Return fragments in the given state."""
        return [f for f in self.fragments if f.state == state]

    def retrievable(self) -> List[MemoryFragment]:
        """Fragments eligible for retrieval (not RAW, CONTRADICTED, or QUARANTINED)."""
        blocked = {MemoryState.RAW, MemoryState.CONTRADICTED, MemoryState.QUARANTINED}
        return [f for f in self.fragments if f.state not in blocked]

    def stats(self) -> Dict[str, int]:
        """Return fragment counts per lifecycle state."""
        counts: Dict[str, int] = {state.value: 0 for state in MemoryState}
        for f in self.fragments:
            counts[f.state.value] += 1
        return counts


class MemoryDocument(BaseModel):
    """
    A structured knowledge source that is chunked into MemoryFragments.

    The Document retains the full original content and the provenance (source URL,
    file path, etc.) so the system can always trace a Fragment back to where it
    came from. Chunking strategy is the caller's responsibility; the Document
    just records which Fragment ids came from it.

    Typical flow:
        doc = MemoryDocument(title="FastAPI docs", content=raw_text,
                             agent_id="coder", source="https://fastapi.tiangolo.com")
        chunks = chunk_document(doc)
        fragments = [MemoryFragment(content=c, agent_id=doc.agent_id) for c in chunks]
        doc.register_chunks(fragments)
    """
    title:        str
    content:      str
    agent_id:     str
    id:           str              = Field(default_factory=lambda: str(uuid4()))
    source:       Optional[str]    = None
    scope:        MemoryScope      = MemoryScope.AGENT
    fragment_ids: List[str]        = Field(default_factory=list)
    tags:         List[str]        = Field(default_factory=list)
    metadata:     Dict[str, Any]   = Field(default_factory=dict)
    created_at:   datetime         = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("title")
    @classmethod
    def title_not_empty(cls, v: str) -> str:
        """Validate that the title is not blank."""
        v = v.strip()
        if not v:
            raise ValueError("title must not be blank")
        return v

    @field_validator("content")
    @classmethod
    def content_not_empty(cls, v: str) -> str:
        """Validate that the content is not blank."""
        v = v.strip()
        if not v:
            raise ValueError("content must not be blank")
        return v

    def register_chunks(self, fragments: List[MemoryFragment]) -> None:
        """Record which fragment ids were produced from this document."""
        for f in fragments:
            if f.id not in self.fragment_ids:
                self.fragment_ids.append(f.id)

    def word_count(self) -> int:
        """Return the number of words in the content."""
        return len(self.content.split())

    def summary(self) -> str:
        """Return a compact human-readable description."""
        snippet = self.content[:100].replace("\n", " ")
        suffix = "…" if len(self.content) > 100 else ""
        return (
            f"[DOC|{self.scope.value}] {self.title!r} "
            f"({self.word_count()} words, {len(self.fragment_ids)} chunks) "
            f"← {snippet}{suffix}"
        )
