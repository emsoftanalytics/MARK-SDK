# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 Somtochukwu Owen Ikeanyi
"""MemoryBlock: a scoped, sealable group of fragments owned by one agent."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator

from mark.types._datetime import ensure_utc
from mark.types.memory_enums import BlockStatus, MemoryScope, MemoryState
from mark.types.memory_fragment import MemoryFragment


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

    @field_validator("created_at", "sealed_at")
    @classmethod
    def _utc_datetimes(cls, v: Optional[datetime]) -> Optional[datetime]:
        """Coerce naive datetimes to UTC."""
        return ensure_utc(v)

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
