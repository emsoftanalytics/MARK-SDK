# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 Somtochukwu Owen Ikeanyi
"""MemoryFragment: the atomic, immutable unit of memory in MARK."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator

from mark.types._datetime import ensure_utc
from mark.types.attribution import SourceAttribution
from mark.types.memory_enums import MemoryScope, MemoryState, MemoryTier


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
    # Frozen: every lifecycle change returns a new copy via model_copy().
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

    @field_validator("created_at", "updated_at", "last_accessed_at")
    @classmethod
    def _utc_datetimes(cls, v: Optional[datetime]) -> Optional[datetime]:
        """Coerce naive datetimes to UTC."""
        return ensure_utc(v)

    @property
    def kind(self) -> Optional[str]:
        """The memory's kind from its ``kind:<k>`` tag ("todo", "bug", ...), else None.

        Read-only view of the tag ``remember(kind=...)`` writes; see
        :func:`mark.kinds.kind_of` for the variant that also reads a leading
        "TODO:" label.
        """
        for tag in self.tags or ():
            if tag.startswith("kind:") and len(tag) > 5:
                return tag[5:]
        return None

    @property
    def is_working_memory(self) -> bool:
        """True if this fragment is short-term (has a TTL)."""
        return self.ttl_seconds is not None or self.tier == MemoryTier.WORKING

    def is_expired(self, *, now: Optional[datetime] = None) -> bool:
        """True if a TTL was set and has elapsed."""
        if self.ttl_seconds is None:
            return False
        # model_copy() skips validation, so coerce defensively here too.
        current = ensure_utc(now) if now is not None else datetime.now(timezone.utc)
        return (current - ensure_utc(self.created_at)).total_seconds() > self.ttl_seconds

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
