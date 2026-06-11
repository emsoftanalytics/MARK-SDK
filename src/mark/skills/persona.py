"""Persistent agent persona memory helper."""
from __future__ import annotations

from typing import TYPE_CHECKING

from mark.types import MemoryScope, MemoryState

if TYPE_CHECKING:
    from mark.memory.runtime import MarkRuntime


class AgentPersona:
    """Persistent agent identity stored as MARK memory."""

    BLOCK_NAME = "__persona__"

    def __init__(self, agent_id: str, runtime: "MarkRuntime") -> None:
        self.agent_id = agent_id
        self._memory = runtime.memory(agent_id)

    async def learn(
        self,
        fact: str,
        *,
        importance: float = 0.8,
        tags: list[str] | None = None,
    ) -> str:
        """Store a persona fact."""
        return await self._memory.store(
            fact,
            importance=importance,
            scope=MemoryScope.AGENT,
            tags=[*(tags or []), "persona"],
            state=MemoryState.VERIFIED,
        )

    def learn_sync(
        self,
        fact: str,
        *,
        importance: float = 0.8,
        tags: list[str] | None = None,
    ) -> str:
        """Synchronous variant of learn()."""
        return self._memory.store_sync(
            fact,
            importance=importance,
            scope=MemoryScope.AGENT,
            tags=[*(tags or []), "persona"],
            state=MemoryState.VERIFIED,
        )

    async def as_context(self, top_k: int = 8) -> str:
        """Render as an LLM-ready context string."""
        return self.as_context_sync(top_k=top_k)

    def as_context_sync(self, top_k: int = 8) -> str:
        """Synchronous variant of as_context()."""
        fragments = self._memory.list(states=[MemoryState.VERIFIED, MemoryState.PROMOTED])
        persona_fragments = [fragment for fragment in fragments if "persona" in fragment.tags][:top_k]
        if not persona_fragments:
            return "<persona>No persona established yet.</persona>"
        lines = ["<persona>"]
        lines.extend(f"  - {fragment.content}" for fragment in persona_fragments)
        lines.append("</persona>")
        return "\n".join(lines)

    def fragment_count(self) -> int:
        """Return the number of stored persona fragments."""
        return sum(1 for fragment in self._memory.list() if "persona" in fragment.tags)

    def clear(self) -> int:
        """Remove all stored entries."""
        fragments = [fragment for fragment in self._memory.list() if "persona" in fragment.tags]
        for fragment in fragments:
            self._memory.forget(fragment.id)
        return len(fragments)
