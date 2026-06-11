"""Turn-by-turn episodic conversation memory."""
from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone

from mark.intelligence import RetrievalPolicy
from mark.memory.mark_memory import MarkMemory
from mark.types import MemoryFragment, MemoryScope, MemoryState


@dataclass(frozen=True)
class ConversationRecall:
    """Recalled conversation turns plus rendered context text."""
    query: str
    summaries: list[MemoryFragment]
    sources: list[MemoryFragment]

    def as_text(self) -> str:
        """Render as a plain text string."""
        if not self.summaries and not self.sources:
            return "No relevant MARK conversation memory found."
        lines: list[str] = []
        if self.summaries:
            lines.append("Summary memory:")
            lines.extend(f"- {fragment.content}" for fragment in self.summaries)
        if self.sources:
            lines.append("Source fragments:")
            lines.extend(f"- {fragment.content}" for fragment in self.sources)
        return "\n".join(lines)


class ConversationMemory:
    """Stores conversation as durable summary and source fragments."""

    def __init__(self, memory: MarkMemory, *, session_id: str | None = None) -> None:
        self.memory = memory
        self.session_id = session_id or str(uuid.uuid4())
        self._sequence = 0

    def observe(
        self,
        *,
        speaker: str,
        content: str,
        summary: str | None = None,
        importance: float = 0.6,
        confidence: float = 0.8,
        tags: list[str] | None = None,
        source: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> dict[str, str]:
        """Record one conversation turn."""
        text = content.strip()
        if not text:
            raise ValueError("content must not be blank")
        self._sequence += 1
        base_tags = ["conversation", f"speaker:{speaker}", *(tags or [])]
        base_metadata = {
            "session_id": self.session_id,
            "speaker": speaker,
            "sequence": self._sequence,
            "observed_at": datetime.now(timezone.utc).isoformat(),
            **(metadata or {}),
        }
        source_id = self.memory.store_sync(
            text,
            importance=max(0.2, importance - 0.2),
            scope=MemoryScope.AGENT,
            tags=[*base_tags, "source", "conversation_source"],
            source=source or f"conversation:{self.session_id}:{self._sequence}",
            state=MemoryState.UNVERIFIED,
            confidence=confidence,
            session_id=self.session_id,
            metadata={**base_metadata, "memory_layer": "source"},
        )
        summary_id = self.memory.store_sync(
            summary.strip() if summary else self._synthesize(speaker, text),
            importance=importance,
            scope=MemoryScope.AGENT,
            tags=[*base_tags, "summary", "conversation_summary"],
            source=f"derived_from:{source_id}",
            state=MemoryState.UNVERIFIED,
            confidence=confidence,
            session_id=self.session_id,
            metadata={**base_metadata, "memory_layer": "summary", "source_fragment_ids": [source_id]},
        )
        return {"summary_id": summary_id, "source_id": source_id}

    def observe_exchange(
        self,
        *,
        user: str,
        assistant: str,
        summary: str | None = None,
        importance: float = 0.7,
        tags: list[str] | None = None,
    ) -> dict[str, str]:
        """Record a user/assistant exchange."""
        user_ids = self.observe(speaker="user", content=user, importance=importance, tags=tags)
        assistant_ids = self.observe(speaker="assistant", content=assistant, importance=importance, tags=tags)
        summary_id = self.memory.store_sync(
            summary or self._synthesize_exchange(user, assistant),
            importance=min(1.0, importance + 0.1),
            scope=MemoryScope.AGENT,
            tags=["conversation", "summary", "exchange_summary", *(tags or [])],
            source=f"derived_from:{user_ids['source_id']},{assistant_ids['source_id']}",
            state=MemoryState.UNVERIFIED,
            confidence=0.85,
            session_id=self.session_id,
            metadata={
                "session_id": self.session_id,
                "memory_layer": "summary",
                "source_fragment_ids": [user_ids["source_id"], assistant_ids["source_id"]],
            },
        )
        return {
            "summary_id": summary_id,
            "user_source_id": user_ids["source_id"],
            "assistant_source_id": assistant_ids["source_id"],
        }

    def recall(self, query: str, *, detail: bool = False, top_k: int = 5) -> ConversationRecall:
        """Retrieve past turns relevant to the query."""
        result = self.memory.retrieve_sync(query, policy=RetrievalPolicy.DEEP if detail else RetrievalPolicy.FAST)
        fragments = [fragment for fragment in result.fragments if "conversation" in fragment.tags]
        summaries = [fragment for fragment in fragments if fragment.metadata.get("memory_layer") == "summary"][:top_k]
        sources = []
        if detail:
            sources = [fragment for fragment in fragments if fragment.metadata.get("memory_layer") == "source"][:top_k]
        return ConversationRecall(query=query, summaries=summaries, sources=sources)

    @staticmethod
    def _synthesize(speaker: str, content: str) -> str:
        compact = re.sub(r"\s+", " ", content).strip()
        if len(compact) > 360:
            compact = compact[:357].rstrip() + "..."
        return f"{speaker} said/discussed: {compact}"

    @staticmethod
    def _synthesize_exchange(user: str, assistant: str) -> str:
        return (
            "Conversation exchange summary: "
            f"{ConversationMemory._synthesize('user', user)}; "
            f"{ConversationMemory._synthesize('assistant', assistant)}"
        )
