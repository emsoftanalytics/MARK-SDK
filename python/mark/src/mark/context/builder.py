"""Builds LLM-ready context bundles from retrieved memory records."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from mark.context.window import within_char_budget


@dataclass(frozen=True)
class ContextBundle:
    """Formatted context text plus the records that produced it."""
    query: str
    memories: list[Any]
    text: str
    token_estimate: int
    scores: dict[str, float] = field(default_factory=dict)

    def as_text(self) -> str:
        """Render as a plain text string."""
        return self.text


class ContextBuilder:
    """Renders scored records into an LLM-ready context string."""
    def build(
        self,
        *,
        query: str,
        scored_records: list[Any],
        max_chars: int = 6000,
    ) -> ContextBundle:
        """Build the context bundle."""
        lines: list[str] = []
        memories: list[MemoryRecord] = []
        scores: dict[str, float] = {}

        if not scored_records:
            text = "MARK context: no relevant local memory found."
            return ContextBundle(query=query, memories=[], text=text, token_estimate=_estimate_tokens(text))

        header = "MARK context:"
        lines.append(header)
        for item in scored_records:
            record = item.record
            line = (
                f"- [{record.source}] {record.content} "
                f"(importance={record.importance:.2f}, confidence={record.confidence:.2f}, score={item.score:.3f})"
            )
            candidate = "\n".join([*lines, line])
            if not within_char_budget(candidate, max_chars=max_chars):
                break
            lines.append(line)
            memories.append(record)
            scores[record.id] = item.score

        text = "\n".join(lines)
        return ContextBundle(
            query=query,
            memories=memories,
            text=text,
            token_estimate=_estimate_tokens(text),
            scores=scores,
        )


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)
