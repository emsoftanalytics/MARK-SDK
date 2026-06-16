# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 MARK Contributors
#
# Optional contextual compression for retrieved memory fragments.
#
# Compressors sit between retrieval/rerank and context injection. They filter
# or rewrite fragment content so only query-relevant text reaches the agent.
#
# Pipeline position:
#   vector search → graph expansion → rerank → [compressor] → context bundle
#
# All implementations satisfy the ContextualCompressor protocol.
# The default is NoopCompressor (zero overhead, no dependency).
# LLMContextualCompressor requires a developer-provided LLMProvider.
"""Contextual compression interfaces and local compressor implementations."""
from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from mark.types import MemoryFragment
    from mark.types.llm import LLMProvider


@runtime_checkable
class ContextualCompressor(Protocol):
    """Protocol for compressing retrieved memory fragments before context injection."""

    def compress(
        self,
        query: str,
        fragments: list["MemoryFragment"],
    ) -> list["MemoryFragment"]:
        """Return a filtered/rewritten subset of fragments relevant to query.

        Implementations may:
          - drop irrelevant fragments entirely
          - replace fragment content with a shorter, relevant excerpt
          - return fragments unchanged (NoopCompressor)

        The original fragment IDs, tags, and metadata are preserved even when
        content is rewritten — callers can still trace back to the full fragment.
        """
        ...


class NoopCompressor:
    """Default compressor: returns fragments unchanged, zero overhead."""

    def compress(
        self,
        query: str,
        fragments: list["MemoryFragment"],
    ) -> list["MemoryFragment"]:
        """Reduce candidate fragments to query-relevant evidence."""
        return fragments


class SimpleWindowCompressor:
    """Trims each fragment to a maximum character count.

    Cheap local compression with no LLM dependency. Useful when fragments are
    long and the agent context window is small. Trimming is a crude heuristic —
    use LLMContextualCompressor when accuracy matters.
    """

    def __init__(self, max_chars: int = 500, *, ellipsis: bool = True) -> None:
        self._max = max_chars
        self._ellipsis = ellipsis

    def compress(
        self,
        query: str,
        fragments: list["MemoryFragment"],
    ) -> list["MemoryFragment"]:
        """Reduce candidate fragments to query-relevant evidence."""
        result = []
        for f in fragments:
            if len(f.content) <= self._max:
                result.append(f)
            else:
                trimmed = f.content[: self._max]
                if self._ellipsis:
                    trimmed = trimmed.rstrip() + "…"
                result.append(f.model_copy(update={"content": trimmed}))
        return result


class LLMContextualCompressor:
    """Uses a developer-provided LLM to extract query-relevant content.

    For each fragment, the LLM is asked to extract only the parts that directly
    answer or support the query. Fragments marked IRRELEVANT are dropped.
    Content is replaced with the extracted excerpt; IDs and metadata are preserved.

    Requires a developer-supplied LLMProvider (Ollama, OpenAI-compatible endpoint,
    llama.cpp, etc.). Does NOT bundle or download any model.

    Example:
        mark = Mark.local(
            project_path=".",
            compressor=LLMContextualCompressor(llm=my_small_llm),
        )
        result = mark.memory("agent").retrieve_sync(
            "What must stay visually consistent for Elena?",
            tags=["character:elena"],
            compress=True,
        )
    """

    _PROMPT = (
        "Query: {query}\n\n"
        "Passage: {passage}\n\n"
        "Extract only the parts of the passage that are directly relevant to the query.\n"
        "If nothing in the passage is relevant, respond with exactly: IRRELEVANT\n"
        "Do not add explanations. Output only the extracted text or IRRELEVANT."
    )

    def __init__(self, llm: "LLMProvider") -> None:
        self._llm = llm

    def compress(
        self,
        query: str,
        fragments: list["MemoryFragment"],
    ) -> list["MemoryFragment"]:
        """Reduce candidate fragments to query-relevant evidence."""
        result = []
        for f in fragments:
            try:
                prompt   = self._PROMPT.format(query=query, passage=f.content)
                response = self._llm.complete(prompt).strip()
                if not response or response.upper() == "IRRELEVANT":
                    continue
                result.append(f.model_copy(update={"content": response}))
            except Exception:
                # Fail-open: keep the original fragment rather than losing context.
                result.append(f)
        return result
