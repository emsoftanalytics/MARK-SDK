# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 MARK Contributors
"""Query expansion interfaces and local expander implementations."""
from __future__ import annotations

import re
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from mark.types.llm import LLMProvider


@runtime_checkable
class QueryExpander(Protocol):
    """Generates one or more alternate phrasings of a query to improve recall."""

    def expand(self, query: str) -> list[str]:
        """Return the query plus alternate phrasings."""
        ...


class NoopQueryExpander:
    """Returns the original query unchanged (default — zero overhead)."""

    def expand(self, query: str) -> list[str]:
        """Return the query plus alternate phrasings."""
        return [query]


class KeywordQueryExpander:
    """Strips stop/question words to produce a bare-keyword variant.

    Example: "What color is Elena's scarf?" → ["What color is Elena's scarf?", "color Elena's scarf"]
    """

    _STRIP_RE = re.compile(
        r"\b(what|who|where|when|how|why|is|are|was|were|does|did|"
        r"can|could|should|would|the|a|an|in|on|at|of|for|to|do|"
        r"have|has|had|be|been|being|with|from|by)\b",
        re.IGNORECASE,
    )

    def expand(self, query: str) -> list[str]:
        """Return the query plus alternate phrasings."""
        keywords = self._STRIP_RE.sub("", query)
        keywords = re.sub(r"\s+", " ", keywords).strip().rstrip("?.,")
        if keywords and keywords.lower() != query.lower() and len(keywords) > 3:
            return [query, keywords]
        return [query]


class LLMQueryExpander:
    """Uses a developer-provided LLM to generate alternate query phrasings."""

    _PROMPT = (
        "Generate {n} alternate phrasings of this query to improve semantic memory retrieval recall.\n"
        "Return ONLY the alternates, one per line. No numbering, no explanation.\n\n"
        "Query: {query}"
    )

    def __init__(self, llm: "LLMProvider", *, n_expansions: int = 2) -> None:
        self._llm = llm
        self._n   = n_expansions

    def expand(self, query: str) -> list[str]:
        """Return the query plus alternate phrasings."""
        try:
            response   = self._llm.complete(self._PROMPT.format(n=self._n, query=query)).strip()
            alternates = [
                line.strip()
                for line in response.splitlines()
                if line.strip() and line.strip().lower() != query.lower()
            ][: self._n]
            return [query, *alternates]
        except Exception:
            return [query]
