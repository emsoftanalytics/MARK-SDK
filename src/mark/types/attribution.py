# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 MARK Contributors
"""Source attribution and credibility types."""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import ClassVar, Dict, Optional

from pydantic import BaseModel, ConfigDict, Field, HttpUrl


class SourceCredibility(str, Enum):
    """
    Rough credibility tier for a web-sourced fragment.

    HUMAN      verified human author          (0.85)
    VERIFIED   authoritative, cross-checked   (0.90)
    AGENT      produced by another AI agent   (0.70)
    WEB        general web search result      (0.50)
    UNKNOWN    no provenance information      (0.40)
    """
    HUMAN    = "human"
    VERIFIED = "verified"
    AGENT    = "agent"
    WEB      = "web"
    UNKNOWN  = "unknown"

    TRUST_SCORES: ClassVar[Dict["SourceCredibility", float]]

    @property
    def trust_score(self) -> float:
        """Return the numeric trust score."""
        return SourceCredibility.TRUST_SCORES[self]


SourceCredibility.TRUST_SCORES = {
    SourceCredibility.VERIFIED: 0.90,
    SourceCredibility.HUMAN:    0.85,
    SourceCredibility.AGENT:    0.70,
    SourceCredibility.WEB:      0.50,
    SourceCredibility.UNKNOWN:  0.40,
}


class SourceAttribution(BaseModel):
    """
    Provenance record attached to any MemoryFragment that came from an
    external source (web search, document ingest, tool output).

    Every fragment whose content was not authored by the agent SHOULD carry a
    SourceAttribution so the LLM (and any downstream consumer) can cite it.
    Web-search fields mirror what search_adapter.verify_and_store() stored:
    url, title, snippet, via="web_search".
    """
    model_config = ConfigDict(frozen=True)

    source_url:   Optional[HttpUrl | str] = None
    source_title: Optional[str]           = None
    snippet:      Optional[str]           = None
    retrieved_at: datetime                = Field(default_factory=lambda: datetime.now(timezone.utc))
    credibility:  SourceCredibility       = SourceCredibility.UNKNOWN
    via:          str                     = "unknown"  # "web_search"|"document"|"tool"|"agent"
    licence_hint: Optional[str]           = None       # e.g. "CC-BY-SA", "fair use", "proprietary"

    @property
    def trust_score(self) -> float:
        """Return the numeric trust score."""
        return self.credibility.trust_score

    def cite(self) -> str:
        """One-line citation string suitable for LLM context injection."""
        parts: list[str] = []
        if self.source_title:
            parts.append(self.source_title)
        if self.source_url:
            parts.append(f"<{self.source_url}>")
        if self.retrieved_at:
            parts.append(f"retrieved {self.retrieved_at.strftime('%Y-%m-%d')}")
        return " · ".join(parts) if parts else "[no attribution]"
