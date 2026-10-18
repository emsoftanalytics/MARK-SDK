"""RetrievalResult: the outcome of one retrieval and its LLM-ready renderings.

Renderings live in ``retrieval_render`` (untrusted-content escaping, best
passage windowing, hard token caps). The score shown per result is its
displayed relevance: calibrated query evidence times ranking penalties,
non-increasing down the list, and the same number the gap report judges.
``outdated`` and ``conflicts`` carry the state flags the ordering stages
found (an older statement an update replaces; statements that disagree).
"""
from __future__ import annotations

import json as _json
from dataclasses import dataclass, field

from mark.intelligence.classifier import QueryAnalysis
from mark.intelligence.retrieval_policy import RetrievalPolicy, RetrievalPolicySpec
from mark.intelligence.retrieval_render import (  # noqa: F401  (re-exported)
    DEFAULT_ITEM_CHARS,
    UNTRUSTED_LABEL,
    UNTRUSTED_TAG,
    render,
)
from mark.types import GapReport, GapSeverity, MemoryFragment


@dataclass(frozen=True)
class RetrievalResult:
    """Full retrieval outcome: fragments, scores, policy, and gap report.

    ``scores`` are final ranking scores. ``relevance`` holds the calibrated
    query-evidence score (0–1) per fragment that drives gap detection.
    """
    query: str
    fragments: list[MemoryFragment]
    scores: list[float]
    policy_used: RetrievalPolicy
    spec_used: RetrievalPolicySpec
    gap_report: GapReport = field(default_factory=GapReport)
    analysis: QueryAnalysis | None = None
    escalation_path: list[str] = field(default_factory=list)
    external_lookup_used: bool = False
    was_compressed: bool = False
    relevance: list[float] = field(default_factory=list)
    channels: dict[str, int] = field(default_factory=dict)
    temporal_intent: str = "none"
    missing_anchors: list[str] = field(default_factory=list)
    facts: list = field(default_factory=list)  # list[mark.memory.facts.Fact]
    outdated: list[str] = field(default_factory=list)  # ids an update in the results replaces
    conflicts: dict[str, list[str]] = field(default_factory=dict)  # id -> ids it disagrees with

    @property
    def gap_detected(self) -> bool:
        """Return True when a memory gap was detected."""
        return self.gap_report.severity != GapSeverity.NONE

    @property
    def gap_severity(self) -> float:
        """Return the missing-coverage ratio."""
        return self.gap_report.missing_coverage

    def top(self) -> MemoryFragment | None:
        """Return the highest-ranked fragment, or None."""
        return self.fragments[0] if self.fragments else None

    def as_context(
        self,
        max_words: int = 1500,
        *,
        format: str = "xml",
        max_tokens: int | None = None,
        max_item_chars: int = DEFAULT_ITEM_CHARS,
    ) -> str:
        """Render retrieved fragments as an LLM-ready context string.

        format          "xml" (default) | "numbered" | "json"
        max_tokens      hard token budget (1 token = 4 chars, rounded up); overrides max_words
        max_item_chars  cap per result; longer memories show their best passage
        """
        return render(self, fmt=format, max_words=max_words, max_tokens=max_tokens,
                      max_item_chars=max_item_chars)

    def as_text(self, max_words: int = 1500, *, max_tokens: int | None = None,
                max_item_chars: int = DEFAULT_ITEM_CHARS) -> str:
        """Compact plain-text rendering (the "numbered" format of :meth:`as_context`)."""
        return self.as_context(max_words, format="numbered", max_tokens=max_tokens, max_item_chars=max_item_chars)

    def to_dict(self, max_words: int = 1500) -> dict:
        """JSON-ready result: the "json" format of :meth:`as_context` plus facts and the gap report."""
        payload = _json.loads(self.as_context(max_words, format="json"))
        payload = payload if isinstance(payload, dict) else {"memories": payload}
        payload.setdefault("facts", [f.as_text() if hasattr(f, "as_text") else str(f) for f in self.facts])
        payload.setdefault("gap", getattr(getattr(self.gap_report, "severity", None), "value", None))
        return payload
