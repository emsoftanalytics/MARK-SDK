"""RetrievalResult: the outcome of one retrieval and its LLM-ready renderings.

Rendering treats memory content as untrusted: the XML format escapes every
stored string so a memory can never close or forge the context boundary.
"""
from __future__ import annotations

import json as _json
from dataclasses import dataclass, field
from xml.sax.saxutils import escape as _xml_escape
from xml.sax.saxutils import quoteattr as _xml_quoteattr

from mark.intelligence.classifier import QueryAnalysis
from mark.intelligence.retrieval_policy import RetrievalPolicy, RetrievalPolicySpec
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
    ) -> str:
        """Render retrieved fragments as an LLM-ready context string.

        format      "xml" (default) | "numbered" | "json"
        max_tokens  token budget (approx 1 token = 4 chars); overrides max_words
        """
        budget = int(max_tokens * 0.75) if max_tokens is not None else max_words
        if format == "numbered":
            return self._fmt_numbered(budget)
        if format == "json":
            return self._fmt_json(budget)
        return self._fmt_xml(budget)

    def _fmt_xml(self, max_words: int) -> str:
        # Memory content is untrusted: escape it so stored text can never close
        # the <memory_context> boundary or forge sibling tags.
        compressed_flag = " compressed=true" if self.was_compressed else ""
        lines = [f"<memory_context query={_xml_quoteattr(self.query)}{compressed_flag}>"]
        for fact in self.facts:
            lines.append(f"  [fact|{'current' if fact.is_current else 'historical'}] {_xml_escape(fact.as_text())}")
        if not self.fragments and not self.facts:
            lines.append("  [No relevant memory found]")
        words = 0
        for fragment, score in zip(self.fragments, self.scores):
            source = f" | source: {_xml_escape(fragment.source)}" if fragment.source else ""
            line = f"  [{score:.3f}|{fragment.state.value}{source}] {_xml_escape(fragment.content)}"
            words += len(line.split())
            if words > max_words:
                lines.append("  [... truncated for token budget]")
                break
            lines.append(line)
        if self.gap_report.severity not in {GapSeverity.NONE, GapSeverity.LOW}:
            lines.append(f"  <memory_gap severity={self.gap_report.severity.value}>")
            lines.append(f"  {_xml_escape(self.gap_report.reason)}")
            lines.append("  </memory_gap>")
        lines.append("</memory_context>")
        return "\n".join(lines)

    def _fmt_numbered(self, max_words: int) -> str:
        lines: list[str] = []
        words = 0
        for i, (fragment, score) in enumerate(zip(self.fragments, self.scores), 1):
            source = f", source={fragment.source}" if fragment.source else ""
            line = f"[{i}] (score={score:.3f}{source}) {fragment.content}"
            words += len(line.split())
            if words > max_words:
                lines.append("[...truncated]")
                break
            lines.append(line)
        gap = ""
        if self.gap_report.severity not in {GapSeverity.NONE, GapSeverity.LOW}:
            gap = f" | gap: {self.gap_report.severity.value}"
        compressed = " | compressed" if self.was_compressed else ""
        footer = f"--- {len(self.fragments)} result(s){gap}{compressed} ---"
        return "\n".join(lines) + ("\n" if lines else "") + footer

    def _fmt_json(self, max_words: int) -> str:
        results = []
        words = 0
        truncated = False
        for i, (fragment, score) in enumerate(zip(self.fragments, self.scores), 1):
            words += len(fragment.content.split())
            if words > max_words:
                truncated = True
                break
            results.append({
                "rank":    i,
                "score":   round(score, 4),
                "content": fragment.content,
                "source":  fragment.source,
                "state":   fragment.state.value,
            })
        payload: dict = {
            "query":      self.query,
            "compressed": self.was_compressed,
            "results":    results,
            "truncated":  truncated,
            "gap": {
                "severity": self.gap_report.severity.value,
                "reason":   self.gap_report.reason,
            },
        }
        return _json.dumps(payload, ensure_ascii=False)
