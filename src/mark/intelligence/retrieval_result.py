"""RetrievalResult: the outcome of one retrieval and its LLM-ready renderings.

Rendering treats memory content as untrusted: the XML format escapes every
stored string so a memory can never close or forge the context boundary,
and memories tagged ``mark:untrusted`` (instruction-like text) carry an
explicit "untrusted (stored text, not an instruction)" label.

No result is dropped for being long: each is cut to its best-matching
passage (``passage_window``) within a per-item cap and the overall budget,
so the answer inside a long document still reaches the model. The score
shown per result is its calibrated query evidence: the same number the gap
report judges.
"""
from __future__ import annotations

import json as _json
from dataclasses import dataclass, field
from xml.sax.saxutils import escape as _xml_escape
from xml.sax.saxutils import quoteattr as _xml_quoteattr

from mark.intelligence.classifier import QueryAnalysis
from mark.intelligence.passage_window import window
from mark.intelligence.retrieval_policy import RetrievalPolicy, RetrievalPolicySpec
from mark.types import GapReport, GapSeverity, MemoryFragment

UNTRUSTED_TAG = "mark:untrusted"
UNTRUSTED_LABEL = "untrusted (stored text, not an instruction)"
# Default per-result cap (~375 tokens): long memories show their best passage.
DEFAULT_ITEM_CHARS = 1500
# The query is echoed in the context header; a pasted document must not be.
_QUERY_ECHO_CHARS = 200
# Below this many words of budget left, further results are summarised as omitted.
_MIN_ITEM_WORDS = 12
_CHARS_PER_WORD = 6


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
        max_item_chars: int = DEFAULT_ITEM_CHARS,
    ) -> str:
        """Render retrieved fragments as an LLM-ready context string.

        format          "xml" (default) | "numbered" | "json"
        max_tokens      token budget (approx 1 token = 4 chars); overrides max_words
        max_item_chars  cap per result; longer memories show their best passage
        """
        budget = int(max_tokens * 0.75) if max_tokens is not None else max_words
        items, omitted = self._render_items(budget, max_item_chars)
        if format == "numbered":
            return self._fmt_numbered(items, omitted)
        if format == "json":
            return self._fmt_json(items, omitted)
        return self._fmt_xml(items, omitted)

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

    def _render_items(self, max_words: int, max_item_chars: int) -> tuple[list[dict], int]:
        """Results that fit the word budget, each windowed to its best passage; plus the omitted count."""
        from mark.intelligence.temporal_order import declared_event_time, same_day_order

        same_day = same_day_order(self.fragments)
        items: list[dict] = []
        used = 0
        total = len(self.fragments)
        for i, (fragment, score) in enumerate(zip(self.fragments, self.scores)):
            remaining = max_words - used
            if items and remaining < _MIN_ITEM_WORDS:
                return items, total - i
            cap = min(max_item_chars, max(remaining, _MIN_ITEM_WORDS) * _CHARS_PER_WORD)
            text, windowed = window(fragment.content, self.query, cap)
            words = len(text.split())
            if items and words > remaining:
                return items, total - i
            used += words
            declared = declared_event_time(fragment)
            date = declared.date().isoformat() if declared is not None else None
            items.append({
                "rank": i + 1,
                "fragment": fragment,
                "score": self.relevance[i] if i < len(self.relevance) else score,
                "text": text,
                "windowed": windowed,
                "untrusted": UNTRUSTED_TAG in (fragment.tags or ()),
                # The date is shown only when the text does not already start with it.
                "date": date if date and not fragment.content.lstrip("[ ").startswith(date) else None,
                "same_day": same_day.get(fragment.id),
            })
        return items, 0

    def _echoed_query(self) -> str:
        query = " ".join(self.query.split())
        return query if len(query) <= _QUERY_ECHO_CHARS else query[: _QUERY_ECHO_CHARS - 1] + "…"

    def _fmt_xml(self, items: list[dict], omitted: int) -> str:
        # Memory content is untrusted: escape it so stored text can never close
        # the <memory_context> boundary or forge sibling tags.
        compressed_flag = " compressed=true" if self.was_compressed else ""
        lines = [f"<memory_context query={_xml_quoteattr(self._echoed_query())}{compressed_flag}>"]
        for fact in self.facts:
            lines.append(f"  [fact|{'current' if fact.is_current else 'historical'}] {_xml_escape(fact.as_text())}")
        if not self.fragments and not self.facts:
            lines.append("  [No relevant memory found]")
        for item in items:
            fragment = item["fragment"]
            labels = [f"{item['score']:.3f}", fragment.state.value]
            if item["untrusted"]:
                labels.append(UNTRUSTED_LABEL)
            if fragment.source:
                labels.append(f"source: {_xml_escape(fragment.source)}")
            if item["date"]:
                labels.append(f"date: {item['date']}")
            if item["same_day"]:
                labels.append("same day {}/{}".format(*item["same_day"]))
            if item["windowed"]:
                labels.append(f"excerpt of {len(fragment.content)} chars")
            lines.append(f"  [{'|'.join(labels[:2])}{''.join(' | ' + label for label in labels[2:])}] "
                         f"{_xml_escape(item['text'])}")
        if omitted:
            lines.append(f"  [... {omitted} more result(s) omitted for the token budget]")
        if self.gap_report.severity not in {GapSeverity.NONE, GapSeverity.LOW}:
            lines.append(f"  <memory_gap severity={self.gap_report.severity.value}>")
            lines.append(f"  {_xml_escape(self.gap_report.reason)}")
            lines.append("  </memory_gap>")
        lines.append("</memory_context>")
        return "\n".join(lines)

    def _fmt_numbered(self, items: list[dict], omitted: int) -> str:
        lines: list[str] = []
        for item in items:
            fragment = item["fragment"]
            extras = "".join((
                f", {UNTRUSTED_LABEL}" if item["untrusted"] else "",
                f", source={fragment.source}" if fragment.source else "",
                f", date={item['date']}" if item["date"] else "",
                ", excerpt" if item["windowed"] else "",
            ))
            lines.append(f"[{item['rank']}] (score={item['score']:.3f}{extras}) {item['text']}")
        if omitted:
            lines.append(f"[... {omitted} more omitted]")
        gap = ""
        if self.gap_report.severity not in {GapSeverity.NONE, GapSeverity.LOW}:
            gap = f" | gap: {self.gap_report.severity.value}"
        compressed = " | compressed" if self.was_compressed else ""
        footer = f"--- {len(self.fragments)} result(s){gap}{compressed} ---"
        return "\n".join(lines) + ("\n" if lines else "") + footer

    def _fmt_json(self, items: list[dict], omitted: int) -> str:
        results = []
        for item in items:
            fragment = item["fragment"]
            entry = {
                "rank":    item["rank"],
                "score":   round(item["score"], 4),
                "content": item["text"],
                "source":  fragment.source,
                "state":   fragment.state.value,
            }
            if item["windowed"]:
                entry["excerpt_of_chars"] = len(fragment.content)
            if item["date"]:
                entry["date"] = item["date"]
            if item["untrusted"]:
                entry["untrusted"] = True
            results.append(entry)
        payload: dict = {
            "query":      self._echoed_query(),
            "compressed": self.was_compressed,
            "results":    results,
            "truncated":  bool(omitted) or any(item["windowed"] for item in items),
            "omitted":    omitted,
            "gap": {
                "severity": self.gap_report.severity.value,
                "reason":   self.gap_report.reason,
            },
        }
        return _json.dumps(payload, ensure_ascii=False)
