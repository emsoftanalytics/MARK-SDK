"""Facts, missing anchors and gap reports for a retrieval.

A gap report answers "does memory actually cover this question?". It is
computed from calibrated query evidence (never from blended ranking
priors), and is escalated to HIGH whenever a distinctive anchor the query
names (a service, ticket or person) appears nowhere in the agent's memory.

Expects the host class to provide ``_store``.
"""
from __future__ import annotations

from mark.intelligence.classifier import QueryAnalysis
from mark.intelligence.fact_relevance import select_facts
from mark.intelligence.query_understanding import (
    TemporalIntent,
    _anchor_norm,
    _contains_anchor,
    detect_temporal_intent,
    query_anchors,
)
from mark.intelligence.retrieval_policy import RetrievalPolicy, RetrievalPolicySpec
from mark.intelligence.retrieval_result import RetrievalResult
from mark.types import GapReport, GapSeverity, MemoryFragment


class _ReportingMixin:
    """Bi-temporal facts, anchor coverage and calibrated gap detection."""

    def _empty_result(
        self,
        query: str,
        agent_id: str,
        policy: RetrievalPolicy,
        spec: RetrievalPolicySpec,
        analysis: QueryAnalysis,
    ) -> RetrievalResult:
        # No text memories in scope, but the agent may still hold facts.
        intent = detect_temporal_intent(query)
        facts = self._matching_facts(query, agent_id, intent)
        missing = self._missing_anchors(query, agent_id, [], facts)
        gap = (
            GapReport() if facts and not missing
            else self._gap_report([], spec, analysis, query, missing)
        )
        return RetrievalResult(
            query=query,
            fragments=[],
            scores=[],
            policy_used=policy,
            spec_used=spec,
            gap_report=gap,
            analysis=analysis,
            temporal_intent=intent.value,
            missing_anchors=missing,
            facts=facts,
        )

    def _matching_facts(self, query: str, agent_id: str, intent: TemporalIntent) -> list:
        """Facts about the query's anchors: current values, or first values for ORIGINAL queries.

        Per subject, :func:`~mark.intelligence.fact_relevance.select_facts`
        keeps only the attributes the query asks about ("port"), none when
        it asks for something no fact covers ("Which region ..."), and every
        attribute for open questions ("Tell me about atlas-core").
        """
        current = getattr(self._store, "fact_current", None)
        if current is None:
            return []
        from mark.memory.facts import _fact

        # Subjects: the query's anchors plus any known fact subject named in
        # it, so plain single-word subjects ("Alice") are found too.
        subjects = list(query_anchors(query))
        subjects_of = getattr(self._store, "fact_subjects", None)
        if subjects_of is not None:
            normalised = _anchor_norm(query)
            for subject in subjects_of(agent_id):
                if subject not in subjects and _contains_anchor(normalised, _anchor_norm(subject)):
                    subjects.append(subject)
        found = []
        for subject in subjects:
            for row in select_facts(query, subject, current(agent_id, subject)):
                if intent == TemporalIntent.ORIGINAL:
                    # Show where it started and where it is now.
                    history = self._store.fact_history(agent_id, row["subject"], row["attribute"])
                    if history and history[0]["id"] != row["id"]:
                        found.append(_fact(history[0]))
                found.append(_fact(row))
        return found

    def _missing_anchors(
        self, query: str, agent_id: str, fragments: list[MemoryFragment], facts: list | None = None,
    ) -> list[str]:
        """Query anchors (names, ids) that appear nowhere in the agent's memory."""
        anchors = query_anchors(query)
        if not anchors:
            return []
        retrieved = [_anchor_norm(fragment.content) for fragment in fragments]
        retrieved += [_anchor_norm(f"{fact.subject} {fact.value}") for fact in (facts or [])]
        phrase_exists = getattr(self._store, "phrase_exists", None)
        missing: list[str] = []
        for anchor in anchors:
            if any(_contains_anchor(content, anchor) for content in retrieved):
                continue
            # Present elsewhere in memory (just not in this top-k) is not a gap.
            if phrase_exists is not None and phrase_exists(agent_id, anchor):
                continue
            missing.append(anchor)
        return missing

    def _gap_report(
        self,
        relevances: list[float],
        spec: RetrievalPolicySpec,
        analysis: QueryAnalysis,
        query: str,
        missing_anchors: list[str] | None = None,
    ) -> GapReport:
        # Gap detection uses calibrated query evidence, never blended priors:
        # importance or graph weight can't make an irrelevant hit look like coverage.
        top = max(relevances, default=0.0)
        report = GapReport.from_score(top, len(relevances), spec.min_confidence)
        if missing_anchors and report.severity in {GapSeverity.NONE, GapSeverity.LOW, GapSeverity.MEDIUM}:
            # Matching common words ("port", "billing") must not hide that the
            # thing actually asked about ("titan-billing") was never seen.
            report = GapReport(
                severity=GapSeverity.HIGH,
                reason="Not found in memory: " + ", ".join(repr(a) for a in missing_anchors) + ".",
                suggested_action="web_search",
                missing_coverage=round(len(missing_anchors) / max(1, len(query_anchors(query))), 3),
                confidence=min(report.confidence, 0.25),
                missing=list(missing_anchors),
            )
        if report.should_search:
            keywords = getattr(analysis, "keywords", None)
            hint = " ".join(keywords[:5]) if keywords else query
            report = report.model_copy(update={"search_query_hint": hint})
        return report
