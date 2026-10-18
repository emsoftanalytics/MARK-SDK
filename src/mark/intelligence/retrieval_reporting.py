"""Facts, missing anchors and gap reports for a retrieval.

A gap report answers "does memory actually cover this question?". It is
computed from calibrated query evidence (never from blended ranking
priors), the same number shown as each result's score, and is escalated
to HIGH whenever a distinctive anchor the query names (a service, ticket,
code identifier or person) appears nowhere in the agent's memory. A top
result that contains every anchor the query names is coverage, never a gap.

Facts come from the subjects the query names, the user subject for "I/my"
questions, or, failing those, the subjects of the matched memories and
facts whose attribute the query names; see :meth:`_matching_facts`.

Expects the host class to provide ``_store`` (and optionally ``_user_subject``).
"""
from __future__ import annotations

from mark.intelligence.classifier import QueryAnalysis
import re

from mark.intelligence.fact_relevance import DEFAULT_FACT_CAP, attribute_named, select_facts
from mark.intelligence.query_understanding import (
    TemporalIntent,
    _anchor_norm,
    _contains_anchor,
    anchor_present,
    detect_temporal_intent,
    gap_anchors,
    query_anchors,
)
from mark.intelligence.query_anchors import surface_form
from mark.intelligence.retrieval_policy import RetrievalPolicy, RetrievalPolicySpec
from mark.intelligence.retrieval_result import RetrievalResult
from mark.types import GapReport, GapSeverity, MemoryFragment

_FIRST_PERSON_RE = re.compile(r"\b(?:i|me|my|mine|myself)\b", re.I)
# Subjects that are pronouns ("My", "her") come from older extractors; matched
# against query text they would attach to every "my ..." question.
_PRONOUNS = {"i", "me", "my", "mine", "myself", "you", "your", "he", "him", "his", "she", "her",
             "it", "its", "we", "us", "our", "they", "them", "their", "this", "that"}
# Subjects inferred from matched memories: only the strongest few results count.
_SUBJECT_SOURCES = 3
# Above this many current facts, the attribute-name scan is skipped (cost).
_ATTRIBUTE_SCAN_LIMIT = 2000


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

    def _matching_facts(
        self, query: str, agent_id: str, intent: TemporalIntent, fragments: list[MemoryFragment] = (),
    ) -> list:
        """Facts the query asks about: current values, or first values for ORIGINAL queries.

        Subjects, in order of preference:

        1. the query's anchors plus any known fact subject named in it, so
           plain single-word subjects ("Alice") are found too;
        2. the user subject (``_user_subject``, default "user") for
           first-person questions ("Where do I live?");
        3. failing both, known subjects named in the top matched memories,
           and facts whose attribute the query names ("test command").

        Per subject, :func:`~mark.intelligence.fact_relevance.select_facts`
        keeps only the attributes asked about, capped at three unless the
        query asks for everything; inferred subjects (2, 3) never fall back
        to "all facts".
        """
        current = getattr(self._store, "fact_current", None)
        if current is None:
            return []
        from mark.memory.facts import _fact

        subjects_of = getattr(self._store, "fact_subjects", None)
        known = [s for s in (subjects_of(agent_id) if subjects_of is not None else [])
                 if _anchor_norm(s) not in _PRONOUNS]
        named = list(query_anchors(query))
        normalised = _anchor_norm(query)
        named += [s for s in known if s not in named and _contains_anchor(normalised, _anchor_norm(s))]
        user = getattr(self, "_user_subject", "user")
        inferred: list[str] = []
        if user and _FIRST_PERSON_RE.search(query) and user not in named:
            inferred.append(user)
        if not named and not inferred:
            for fragment in list(fragments)[:_SUBJECT_SOURCES]:
                content = _anchor_norm(fragment.content)
                inferred += [s for s in known if s not in inferred and _contains_anchor(content, _anchor_norm(s))]
        rows: list = []
        for subject in named:
            rows += select_facts(query, subject, current(agent_id, subject))
        for subject in inferred:
            rows += select_facts(query, subject, current(agent_id, subject), allow_open=False)
        if not rows and not named:
            rows = self._facts_by_attribute(query, agent_id, current)
        found = []
        for row in rows:
            if intent == TemporalIntent.ORIGINAL:
                # Show where it started and where it is now.
                history = self._store.fact_history(agent_id, row["subject"], row["attribute"])
                if history and history[0]["id"] != row["id"]:
                    found.append(_fact(history[0]))
            found.append(_fact(row))
        return found

    def _facts_by_attribute(self, query: str, agent_id: str, current) -> list:
        """Current facts whose attribute the query names, when they belong to few subjects."""
        everything = current(agent_id)
        if len(everything) > _ATTRIBUTE_SCAN_LIMIT:
            return []
        rows = attribute_named(query, everything)
        # A generic attribute ("status") held by many subjects is not an answer.
        if len({_anchor_norm(str(row["subject"])) for row in rows}) > DEFAULT_FACT_CAP:
            return []
        return rows[:DEFAULT_FACT_CAP]

    def _stale_fact_values(self, agent_id: str, facts: list) -> tuple[list[str], list[str]]:
        """(current values, earlier values) of the current facts in *facts*."""
        current_values, stale = [], []
        history_of = getattr(self._store, "fact_history", None)
        for fact in facts:
            if not getattr(fact, "is_current", False) or history_of is None:
                continue
            current_values.append(str(fact.value))
            stale += [str(row["value"]) for row in history_of(agent_id, fact.subject, fact.attribute)
                      if str(row["value"]) != str(fact.value)]
        return current_values, stale

    def _missing_anchors(
        self, query: str, agent_id: str, fragments: list[MemoryFragment], facts: list | None = None,
    ) -> list[str]:
        """Query anchors (names, ids, code identifiers) that appear nowhere in the agent's memory."""
        anchors = gap_anchors(query)
        if not anchors:
            return []
        retrieved = [_anchor_norm(fragment.content) for fragment in fragments]
        retrieved += [_anchor_norm(f"{fact.subject} {fact.value}") for fact in (facts or [])]
        phrase_exists = getattr(self._store, "phrase_exists", None)
        missing: list[str] = []
        for anchor in anchors:
            if any(anchor_present(content, anchor) for content in retrieved):
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
        top_covers_anchors: bool = False,
    ) -> GapReport:
        # Gap detection uses calibrated query evidence, never blended priors:
        # importance or graph weight can't make an irrelevant hit look like coverage.
        top = max(relevances, default=0.0)
        report = GapReport.from_score(top, len(relevances), spec.min_confidence)
        shown = [surface_form(query, anchor) for anchor in missing_anchors or ()]
        if shown and report.severity == GapSeverity.CRITICAL:
            # Nothing retrieved: say what was asked about, as the user wrote it.
            report = report.model_copy(update={
                "reason": "Not found in memory: " + ", ".join(repr(a) for a in shown) + ".", "missing": shown})
        if top_covers_anchors and not missing_anchors and report.severity != GapSeverity.CRITICAL:
            # The best result names everything the question is about (an
            # exact identifier or name match): coverage, whatever the wording.
            report = GapReport()
        if missing_anchors and report.severity != GapSeverity.CRITICAL:
            # Matching common words ("port", "billing") must not hide that the
            # thing actually asked about ("titan-billing") was never seen.
            report = GapReport(
                severity=GapSeverity.HIGH,
                reason="Not found in memory: " + ", ".join(repr(a) for a in shown) + ".",
                suggested_action="web_search",
                missing_coverage=round(len(missing_anchors) / max(1, len(gap_anchors(query))), 3),
                confidence=min(report.confidence, 0.25),
                missing=shown,
            )
        if report.should_search:
            keywords = getattr(analysis, "keywords", None)
            hint = " ".join(keywords[:5]) if keywords else query
            report = report.model_copy(update={"search_query_hint": hint})
        return report
