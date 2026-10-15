"""Post-ranking stage: point-in-time ordering, facts and the gap report.

Runs on the materialised top-k of one query, in this order:

1. ``temporal_order`` — records about the asked subject, newest first for
   "now/latest/last" questions, oldest first for "original" ones;
2. ``order_by_supersession`` — a memory that updates an older one ("moved
   from Lisbon to Porto", "Correction: ...") goes ahead of it for
   present-tense questions, behind it for "originally/previously" ones.
   Skipped when the question names a date: "where did I live in 2024?"
   asks about that time, not about now;
3. facts — the subjects asked about (see ``_ReportingMixin._matching_facts``);
4. ``order_by_fact_agreement`` — memories stating a current fact value go
   ahead of memories stating only an earlier value of it;
5. missing anchors and the calibrated gap report;
6. optional contextual compression (``_maybe_compress``).

Expects the host class to provide ``_compressor`` and the ``_ReportingMixin`` methods.
"""
from __future__ import annotations

from mark.intelligence.compressor import NoopCompressor
from mark.intelligence.query_understanding import (
    TemporalIntent,
    _anchor_norm,
    anchor_present,
    query_anchors,
)
from mark.intelligence.supersession import order_by_fact_agreement, order_by_supersession
from mark.intelligence.temporal_order import temporal_order
from mark.types import GapReport, MemoryFragment


class _OrderingMixin:
    """Temporal, supersession and fact-agreement ordering plus reporting."""

    def _order_and_report(
        self,
        query: str,
        agent_id: str,
        spec,
        analysis,
        fragments: list[MemoryFragment],
        scores: list[float],
        relevances: list[float],
        *,
        names_date: bool,
    ) -> tuple[list[MemoryFragment], list[float], list[float], TemporalIntent, list, list[str], GapReport]:
        """Order the top-k for the query's point in time; return it with facts, missing anchors, gap."""
        fragments, scores, relevances, intent = temporal_order(query, fragments, scores, relevances)
        if not names_date:
            fragments, scores, relevances = order_by_supersession(query, intent, fragments, scores, relevances)
        facts = self._matching_facts(query, agent_id, intent, fragments)
        if facts and intent != TemporalIntent.ORIGINAL:
            current, stale = self._stale_fact_values(agent_id, facts)
            fragments, scores, relevances = order_by_fact_agreement(fragments, scores, relevances, current, stale)
        missing = self._missing_anchors(query, agent_id, fragments, facts)
        gap = self._gap_report(relevances, spec, analysis, query, missing, top_covers_anchors(query, fragments))
        return fragments, scores, relevances, intent, facts, missing, gap

    def _maybe_compress(self, compress, query, spec, analysis, fragments, scores, relevances, missing, gap):
        """Filter / rewrite fragments with the configured compressor, keeping scores by id.

        Returns (fragments, scores, relevances, gap report, was_compressed).
        """
        if not compress or isinstance(self._compressor, NoopCompressor):
            return fragments, scores, relevances, gap, False
        compressed = self._compressor.compress(query, fragments)
        if len(compressed) == len(fragments) and all(
            c.content == f.content for c, f in zip(compressed, fragments)
        ):
            return fragments, scores, relevances, gap, False
        id_to_score = dict(zip([f.id for f in fragments], scores))
        id_to_rel = dict(zip([f.id for f in fragments], relevances))
        scores = [id_to_score.get(f.id, 0.0) for f in compressed]
        relevances = [id_to_rel.get(f.id, 0.0) for f in compressed]
        gap = self._gap_report(relevances, spec, analysis, query, missing, top_covers_anchors(query, compressed))
        return compressed, scores, relevances, gap, True


def top_covers_anchors(query: str, fragments: list[MemoryFragment]) -> bool:
    """True when the best result contains every identifier / name the query is about."""
    anchors = query_anchors(query)
    if not anchors or not fragments:
        return False
    top = _anchor_norm(fragments[0].content)
    return all(anchor_present(top, anchor) for anchor in anchors)
