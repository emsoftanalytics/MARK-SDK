"""Candidate generation for the retrieval pipeline: fusion, seed scoring, expansion, ranking.

The stages, in order:

1. **Channels.** Each query text is searched in the vector index and the
   lexical (BM25) index; both channels are fused with Reciprocal Rank Fusion,
   which also selects the candidate pool.
2. **Seed scores.** The pool is scored per the fusion strategy (see
   ``fusion_scoring``): RRF rank evidence blended with calibrated relevance
   for lexical embedders, or relative cosine plus term coverage for
   semantic embedders.
3. **Graph expansion.** Optional, per policy: neighbours of the top seeds are
   pulled in with damped scores; contradicted fragments are collected.
4. **Ranking.** Non-retrievable states and duplicate contents are dropped,
   then a scoring plugin or the built-in reranker produces the final order.

Expects the host class to provide ``_store``, ``_embedder``, ``_fusion``,
``_expander``, ``_reranker``, ``_plugins`` and the search helpers of
``_IndexSyncMixin``.
"""
from __future__ import annotations

import heapq

from mark.index import VectorSearchResult
from mark.intelligence.classifier import QueryAnalysis
from mark.intelligence.fusion_scoring import (
    lexical_share,
    max_rrf,
    merge_max,
    normalised_cosines,
    rrf_seed_score,
    score_fusion_seed,
    uses_score_fusion,
)
from mark.intelligence.query_understanding import (
    TemporalIntent,
    _coverage,
    _relevance,
    detect_temporal_intent,
    query_anchors,
)
from mark.intelligence.retrieval_time import DateRange, matches_dates
from mark.intelligence.reranker import RerankResult
from mark.intelligence.retrieval_policy import (
    _RETRIEVABLE_STATES,
    RRF_K,
    RetrievalPolicy,
    RetrievalPolicySpec,
)
from mark.plugins import HOOK_SCORING
from mark.types import MemoryFragment


class _FusionMixin:
    """Channel search + RRF fusion, seed scoring, graph expansion and reranking."""

    def _fuse_channels(
        self,
        query_texts: list[str],
        agent_id: str,
        allowed_ids: set[str],
        scope_ids: set[str] | None,
        depth: int,
    ) -> tuple[dict[str, float], dict[str, float], dict[str, int], dict[str, float]]:
        """Search both channels for every query text and fuse them with RRF.

        Returns (fused RRF score, best cosine similarity, hit count per
        channel, best cosine relative to each query's top hit).
        """
        fused: dict[str, float] = {}
        cosine: dict[str, float] = {}
        vector_norm: dict[str, float] = {}
        channels = {"vector": 0, "lexical": 0}
        for q_text in query_texts:
            vector_hits = self._vector_search(q_text, agent_id, allowed_ids, scope_ids, depth)
            lexical_hits = self._lexical_search(q_text, agent_id, allowed_ids, scope_ids, depth)
            channels["vector"] += len(vector_hits)
            channels["lexical"] += len(lexical_hits)
            for rank, hit in enumerate(vector_hits, 1):
                fused[hit.fragment_id] = fused.get(hit.fragment_id, 0.0) + 1.0 / (RRF_K + rank)
                if hit.score > cosine.get(hit.fragment_id, -1.0):
                    cosine[hit.fragment_id] = hit.score
            merge_max(vector_norm, normalised_cosines(vector_hits, also=(fid for fid, _ in lexical_hits)))
            for rank, (fragment_id, _bm25) in enumerate(lexical_hits, 1):
                fused[fragment_id] = fused.get(fragment_id, 0.0) + 1.0 / (RRF_K + rank)
        return fused, cosine, channels, vector_norm

    def _score_seeds(
        self,
        query: str,
        query_terms: list[str],
        query_count: int,
        fused: dict[str, float],
        cosine: dict[str, float],
        vector_norm: dict[str, float],
        depth: int,
        date_ranges: list[DateRange] | None = None,
        extra: dict[str, MemoryFragment] | None = None,
    ) -> tuple[dict[str, MemoryFragment], dict[str, float], dict[str, float]]:
        """Fetch the strongest fused candidates and score them.

        ``date_ranges`` (dates the query names) count as one more query term,
        matched by a memory's event date or a date in its text; ``extra``
        candidates (memories from those dates) join the pool even without
        channel hits. Returns (fragments by id, calibrated relevance, seed score).
        """
        # RRF picks the pool for both strategies, so recall does not change;
        # only the scoring of the pool depends on the embedder.
        seed_ids = heapq.nlargest(depth, fused, key=fused.__getitem__)
        fragments_by_id = self._get_many(seed_ids)
        for fid, fragment in (extra or {}).items():
            if fid not in fragments_by_id:
                fragments_by_id[fid] = fragment
                seed_ids.append(fid)
        coverage = {
            fid: _coverage(query_terms, fragments_by_id[fid].content)
            for fid in seed_ids if fid in fragments_by_id
        }
        if date_ranges:
            terms = len(query_terms)
            coverage = {
                fid: (cov * terms + matches_dates(fragments_by_id[fid], date_ranges)) / (terms + 1)
                for fid, cov in coverage.items()
            }
        # Calibrated relevance (gap detection, reinforcement) is the same
        # under either strategy: max of cosine and term coverage.
        relevance = {fid: max(0.0, min(1.0, max(cosine.get(fid, 0.0), cov))) for fid, cov in coverage.items()}
        if uses_score_fusion(self._fusion, self._embedder):
            weight = lexical_share(
                self._fusion, query_anchors(query), (fragments_by_id[fid].content for fid in coverage),
            )
            seed_scores = {
                fid: score_fusion_seed(vector_norm.get(fid, 0.0), cov, weight)
                for fid, cov in coverage.items()
            }
        else:
            # Fused rank evidence is combined with absolute evidence so
            # ranking stays query-aware even when one channel is empty.
            best = max_rrf(query_count)
            seed_scores = {fid: rrf_seed_score(fused.get(fid, 0.0), best, relevance[fid]) for fid in relevance}
        return fragments_by_id, relevance, seed_scores

    def _expand_seeds(
        self,
        spec: RetrievalPolicySpec,
        seed_scores: dict[str, float],
        effective_top_k: int,
        agent_id: str,
        allowed_ids: set[str],
        fragments_by_id: dict[str, MemoryFragment],
        relevance: dict[str, float],
        cosine: dict[str, float],
        query_terms: list[str],
    ) -> tuple[dict[str, float], set[str]]:
        """Graph-expand the top seeds when the policy asks for it.

        Fragments reached only through the graph are fetched into
        ``fragments_by_id`` and scored into ``relevance`` (both updated in
        place). Returns (expanded scores, contradicted fragment ids).
        """
        contradiction_ids: set[str] = set()
        expanded_scores = seed_scores
        if spec.graph_depth > 0 and seed_scores:
            top_seeds = dict(heapq.nlargest(effective_top_k, seed_scores.items(), key=lambda kv: kv[1]))
            graph_scores = self._expander.expand(
                top_seeds, agent_id, direction=spec.expansion_direction, max_hops=spec.graph_depth,
            )
            expanded_scores = {**seed_scores}
            for fid, score in graph_scores.items():
                # The graph may reach fragments outside the caller's scope.
                if fid in allowed_ids and score > expanded_scores.get(fid, 0.0):
                    expanded_scores[fid] = score
            contradiction_ids = self._expander.contradiction_penalty_ids(list(top_seeds))
            missing = [fid for fid in expanded_scores if fid not in fragments_by_id]
            if missing:
                fragments_by_id.update(self._get_many(missing))
                for fid in missing:
                    if fid in fragments_by_id:
                        relevance[fid] = _relevance(
                            query_terms, fragments_by_id[fid].content, cosine.get(fid, 0.0),
                        )
        return expanded_scores, contradiction_ids

    def _rank(
        self,
        query: str,
        policy: RetrievalPolicy,
        agent_id: str,
        analysis: QueryAnalysis,
        expanded_scores: dict[str, float],
        fragments_by_id: dict[str, MemoryFragment],
        contradiction_ids: set[str],
        effective_top_k: int,
    ) -> list[RerankResult]:
        """Final ordering: a HOOK_SCORING plugin if registered, else the prior-aware reranker."""
        # Hash-check failures and stale index entries never surface.
        candidates = [
            VectorSearchResult(fragment_id=fid, score=score, content=fragments_by_id[fid].content)
            for fid, score in expanded_scores.items()
            if fid in fragments_by_id and fragments_by_id[fid].state in _RETRIEVABLE_STATES
        ]
        candidates = _dedupe_content(candidates)

        scorer = self._plugins.get(HOOK_SCORING)
        if scorer:
            return list(scorer(
                candidates,
                query=query,
                policy=policy,
                agent_id=agent_id,
                contradiction_ids=contradiction_ids,
            ))[:effective_top_k]
        return self._reranker.rerank(
            query, candidates, effective_top_k, agent_id, contradiction_ids,
            fragments=fragments_by_id,
            # Temporal queries ("latest deploy") give recency real weight.
            recency_weight=0.5 if getattr(analysis, "has_temporal", False) else 0.0,
            # Superseded / resolved memories answer "original" questions only.
            demote_retired=detect_temporal_intent(query) != TemporalIntent.ORIGINAL,
        )


def _dedupe_content(candidates: list[VectorSearchResult]) -> list[VectorSearchResult]:
    """Drop exact duplicate contents, keeping the best-scored copy."""
    best: dict[str, VectorSearchResult] = {}
    for candidate in candidates:
        key = " ".join(candidate.content.lower().split())
        current = best.get(key)
        if current is None or candidate.score > current.score:
            best[key] = candidate
    return list(best.values())
