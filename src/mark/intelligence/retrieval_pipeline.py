"""RetrievalPipeline: the orchestrator of local hybrid retrieval.

One retrieval runs these stages (each implemented in its own module):

    scope filter + index sync      retrieval_index     (_IndexSyncMixin)
    vector + lexical, RRF fusion   retrieval_fusion    (_FusionMixin)
    graph expansion + reranking    retrieval_fusion    (GraphExpander, MemoryReranker)
    temporal ordering              temporal_order
    plasticity feedback            retrieval_feedback  (_ReinforcementMixin)
    facts, anchors, gap report     retrieval_reporting (_ReportingMixin)
    optional compression           this module

The pipeline is safe to share between threads: index synchronisation and
rebuilds are serialised by ``_sync_lock``; everything else works on
per-call state.
"""
from __future__ import annotations

import threading
from dataclasses import replace

from mark.embeddings import EmbeddingProvider, HashEmbeddingProvider
from mark.index import VectorIndex
from mark.intelligence.classifier import QueryAnalysis, QueryClassifier
from mark.intelligence.compressor import ContextualCompressor, NoopCompressor
from mark.intelligence.graph_expansion import GraphExpander
from mark.intelligence.query_expander import NoopQueryExpander, QueryExpander
from mark.intelligence.query_understanding import _content_terms
from mark.intelligence.reranker import MemoryReranker
from mark.intelligence.retrieval_feedback import _ReinforcementMixin
from mark.intelligence.retrieval_fusion import _FusionMixin
from mark.intelligence.retrieval_index import _IndexSyncMixin
from mark.intelligence.retrieval_policy import (
    _CHANNEL_DEPTH_FACTOR,
    _CHANNEL_DEPTH_MIN,
    _RETRIEVABLE_STATES,
    BUILTIN_POLICIES,
    FusionConfig,
    QueryAnalyzer,
    ReinforceMode,
    RetrievalPolicy,
    RetrievalPolicySpec,
    SessionFilter,
)
from mark.intelligence.retrieval_reporting import _ReportingMixin
from mark.intelligence.retrieval_result import RetrievalResult
from mark.intelligence.temporal_order import temporal_order
from mark.plasticity.hebbian import EdgeCoActivation, HebbianReinforcement
from mark.plugins import HOOK_POLICY_ENGINE, PluginRegistry
from mark.store.sqlite import LocalMemoryStore
from mark.types import MemoryFragment


class RetrievalPipeline(_IndexSyncMixin, _FusionMixin, _ReinforcementMixin, _ReportingMixin):
    """Local hybrid retrieval: lexical (BM25) + vector channels fused with RRF,
    graph expansion, prior-aware reranking, calibrated gap reporting."""

    EMBEDDER_META_KEY = "embedder_fingerprint"

    def __init__(
        self,
        store: LocalMemoryStore,
        *,
        index: VectorIndex | None = None,
        embedder: EmbeddingProvider | None = None,
        analyzer: QueryAnalyzer | None = None,
        expander: GraphExpander | None = None,
        reranker: MemoryReranker | None = None,
        plugins: PluginRegistry | None = None,
        compressor: ContextualCompressor | None = None,
        query_expander: QueryExpander | None = None,
        top_k_with_llm: int = 20,
        reinforce: ReinforceMode | str = ReinforceMode.RELEVANT,
        fusion: FusionConfig | None = None,
    ) -> None:
        # VectorIndex defines __len__, so an empty index is falsy: never use `or`.
        self._store = store
        self._index = index if index is not None else VectorIndex()
        self._embedder = embedder if embedder is not None else HashEmbeddingProvider()
        self._analyzer = analyzer if analyzer is not None else QueryClassifier()
        self._expander = expander if expander is not None else GraphExpander(store)
        self._reranker = reranker if reranker is not None else MemoryReranker(store)
        self._plugins  = plugins if plugins is not None else PluginRegistry()
        self._compressor: ContextualCompressor = compressor or NoopCompressor()
        self._query_expander: QueryExpander     = query_expander or NoopQueryExpander()
        self._top_k_with_llm = max(1, top_k_with_llm)
        self._reinforce = ReinforceMode(reinforce)
        self._fusion = fusion if fusion is not None else FusionConfig()
        self._hebbian = HebbianReinforcement()
        self._coact   = EdgeCoActivation()
        # Serialises index sync / rebuilds; agents whose embedder fingerprint
        # was already checked in this process are remembered.
        self._sync_lock = threading.RLock()
        self._verified_agents: set[str] = set()

    # ── Configuration ─────────────────────────────────────────────────────────

    def configure_compressor(self, compressor: ContextualCompressor) -> None:
        """Replace the active compressor at runtime (late-binding)."""
        self._compressor = compressor

    def configure_query_expander(self, expander: QueryExpander) -> None:
        """Replace the active query expander at runtime (late-binding)."""
        self._query_expander = expander

    def configure_top_k_with_llm(self, top_k: int) -> None:
        """Set the candidate pool size used when an LLM compressor is active."""
        self._top_k_with_llm = max(1, top_k)

    def configure_reinforce(self, mode: ReinforceMode | str) -> None:
        """Set how retrieval feeds back into plasticity (relevant | all | none)."""
        self._reinforce = ReinforceMode(mode)

    def configure_fusion(self, fusion: FusionConfig) -> None:
        """Set how vector and lexical evidence are fused (see FusionConfig)."""
        self._fusion = fusion

    # ── Read path ─────────────────────────────────────────────────────────────
    # (The write path, index_fragment / index_fragments, lives in _IndexSyncMixin.)

    def retrieve(
        self,
        query: str,
        agent_id: str,
        policy: RetrievalPolicy = RetrievalPolicy.BALANCED,
        *,
        session_filter: SessionFilter | None = None,
        compress: bool = False,
        expand: bool = False,
        reinforce: ReinforceMode | str | None = None,
    ) -> RetrievalResult:
        """Run the full pipeline: filter, lexical + vector search, fuse, expand, rerank."""
        return self.retrieve_many(
            [query], agent_id, policy,
            session_filter=session_filter, compress=compress, expand=expand, reinforce=reinforce,
        )[0]

    def retrieve_many(
        self,
        queries: list[str],
        agent_id: str,
        policy: RetrievalPolicy = RetrievalPolicy.BALANCED,
        *,
        session_filter: SessionFilter | None = None,
        compress: bool = False,
        expand: bool = False,
        reinforce: ReinforceMode | str | None = None,
    ) -> list[RetrievalResult]:
        """Answer several queries in one pass; results are in query order.

        Scope filtering and index synchronisation run once for the whole
        batch, so N questions cost one setup plus N searches.
        """
        policy = RetrievalPolicy(policy)
        reinforce_mode = self._reinforce if reinforce is None else ReinforceMode(reinforce)
        # Filters are applied in SQL before scoring so other agents, other
        # sessions, deleted or quarantined memories can never crowd the top-k.
        sf = session_filter or SessionFilter()
        allowed_ids = self._allowed_ids(agent_id, sf)
        if not allowed_ids:
            results = []
            for query in queries:
                analysis = self._analyzer.analyze(query)
                results.append(self._empty_result(
                    query, agent_id, policy, self._resolve_spec(policy, analysis), analysis,
                ))
            return results
        indexed_ids = self._sync_index(agent_id, allowed_ids)
        # Unfiltered whole-agent search: skip per-id filtering (one matvec, no
        # id-set intersection) when the index partition is exactly the allowed set.
        scope_ids = None if indexed_ids == allowed_ids else allowed_ids
        return [
            self._retrieve_one(
                query, agent_id, policy, allowed_ids, scope_ids,
                compress=compress, expand=expand, reinforce_mode=reinforce_mode,
            )
            for query in queries
        ]

    def _retrieve_one(
        self,
        query: str,
        agent_id: str,
        policy: RetrievalPolicy,
        allowed_ids: set[str],
        scope_ids: set[str] | None,
        *,
        compress: bool,
        expand: bool,
        reinforce_mode: ReinforceMode,
    ) -> RetrievalResult:
        """Run every stage for one query against an already-synced scope."""
        analysis = self._analyzer.analyze(query)
        spec = self._resolve_spec(policy, analysis)

        # When an LLM compressor is active and compression is requested, fetch a
        # larger candidate pool so the compressor has more to work with.
        has_llm = compress and not isinstance(self._compressor, NoopCompressor)
        effective_top_k = self._top_k_with_llm if has_llm else spec.top_k
        depth = max(_CHANNEL_DEPTH_MIN, effective_top_k * _CHANNEL_DEPTH_FACTOR)

        query_texts = (
            self._query_expander.expand(query)
            if expand and not isinstance(self._query_expander, NoopQueryExpander)
            else [query]
        )
        fused, cosine, channels, vector_norm = self._fuse_channels(
            query_texts, agent_id, allowed_ids, scope_ids, depth,
        )
        query_terms = _content_terms(query)
        fragments_by_id, relevance, seed_scores = self._score_seeds(
            query, query_terms, len(query_texts), fused, cosine, vector_norm, depth,
        )
        expanded_scores, contradiction_ids = self._expand_seeds(
            spec, seed_scores, effective_top_k, agent_id, allowed_ids,
            fragments_by_id, relevance, cosine, query_terms,
        )
        ranked = self._rank(
            query, policy, agent_id, analysis, expanded_scores,
            fragments_by_id, contradiction_ids, effective_top_k,
        )

        # Re-check state at materialisation: a plugin scorer may return ids
        # that were never fetched, or whose state changed since.
        fragments: list[MemoryFragment] = []
        scores: list[float] = []
        relevances: list[float] = []
        for result in ranked:
            fragment = fragments_by_id.get(result.fragment_id) or self._store.get(result.fragment_id)
            if fragment is None or fragment.state not in _RETRIEVABLE_STATES:
                continue
            fragments.append(fragment)
            scores.append(result.score)
            relevances.append(relevance.get(result.fragment_id, 0.0))

        fragments, scores, relevances, intent = temporal_order(query, fragments, scores, relevances)

        self._reinforce_results(fragments, relevances, spec, reinforce_mode)

        facts = self._matching_facts(query, agent_id, intent)
        missing = self._missing_anchors(query, agent_id, fragments, facts)
        gap_report = self._gap_report(relevances, spec, analysis, query, missing)

        # Optional contextual compression: filter/rewrite fragments before context injection.
        was_compressed = False
        if compress and not isinstance(self._compressor, NoopCompressor):
            compressed = self._compressor.compress(query, fragments)
            if len(compressed) != len(fragments) or any(
                c.content != f.content for c, f in zip(compressed, fragments)
            ):
                id_to_score = dict(zip([f.id for f in fragments], scores))
                id_to_rel = dict(zip([f.id for f in fragments], relevances))
                fragments = compressed
                scores = [id_to_score.get(f.id, 0.0) for f in fragments]
                relevances = [id_to_rel.get(f.id, 0.0) for f in fragments]
                was_compressed = True
                gap_report = self._gap_report(relevances, spec, analysis, query, missing)

        return RetrievalResult(
            query=query,
            fragments=fragments,
            scores=scores,
            policy_used=policy,
            spec_used=spec,
            gap_report=gap_report,
            analysis=analysis,
            was_compressed=was_compressed,
            relevance=relevances,
            channels=channels,
            temporal_intent=intent.value,
            missing_anchors=missing,
            facts=facts,
        )

    def _resolve_spec(self, policy: RetrievalPolicy, analysis: QueryAnalysis) -> RetrievalPolicySpec:
        """Concrete knobs for this query: plugin engine, explicit policy, or AUTO by depth."""
        policy_engine = self._plugins.get(HOOK_POLICY_ENGINE)
        if policy_engine:
            return policy_engine(analysis, hint=policy)
        if policy != RetrievalPolicy.AUTO:
            # Explicit policies are authoritative.
            return BUILTIN_POLICIES[policy.value]
        depth = getattr(analysis, "question_depth", 1)
        base = BUILTIN_POLICIES["deep" if depth >= 3 else "balanced" if depth == 2 else "fast"]
        top_k = getattr(analysis, "estimated_top_k", base.top_k) or base.top_k
        return replace(base, top_k=top_k)
