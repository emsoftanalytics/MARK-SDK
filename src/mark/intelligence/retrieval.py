"""Local hybrid retrieval: lexical + vector channels, rank fusion, graph expansion, rerank.

This module is the stable import surface for retrieval. The implementation
is split by responsibility:

    retrieval_policy      policies, per-policy specs, SessionFilter, constants
    retrieval_result      RetrievalResult and its xml / numbered / json renderings
    query_understanding   temporal intent, query anchors, stemmer, lexical evidence
    temporal_order        event-time ordering for "current" / "original" queries
    graph_expansion       GraphExpander (score propagation through memory edges)
    retrieval_pipeline    RetrievalPipeline (orchestration), built from mixins in
                          retrieval_index, retrieval_fusion, retrieval_feedback
                          and retrieval_reporting

Every name previously defined here, including private helpers used by other
modules and tests, is re-exported unchanged (same objects, same identities).
"""
from __future__ import annotations

import logging

# Names this module historically imported and that callers may still pull from it.
from mark.embeddings import EmbeddingProvider, HashEmbeddingProvider
from mark.index import VectorIndex, VectorSearchResult
from mark.intelligence.classifier import QueryAnalysis, QueryClassifier, _STOP_WORDS
from mark.intelligence.compressor import ContextualCompressor, NoopCompressor
from mark.intelligence.graph_expansion import GraphExpander
from mark.intelligence.query_expander import NoopQueryExpander, QueryExpander
from mark.intelligence.query_understanding import (
    _COMPOUND_RE,
    _CURRENT_RE,
    _DIGIT_TOKEN_RE,
    _FALSE_TEMPORAL_RE,
    _NAME_LEAD_WORDS,
    _NAME_RE,
    _NUMBER_WITH_UNIT_RE,
    _ORIGINAL_RE,
    _POSSESSIVE_RE,
    _TEMPORAL_WORDS,
    _TOKEN_RE,
    TemporalIntent,
    _anchor_norm,
    _contains_anchor,
    _content_stems,
    _content_terms,
    _coverage,
    _is_identifier,
    _relevance,
    _stem,
    _stem_once,
    detect_temporal_intent,
    query_anchors,
)
from mark.intelligence.reranker import MemoryReranker, RerankResult
from mark.intelligence.retrieval_fusion import _dedupe_content
from mark.intelligence.retrieval_index import _same_embedding_space
from mark.intelligence.retrieval_pipeline import RetrievalPipeline
from mark.intelligence.retrieval_policy import (
    _CHANNEL_DEPTH_FACTOR,
    _CHANNEL_DEPTH_MIN,
    _GRAPH_HOP_DECAY,
    _RETRIEVABLE_STATES,
    BUILTIN_POLICIES,
    RRF_K,
    QueryAnalyzer,
    ReinforceMode,
    RetrievalPolicy,
    RetrievalPolicySpec,
    SessionFilter,
)
from mark.intelligence.retrieval_result import RetrievalResult
from mark.intelligence.temporal_order import (
    _LEADING_DATE_RE,
    _order_band,
    _parse_when,
    _stored_at,
    declared_event_time,
    event_time,
    temporal_order,
)
from mark.plasticity.hebbian import EdgeCoActivation, HebbianReinforcement
from mark.plugins import HOOK_POLICY_ENGINE, HOOK_SCORING, PluginRegistry
from mark.store.sqlite import LocalMemoryStore
from mark.types import EdgeRelation, GapReport, GapSeverity, MemoryFragment, MemoryScope, MemoryState, MemoryTier

# No __all__ on purpose: star-imports keep exporting every public name, as before.
logger = logging.getLogger("mark.retrieval")
