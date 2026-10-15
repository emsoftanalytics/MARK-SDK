"""Local hybrid retrieval: lexical + vector channels, rank fusion, graph expansion, rerank."""
from __future__ import annotations

import heapq
from datetime import datetime, timezone
import json as _json
import logging
import re
import threading
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Iterable, Protocol
from xml.sax.saxutils import escape as _xml_escape
from xml.sax.saxutils import quoteattr as _xml_quoteattr

from mark.embeddings import EmbeddingProvider, HashEmbeddingProvider
from mark.index import VectorIndex, VectorSearchResult
from mark.intelligence.classifier import QueryAnalysis, QueryClassifier, _STOP_WORDS
from mark.intelligence.compressor import ContextualCompressor, NoopCompressor
from mark.intelligence.query_expander import NoopQueryExpander, QueryExpander
from mark.intelligence.reranker import MemoryReranker, RerankResult
from mark.plasticity.hebbian import EdgeCoActivation, HebbianReinforcement
from mark.plugins import HOOK_POLICY_ENGINE, HOOK_SCORING, PluginRegistry
from mark.store import LocalMemoryStore
from mark.types import EdgeRelation, GapReport, GapSeverity, MemoryFragment, MemoryScope, MemoryState, MemoryTier

logger = logging.getLogger("mark.retrieval")


class RetrievalPolicy(str, Enum):
    """Built-in retrieval policy names.

    FAST / BALANCED / DEEP are authoritative: the pipeline uses exactly that
    policy's knobs. AUTO lets the query classifier pick one per query.
    """
    FAST = "fast"
    BALANCED = "balanced"
    DEEP = "deep"
    AUTO = "auto"


class ReinforceMode(str, Enum):
    """How retrieval feeds back into plasticity.

    RELEVANT  record access for every result; boost importance and co-activate
              edges only for results with real query evidence (default)
    ALL       legacy behaviour: boost every returned result
    NONE      read-only retrieval; no writes at all
    """
    RELEVANT = "relevant"
    ALL = "all"
    NONE = "none"


@dataclass(frozen=True)
class RetrievalPolicySpec:
    """Concrete retrieval knobs for one policy (top_k, depth, thresholds)."""
    name: str
    display_name: str
    description: str = ""
    top_k: int = 10
    graph_depth: int = 1
    rerank_depth: int = 1
    min_confidence: float = 0.30
    enable_decomposition: bool = False
    max_subqueries: int = 1
    critic_enabled: bool = False
    enable_external: bool = False
    is_builtin: bool = True
    expansion_direction: str = "both"   # "outgoing" | "incoming" | "both"


BUILTIN_POLICIES: dict[str, RetrievalPolicySpec] = {
    "fast": RetrievalPolicySpec(
        name="fast",
        display_name="Fast Lookup",
        description="Single-hop local recall.",
        top_k=5,
        graph_depth=0,
        min_confidence=0.25,
        expansion_direction="outgoing",
    ),
    "balanced": RetrievalPolicySpec(
        name="balanced",
        display_name="Balanced",
        description="Hybrid retrieval plus one-hop graph expansion.",
        top_k=10,
        graph_depth=1,
        min_confidence=0.40,
        expansion_direction="both",
    ),
    "deep": RetrievalPolicySpec(
        name="deep",
        display_name="Deep Recall",
        description="Hybrid retrieval plus two-hop graph expansion.",
        top_k=15,
        graph_depth=2,
        min_confidence=0.55,
        expansion_direction="both",
    ),
}

_RETRIEVABLE_STATES = {MemoryState.UNVERIFIED, MemoryState.VERIFIED, MemoryState.PROMOTED}

# Reciprocal Rank Fusion constant (Cormack et al., SIGIR 2009).
RRF_K = 60
# Candidates fetched per channel before fusion, as a multiple of top_k.
_CHANNEL_DEPTH_FACTOR = 5
_CHANNEL_DEPTH_MIN = 50
# Graph neighbours inherit a damped share of the seed score per hop.
_GRAPH_HOP_DECAY = 0.5


@dataclass(frozen=True)
class SessionFilter:
    """Optional session-scoping parameters for RetrievalPipeline.retrieve().

    session_id      exact session match (e.g. "episode-03")
    session_prefix  prefix match (e.g. "season-01/" matches all season-01 sessions)
    tags            all listed tags must be present on returned fragments (AND)
    tier            restrict to a specific MemoryTier
    scope           restrict to a specific MemoryScope
    block_id        restrict to fragments belonging to one memory block
    block_ids       restrict to fragments belonging to any listed block

    Pass no arguments for whole-agent retrieval (default, cross-session).
    Fragments inside QUARANTINED blocks are always excluded regardless of
    the filters used.
    """
    session_id:     str | None         = None
    session_prefix: str | None         = None
    tags:           list[str] | None   = None
    tier:           MemoryTier | None  = None
    scope:          MemoryScope | None = None
    block_id:       str | None         = None
    block_ids:      list[str] | None   = None


# QueryAnalysis is now the richer Pydantic model from intelligence.classifier.
# Imported at the top of this file. Re-exported here for backward compat.

class QueryAnalyzer(Protocol):
    """Protocol for query analysis/classification implementations."""
    def analyze(self, query: str) -> QueryAnalysis:
        """Analyze the query and return the analysis."""
        ...
    def classify(self, query: str) -> QueryAnalysis:
        """Classify the query and return the analysis."""
        ...


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


class GraphExpander:
    """Expands retrieval hits through local memory graph edges.

    Best-first traversal: each neighbour keeps the strongest path score, and
    every hop is damped so expanded neighbours never tie with direct hits.
    """

    def __init__(self, store: LocalMemoryStore, max_hops: int = 2, hop_decay: float = _GRAPH_HOP_DECAY) -> None:
        self._store = store
        self.max_hops = max_hops
        self.hop_decay = hop_decay

    def expand(
        self,
        seed_scores: dict[str, float],
        agent_id: str,
        *,
        direction: str = "both",
        max_hops: int | None = None,
    ) -> dict[str, float]:
        """Expand seed fragment scores through graph edges.

        direction:
            "outgoing" — follow edges where seed node is the source
            "incoming" — follow edges where seed node is the target
            "both"     — traverse in both directions (default; recommended for BALANCED/DEEP)
        """
        hops_limit = self.max_hops if max_hops is None else max_hops
        merged = dict(seed_scores)
        best: dict[str, float] = {}
        frontier: list[tuple[float, str, int]] = []
        for fragment_id, score in seed_scores.items():
            for node in self._store.nodes_for_fragment(fragment_id):
                if node.agent_id in {None, agent_id} and score > best.get(node.id, -1.0):
                    best[node.id] = score
                    heapq.heappush(frontier, (-score, node.id, 0))

        expansive = [rel for rel in EdgeRelation if rel.is_expansive]
        supports_incoming = hasattr(self._store, "edges_to_node")
        expanded: set[str] = set()

        while frontier:
            neg_score, node_id, depth = heapq.heappop(frontier)
            score = -neg_score
            if node_id in expanded or depth >= hops_limit:
                continue
            expanded.add(node_id)

            neighbor_edges: list = []
            if direction in {"outgoing", "both"}:
                neighbor_edges += self._store.edges_from_node(node_id, relation_filter=expansive)
            if direction in {"incoming", "both"} and supports_incoming:
                neighbor_edges += [
                    e for e in self._store.edges_to_node(node_id)
                    if e.relation in expansive
                ]

            for edge in neighbor_edges:
                neighbor_id = edge.target_id if edge.source_id == node_id else edge.source_id
                if neighbor_id in expanded:
                    continue
                neighbor = self._store.get_node(neighbor_id)
                if neighbor is None or neighbor.agent_id not in {None, agent_id}:
                    continue
                propagated = score * edge.weight * neighbor.weight * self.hop_decay
                if propagated <= best.get(neighbor_id, 0.0):
                    continue
                best[neighbor_id] = propagated
                if neighbor.fragment_id and propagated > merged.get(neighbor.fragment_id, 0.0):
                    merged[neighbor.fragment_id] = propagated
                heapq.heappush(frontier, (-propagated, neighbor_id, depth + 1))

        return merged

    def contradiction_penalty_ids(self, seed_fragment_ids: list[str]) -> set[str]:
        """Return fragment ids penalized for contradictions."""
        penalized: set[str] = set()
        for fragment_id in seed_fragment_ids:
            for node in self._store.nodes_for_fragment(fragment_id):
                for edge in self._store.contradiction_edges(node.id):
                    other_id = edge.target_id if edge.source_id == node.id else edge.source_id
                    other = self._store.get_node(other_id)
                    if other and other.fragment_id:
                        penalized.add(other.fragment_id)
        return penalized


class RetrievalPipeline:
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
        self._hebbian = HebbianReinforcement()
        self._coact   = EdgeCoActivation()
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

    # ── Write path ────────────────────────────────────────────────────────────

    def index_fragment(self, fragment: MemoryFragment) -> MemoryFragment:
        """Embed, persist, and index a fragment."""
        embedding = self._embedder.embed(fragment.content)
        indexed = fragment.model_copy(update={"embedding": embedding})
        self._store.store(indexed)
        self._index_add(indexed.id, embedding, indexed.content, indexed.agent_id)
        return indexed

    def index_fragments(self, fragments: list[MemoryFragment], batch_size: int = 64) -> list[MemoryFragment]:
        """Embed (in batches), persist (one transaction per batch), and index fragments."""
        indexed_all: list[MemoryFragment] = []
        store_many = getattr(self._store, "store_many", None)
        for start in range(0, len(fragments), batch_size):
            chunk = fragments[start:start + batch_size]
            vectors = self._embedder.embed_batch([fragment.content for fragment in chunk])
            indexed = [f.model_copy(update={"embedding": v}) for f, v in zip(chunk, vectors)]
            if store_many is not None:
                store_many(indexed)
            else:
                for fragment in indexed:
                    self._store.store(fragment)
            for fragment in indexed:
                self._index_add(fragment.id, fragment.embedding, fragment.content, fragment.agent_id)
            indexed_all.extend(indexed)
        return indexed_all

    # ── Read path ─────────────────────────────────────────────────────────────

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
        fused: dict[str, float] = {}
        cosine: dict[str, float] = {}
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
            for rank, (fragment_id, _bm25) in enumerate(lexical_hits, 1):
                fused[fragment_id] = fused.get(fragment_id, 0.0) + 1.0 / (RRF_K + rank)

        # Keep the strongest fused candidates, then fetch them in one query.
        seed_ids = heapq.nlargest(depth, fused, key=fused.__getitem__)
        fragments_by_id = self._get_many(seed_ids)
        query_terms = _content_terms(query)
        relevance = {
            fid: _relevance(query_terms, fragments_by_id[fid].content, cosine.get(fid, 0.0))
            for fid in seed_ids if fid in fragments_by_id
        }
        # Fused rank evidence (normalised to the best possible fused score) is
        # combined with absolute evidence so ranking stays query-aware even
        # when one channel is empty.
        max_fused = len(query_texts) * 2.0 / (RRF_K + 1)
        seed_scores = {
            fid: 0.5 * min(1.0, fused[fid] / max_fused) + 0.5 * relevance[fid]
            for fid in relevance
        }

        contradiction_ids: set[str] = set()
        expanded_scores = seed_scores
        if spec.graph_depth > 0 and seed_scores:
            top_seeds = dict(heapq.nlargest(effective_top_k, seed_scores.items(), key=lambda kv: kv[1]))
            graph_scores = self._expander.expand(
                top_seeds, agent_id, direction=spec.expansion_direction, max_hops=spec.graph_depth,
            )
            expanded_scores = {**seed_scores}
            for fid, score in graph_scores.items():
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

        # Hash-check failures and stale index entries never surface.
        candidates = [
            VectorSearchResult(fragment_id=fid, score=score, content=fragments_by_id[fid].content)
            for fid, score in expanded_scores.items()
            if fid in fragments_by_id and fragments_by_id[fid].state in _RETRIEVABLE_STATES
        ]
        candidates = _dedupe_content(candidates)

        scorer = self._plugins.get(HOOK_SCORING)
        if scorer:
            ranked: list[RerankResult] = list(scorer(
                candidates,
                query=query,
                policy=policy,
                agent_id=agent_id,
                contradiction_ids=contradiction_ids,
            ))[:effective_top_k]
        else:
            ranked = self._reranker.rerank(
                query, candidates, effective_top_k, agent_id, contradiction_ids,
                fragments=fragments_by_id,
                recency_weight=0.5 if getattr(analysis, "has_temporal", False) else 0.0,
            )

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

    # ── Internals ─────────────────────────────────────────────────────────────

    def _resolve_spec(self, policy: RetrievalPolicy, analysis: QueryAnalysis) -> RetrievalPolicySpec:
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

        When the query names attributes that facts exist for ("port"), only
        those are returned; otherwise every attribute of the subject is.
        """
        current = getattr(self._store, "fact_current", None)
        anchors = query_anchors(query)
        if current is None or not anchors:
            return []
        from mark.memory.facts import _fact

        terms = set(_content_terms(query))
        found = []
        for anchor in anchors:
            rows = current(agent_id, anchor)
            if not rows:
                continue
            named = [r for r in rows if set(_content_terms(r["attribute"])) & terms]
            for row in named or rows:
                if intent == TemporalIntent.ORIGINAL:
                    history = self._store.fact_history(agent_id, row["subject"], row["attribute"])
                    row = history[0] if history else row
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

    def _allowed_ids(self, agent_id: str, sf: SessionFilter) -> set[str]:
        kwargs: dict = dict(
            session_id=sf.session_id,
            session_prefix=sf.session_prefix,
            tags=sf.tags,
            tier=sf.tier,
            scope=sf.scope,
        )
        if sf.block_id is not None or sf.block_ids:
            kwargs["block_id"] = sf.block_id
            kwargs["block_ids"] = sf.block_ids
        retrievable_ids = getattr(self._store, "retrievable_ids", None)
        if retrievable_ids is not None:
            return set(retrievable_ids(agent_id, **kwargs))
        # Legacy/custom stores: uncapped list, TTL filtered in Python.
        try:
            rows = self._store.list_by_agent(agent_id, states=_RETRIEVABLE_STATES, limit=None, **kwargs)
        except TypeError:
            rows = self._store.list_by_agent(agent_id, states=_RETRIEVABLE_STATES, limit=1_000_000, **kwargs)
        return {fragment.id for fragment in rows if not fragment.is_expired()}

    def _sync_index(self, agent_id: str, allowed_ids: set[str]) -> set[str]:
        """Make sure every allowed fragment is indexed with the current embedder.

        Returns the agent's indexed ids after syncing.
        """
        with self._sync_lock:
            if agent_id not in self._verified_agents:
                self._verify_embedder(agent_id)
                self._verified_agents.add(agent_id)
            indexed = self._index_ids(agent_id)
            missing = allowed_ids - indexed
            if not missing:
                return indexed
            embeddings = self._stored_embeddings(agent_id, missing)
            to_embed: list[str] = []
            dim = getattr(self._embedder, "dim", None)
            for fragment_id in missing:
                vector = embeddings.get(fragment_id)
                if vector and (dim is None or len(vector) == dim):
                    self._index_add(fragment_id, vector, "", agent_id)
                else:
                    to_embed.append(fragment_id)
            if to_embed:
                self._embed_and_index(agent_id, to_embed)
            return self._index_ids(agent_id)

    def _verify_embedder(self, agent_id: str) -> None:
        """Re-embed an agent's memories when the embedder changed since they were stored."""
        get_meta = getattr(self._store, "get_meta", None)
        set_meta = getattr(self._store, "set_meta", None)
        fingerprint = getattr(self._embedder, "fingerprint", None)
        if get_meta is None or set_meta is None or not fingerprint:
            return
        key = f"{self.EMBEDDER_META_KEY}:{agent_id}"
        stored = get_meta(key)
        if stored is not None and _same_embedding_space(stored, fingerprint):
            return
        if fingerprint.endswith(":None"):
            # Dimension not known until the first embed call: never persist a
            # partial fingerprint (it would force a re-embed on every restart).
            return
        if stored is not None:
            logger.info("embedder changed for agent %r (%s -> %s); re-embedding", agent_id, stored, fingerprint)
            ids = list(self._stored_embeddings(agent_id, None))
            self._remove_agent_from_index(agent_id, ids)
            self._embed_and_index(agent_id, ids)
        set_meta(key, fingerprint)

    def _embed_and_index(self, agent_id: str, fragment_ids: list[str], batch_size: int = 64) -> None:
        update = getattr(self._store, "update_embedding", None)
        for start in range(0, len(fragment_ids), batch_size):
            chunk = fragment_ids[start:start + batch_size]
            fragments = self._get_many(chunk)
            ordered = [fragments[fid] for fid in chunk if fid in fragments]
            if not ordered:
                continue
            vectors = self._embedder.embed_batch([f.content for f in ordered])
            for fragment, vector in zip(ordered, vectors):
                if update is not None:
                    update(fragment.id, vector)
                self._index_add(fragment.id, vector, fragment.content, agent_id)

    def _stored_embeddings(self, agent_id: str, ids: set[str] | None) -> dict[str, list[float] | None]:
        fetch = getattr(self._store, "embeddings_for_agent", None)
        if fetch is not None:
            rows = fetch(agent_id)
            return {fid: vec for fid, vec in rows if ids is None or fid in ids}
        if ids is None:
            try:
                fragments = self._store.list_by_agent(agent_id, states=_RETRIEVABLE_STATES, limit=None)
            except TypeError:
                fragments = self._store.list_by_agent(agent_id, states=_RETRIEVABLE_STATES, limit=1_000_000)
            return {f.id: f.embedding for f in fragments}
        found = self._get_many(list(ids))
        return {fid: f.embedding for fid, f in found.items()}

    def _vector_search(
        self, text: str, agent_id: str, allowed_ids: set[str], scope_ids: set[str] | None, depth: int,
    ) -> list[VectorSearchResult]:
        q_vec = self._embedder.embed(text)
        try:
            return self._index_search(q_vec, agent_id, scope_ids, depth, allowed_ids)
        except ValueError as exc:  # DimensionMismatchError subclasses ValueError
            logger.warning("vector search failed (%s); rebuilding index for agent %r", exc, agent_id)
            with self._sync_lock:
                ids = list(allowed_ids)
                self._remove_agent_from_index(agent_id, ids)
                self._embed_and_index(agent_id, ids)
            return self._index_search(q_vec, agent_id, allowed_ids, depth, allowed_ids)

    def _lexical_search(
        self, text: str, agent_id: str, allowed_ids: set[str], scope_ids: set[str] | None, depth: int,
    ) -> list[tuple[str, float]]:
        search = getattr(self._store, "lexical_search", None)
        if search is None:
            return []
        try:
            hits = search(agent_id, text, limit=depth, include_ids=scope_ids)
        except Exception as exc:  # lexical channel is best-effort; vector still answers
            logger.warning("lexical search failed: %s", exc)
            return []
        return [(fid, score) for fid, score in hits if fid in allowed_ids]

    def _reinforce_results(
        self,
        fragments: list[MemoryFragment],
        relevances: list[float],
        spec: RetrievalPolicySpec,
        mode: ReinforceMode,
    ) -> None:
        if mode == ReinforceMode.NONE or not fragments:
            return
        ids = [f.id for f in fragments]
        touch_many = getattr(self._store, "touch_many", None)
        if touch_many is not None:
            touch_many(ids)
        elif hasattr(self._store, "touch"):
            for fid in ids:
                self._store.touch(fid)

        # Only results backed by real query evidence are reinforced, otherwise
        # whatever lands in the top-k grows more important on every read.
        if mode == ReinforceMode.ALL:
            reinforced = list(fragments)
        else:
            reinforced = [f for f, rel in zip(fragments, relevances) if rel >= spec.min_confidence]
        if not reinforced:
            return
        refreshed = self._get_many([f.id for f in reinforced])
        updates: dict[str, float] = {}
        for fragment in reinforced:
            current = refreshed.get(fragment.id, fragment)
            new_imp = self._hebbian.on_access(current)
            if abs(new_imp - current.importance) > 1e-9:
                updates[fragment.id] = new_imp
        if updates:
            update_many = getattr(self._store, "update_importance_many", None)
            if update_many is not None:
                update_many(updates)
            elif hasattr(self._store, "update_importance"):
                for fid, value in updates.items():
                    self._store.update_importance(fid, value)

        self._coactivate({f.id for f in reinforced})

    def _coactivate(self, frag_ids: set[str]) -> None:
        """Strengthen edges between co-retrieved fragments (undirected, once per edge)."""
        if len(frag_ids) < 2 or not hasattr(self._store, "update_edge_weight"):
            return
        node_to_fragment: dict[str, str] = {}
        for fid in frag_ids:
            for node in self._store.nodes_for_fragment(fid):
                node_to_fragment[node.id] = fid
        seen_edges: set[str] = set()
        for node_id, fid in node_to_fragment.items():
            edges = list(self._store.edges_from_node(node_id))
            if hasattr(self._store, "edges_to_node"):
                edges += self._store.edges_to_node(node_id)
            for edge in edges:
                if edge.id in seen_edges:
                    continue
                other = edge.target_id if edge.source_id == node_id else edge.source_id
                other_fid = node_to_fragment.get(other)
                if other_fid is None or other_fid == fid:
                    continue
                seen_edges.add(edge.id)
                new_w = self._coact.on_co_retrieval(edge.weight)
                if abs(new_w - edge.weight) > 1e-9:
                    self._store.update_edge_weight(edge.id, new_w)

    # ── Store / index compatibility helpers ───────────────────────────────────

    def _get_many(self, ids: Iterable[str]) -> dict[str, MemoryFragment]:
        ids = list(ids)
        get_many = getattr(self._store, "get_many", None)
        if get_many is not None:
            return dict(get_many(ids))
        found: dict[str, MemoryFragment] = {}
        for fid in ids:
            fragment = self._store.get(fid)
            if fragment is not None:
                found[fid] = fragment
        return found

    def _index_add(self, fragment_id: str, vector: list[float], content: str, agent_id: str) -> None:
        try:
            self._index.add(fragment_id, vector, content, agent_id=agent_id)
        except TypeError:  # legacy/custom index without agent partitions
            self._index.add(fragment_id, vector, content)

    def _index_ids(self, agent_id: str) -> set[str]:
        ids = getattr(self._index, "ids", None)
        if ids is not None:
            try:
                return set(ids(agent_id))
            except TypeError:
                return set(ids())
        return {fid for fid in getattr(self._index, "_vectors", {})}

    def _index_search(
        self,
        q_vec: list[float],
        agent_id: str,
        scope_ids: set[str] | None,
        depth: int,
        allowed_ids: set[str],
    ) -> list[VectorSearchResult]:
        try:
            return self._index.search(q_vec, top_k=depth, include_ids=scope_ids, agent_id=agent_id)
        except TypeError:
            # Legacy index without agent partitions: always restrict to the agent's ids.
            return self._index.search(q_vec, top_k=depth, include_ids=allowed_ids)

    def _remove_agent_from_index(self, agent_id: str, ids: Iterable[str]) -> None:
        clear = getattr(self._index, "clear", None)
        if clear is not None:
            try:
                clear(agent_id)
                return
            except TypeError:
                pass
        for fid in ids:
            self._index.remove(fid)


def _same_embedding_space(stored: str, current: str) -> bool:
    """Fingerprints match, or match on model identity when a dimension is unknown."""
    if stored == current:
        return True
    stored_model, _, stored_dim = stored.rpartition(":")
    current_model, _, current_dim = current.rpartition(":")
    return stored_model == current_model and "None" in {stored_dim, current_dim}


# ── Lexical evidence helpers ──────────────────────────────────────────────────

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)

# Words that express *when* the caller cares about, not *what*: they drive
# temporal ordering and must not count as content terms.
_CURRENT_RE = re.compile(r"\b(now|current(?:ly)?|latest|most recent|today|nowadays|these days|as of (?:now|today))\b", re.I)
_ORIGINAL_RE = re.compile(r"\b(original(?:ly)?|initial(?:ly)?|first|earliest|at first|before any)\b", re.I)
_TEMPORAL_WORDS = {"now", "current", "currently", "latest", "recent", "today", "nowadays",
                   "original", "originally", "initial", "initially", "first", "earliest"}

# Distinctive anchors: compound identifiers (atlas-core, HAL-1234, eu-west-1,
# v2.3.1), tokens with digits, and multi-word capitalised names.
_COMPOUND_RE = re.compile(r"\b\w+(?:[-_.:/]\w+)+\b")
_DIGIT_TOKEN_RE = re.compile(r"\b(?=\w*\d)\w{3,}\b")
_NAME_RE = re.compile(r"\b([A-Z][\w'’]+(?:\s+[A-Z][\w'’]+)+)\b")
_QUESTION_WORDS = {"Who", "What", "Which", "When", "Where", "Why", "How", "Is", "Are", "Does", "Do", "Did", "Per"}

# Event time: only a date stamp at the very start of a memory counts
# ("[2026-03-05] ...", "2026-03-05T10:00Z: ..."). Dates elsewhere in the text
# ("drill scheduled for 2026-04-16") describe other events and are ignored.
_LEADING_DATE_RE = re.compile(r"^\s*\[?(\d{4}-\d{2}-\d{2})(?:[T ](\d{2}:\d{2}(?::\d{2})?))?")


class TemporalIntent(str, Enum):
    """What point in time a query asks about."""
    NONE = "none"
    CURRENT = "current"
    ORIGINAL = "original"


def detect_temporal_intent(query: str) -> TemporalIntent:
    """CURRENT for "now/current/latest" queries, ORIGINAL for "original/first", else NONE."""
    if _ORIGINAL_RE.search(query):
        return TemporalIntent.ORIGINAL
    if _CURRENT_RE.search(query):
        return TemporalIntent.CURRENT
    return TemporalIntent.NONE


def query_anchors(query: str) -> list[str]:
    """Distinctive identifiers and names in a query, normalised to lowercase."""
    anchors: list[str] = []
    for match in _COMPOUND_RE.findall(query):
        anchors.append(match)
    for match in _DIGIT_TOKEN_RE.findall(query):
        if not any(match in compound for compound in anchors):
            anchors.append(match)
    for match in _NAME_RE.findall(query):
        words = match.split()
        while words and words[0] in _QUESTION_WORDS:
            words = words[1:]
        if len(words) >= 2:
            anchors.append(" ".join(words))
    seen: list[str] = []
    for anchor in anchors:
        norm = _anchor_norm(anchor)
        if norm and norm not in seen:
            seen.append(norm)
    return seen


def _anchor_norm(text: str) -> str:
    """Normalise identifiers so 'Atlas-Core', 'atlas core', 'atlas_core' compare equal."""
    return " ".join(_TOKEN_RE.findall(text.lower().replace("’", "'")))


def _contains_anchor(content_norm: str, anchor: str) -> bool:
    return re.search(rf"(?<!\w){re.escape(anchor)}(?!\w)", content_norm) is not None


def declared_event_time(fragment: MemoryFragment) -> datetime | None:
    """The event time a memory declares, or None.

    Explicit ``metadata["occurred_at"]`` wins; otherwise a date stamp at the
    very start of the content.
    """
    raw = fragment.metadata.get("occurred_at") if fragment.metadata else None
    if raw:
        parsed = _parse_when(str(raw))
        if parsed is not None:
            return parsed
    match = _LEADING_DATE_RE.match(fragment.content)
    if match:
        return _parse_when(match.group(1) + (f"T{match.group(2)}" if match.group(2) else ""))
    return None


def event_time(fragment: MemoryFragment) -> datetime:
    """When the remembered event happened: declared time, else when it was stored."""
    declared = declared_event_time(fragment)
    if declared is not None:
        return declared
    stamp = fragment.created_at
    return stamp if stamp.tzinfo else stamp.replace(tzinfo=timezone.utc)


def _parse_when(text: str) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(text.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _stem(token: str) -> str:
    """Tiny suffix stemmer: listens/listened → listen, owns/owned/owner → own,
    gates → gate, boxes → box, policies → policy, running → run.

    Deliberately conservative (no prefix matching): identifiers and names such
    as "titan" must never match "titanium". Query and content are stemmed the
    same way, so an occasional over-merge only widens a match.
    """
    if len(token) < 4 or not token.isalpha():
        return token
    if token.endswith(("ies", "ied")) and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith("es") and token[:-2].endswith(("s", "x", "z", "ch", "sh")):
        return token[:-2]
    if token.endswith("s") and not token.endswith(("ss", "us", "is")):
        token = token[:-1]
        if len(token) < 4:
            return token
    for suffix in ("ments", "ment", "ing", "ers", "er", "ed"):
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            stem = token[: -len(suffix)]
            if len(stem) >= 4 and stem[-1] == stem[-2] and stem[-1] not in "lsz":
                stem = stem[:-1]  # running → runn → run
            return stem
    return token


def _content_terms(text: str) -> list[str]:
    """Salient query terms: lowercase stems without stopwords or temporal words."""
    terms: list[str] = []
    for token in _TOKEN_RE.findall(text.lower()):
        if token.endswith("'s"):
            token = token[:-2]
        if token in _STOP_WORDS or token in _TEMPORAL_WORDS or (len(token) < 2 and not token.isdigit()):
            continue
        stem = _stem(token)
        if stem not in terms:
            terms.append(stem)
    return terms


def _content_stems(content: str) -> set[str]:
    return {_stem(token) for token in _TOKEN_RE.findall(content.lower())}


def _coverage(query_terms: list[str], content: str) -> float:
    if not query_terms:
        return 0.0
    stems = _content_stems(content)
    return sum(1 for term in query_terms if term in stems) / len(query_terms)


def _relevance(query_terms: list[str], content: str, cosine: float) -> float:
    """Calibrated query evidence in [0, 1]: max of vector similarity and term coverage."""
    return max(0.0, min(1.0, max(cosine, _coverage(query_terms, content))))


def temporal_order(
    query: str,
    fragments: list[MemoryFragment],
    scores: list[float],
    relevances: list[float],
) -> tuple[list[MemoryFragment], list[float], list[float], TemporalIntent]:
    """Reorder results about the query's subject by event time.

    For CURRENT queries the newest matching record comes first; for ORIGINAL
    queries the oldest. Only fragments that mention every query anchor (e.g.
    the service name) and cover the remaining query terms at least as well as
    the best such fragment are reordered, so a newer record about a different
    attribute never jumps ahead. Other results keep their rank. Scores are
    re-assigned so they stay non-increasing.
    """
    intent = detect_temporal_intent(query)
    anchors = query_anchors(query)
    if intent == TemporalIntent.NONE or not anchors or len(fragments) < 2:
        return fragments, scores, relevances, intent
    anchor_words = {word for anchor in anchors for word in anchor.split()}
    attribute_terms = [t for t in _content_terms(query) if t not in anchor_words]
    subject: list[int] = []
    coverage: dict[int, float] = {}
    for i, fragment in enumerate(fragments):
        content_norm = _anchor_norm(fragment.content)
        if all(_contains_anchor(content_norm, anchor) for anchor in anchors):
            subject.append(i)
            coverage[i] = _coverage(attribute_terms, fragment.content) if attribute_terms else 1.0
    if len(subject) < 2:
        return fragments, scores, relevances, intent
    # Attribute match is bucketed: "port" matches both "moved ... to port X"
    # and "listens on port Y"; exact phrasing must not outrank recency.
    best = max(coverage.values())
    band = [i for i in subject if coverage[i] > 0 and coverage[i] >= best * 0.5] or subject
    newest_first = intent == TemporalIntent.CURRENT
    # Never compare two clocks: declared event times (e.g. a log's dates) and
    # storage times can be years apart. Dated records are ordered by their
    # declared time; undated ones keep relevance order after them. Only when
    # nothing in the band is dated is storage time a consistent clock.
    dated = [i for i in band if declared_event_time(fragments[i]) is not None]
    undated = [i for i in band if declared_event_time(fragments[i]) is None]
    if dated:
        dated.sort(key=lambda i: declared_event_time(fragments[i]), reverse=newest_first)
        band = dated + undated
    else:
        band.sort(key=lambda i: event_time(fragments[i]), reverse=newest_first)
    band_set = set(band)
    rest = [i for i in range(len(fragments)) if i not in band_set]
    order = band + rest
    top = max(scores) if scores else 0.0
    new_scores = [round(max(0.0, top - 1e-4 * rank), 6) for rank in range(len(band))]
    floor = new_scores[-1] if new_scores else top
    for i in rest:
        floor = min(floor, scores[i])
        new_scores.append(floor)
    return ([fragments[i] for i in order], new_scores, [relevances[i] for i in order], intent)


def _dedupe_content(candidates: list[VectorSearchResult]) -> list[VectorSearchResult]:
    """Drop exact duplicate contents, keeping the best-scored copy."""
    best: dict[str, VectorSearchResult] = {}
    for candidate in candidates:
        key = " ".join(candidate.content.lower().split())
        current = best.get(key)
        if current is None or candidate.score > current.score:
            best[key] = candidate
    return list(best.values())
