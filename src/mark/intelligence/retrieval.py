"""Local retrieval pipeline: filtered vector search, graph expansion, rerank."""
from __future__ import annotations

import json as _json
from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol

from mark.embeddings import EmbeddingProvider, HashEmbeddingProvider
from mark.index import VectorIndex, VectorSearchResult
from mark.intelligence.classifier import QueryAnalysis, QueryClassifier
from mark.intelligence.compressor import ContextualCompressor, NoopCompressor
from mark.intelligence.query_expander import NoopQueryExpander, QueryExpander
from mark.intelligence.reranker import MemoryReranker, RerankResult
from mark.plasticity.hebbian import EdgeCoActivation, HebbianReinforcement
from mark.plugins import HOOK_POLICY_ENGINE, HOOK_SCORING, PluginRegistry
from mark.store import LocalMemoryStore
from mark.types import EdgeRelation, GapReport, GapSeverity, MemoryFragment, MemoryScope, MemoryState, MemoryTier


class RetrievalPolicy(str, Enum):
    """Built-in retrieval policy names: FAST, BALANCED, DEEP."""
    FAST = "fast"
    BALANCED = "balanced"
    DEEP = "deep"


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
        description="Vector retrieval plus one-hop graph expansion.",
        top_k=10,
        graph_depth=1,
        min_confidence=0.40,
        expansion_direction="both",
    ),
    "deep": RetrievalPolicySpec(
        name="deep",
        display_name="Deep Recall",
        description="Vector retrieval plus two-hop graph expansion.",
        top_k=15,
        graph_depth=2,
        min_confidence=0.55,
        expansion_direction="both",
    ),
}

_RETRIEVABLE_STATES = {MemoryState.UNVERIFIED, MemoryState.VERIFIED, MemoryState.PROMOTED}


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
    """Full retrieval outcome: fragments, scores, policy, and gap report."""
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
        compressed_flag = " compressed=true" if self.was_compressed else ""
        lines = [f"<memory_context query={self.query!r}{compressed_flag}>"]
        if not self.fragments:
            lines.append("  [No relevant memory found]")
        words = 0
        for fragment, score in zip(self.fragments, self.scores):
            source = f" | source: {fragment.source}" if fragment.source else ""
            line = f"  [{score:.3f}|{fragment.state.value}{source}] {fragment.content}"
            words += len(line.split())
            if words > max_words:
                lines.append("  [... truncated for token budget]")
                break
            lines.append(line)
        if self.gap_report.severity not in {GapSeverity.NONE, GapSeverity.LOW}:
            lines.append(f"  <memory_gap severity={self.gap_report.severity.value}>")
            lines.append(f"  {self.gap_report.reason}")
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
    """Expands vector hits through local memory graph edges."""

    def __init__(self, store: LocalMemoryStore, max_hops: int = 2) -> None:
        self._store = store
        self.max_hops = max_hops

    def expand(
        self,
        seed_scores: dict[str, float],
        agent_id: str,
        *,
        direction: str = "both",
    ) -> dict[str, float]:
        """Expand seed fragment scores through graph edges.

        direction:
            "outgoing" — follow edges where seed node is the source (original behaviour)
            "incoming" — follow edges where seed node is the target
            "both"     — traverse in both directions (default; recommended for BALANCED/DEEP)
        """
        merged = dict(seed_scores)
        visited: set[str] = set()
        frontier: list[tuple[str, float, int]] = []
        for fragment_id, score in seed_scores.items():
            for node in self._store.nodes_for_fragment(fragment_id):
                if node.agent_id in {None, agent_id} and node.id not in visited:
                    visited.add(node.id)
                    frontier.append((node.id, score, 0))

        expansive = [rel for rel in EdgeRelation if rel.is_expansive]
        supports_incoming = hasattr(self._store, "edges_to_node")

        while frontier:
            node_id, score, depth = frontier.pop(0)
            if depth >= self.max_hops:
                continue

            neighbor_edges: list = []
            if direction in {"outgoing", "both"}:
                neighbor_edges += self._store.edges_from_node(node_id, relation_filter=expansive)
            if direction in {"incoming", "both"} and supports_incoming:
                # Incoming traversal uses the same expansive filter
                neighbor_edges += [
                    e for e in self._store.edges_to_node(node_id)
                    if e.relation in expansive
                ]

            for edge in neighbor_edges:
                neighbor_id = edge.target_id if edge.source_id == node_id else edge.source_id
                if neighbor_id in visited:
                    continue
                neighbor = self._store.get_node(neighbor_id)
                if neighbor is None or neighbor.agent_id not in {None, agent_id}:
                    continue
                visited.add(neighbor_id)
                propagated = score * edge.weight * neighbor.weight
                if neighbor.fragment_id and propagated > merged.get(neighbor.fragment_id, 0.0):
                    merged[neighbor.fragment_id] = propagated
                frontier.append((neighbor_id, propagated, depth + 1))

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
    """Local vector retrieval, graph expansion, reranking, and gap reporting."""

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
    ) -> None:
        self._store = store
        self._index = index or VectorIndex()
        self._embedder = embedder or HashEmbeddingProvider()
        self._analyzer = analyzer or QueryClassifier()
        self._expander = expander or GraphExpander(store)
        self._reranker = reranker or MemoryReranker(store)
        self._plugins  = plugins or PluginRegistry()
        self._compressor: ContextualCompressor = compressor or NoopCompressor()
        self._query_expander: QueryExpander     = query_expander or NoopQueryExpander()
        self._top_k_with_llm = max(1, top_k_with_llm)
        self._hebbian = HebbianReinforcement()
        self._coact   = EdgeCoActivation()

    def index_fragment(self, fragment: MemoryFragment) -> MemoryFragment:
        """Embed, persist, and index a fragment."""
        embedding = self._embedder.embed(fragment.content)
        indexed = fragment.model_copy(update={"embedding": embedding})
        self._store.store(indexed)
        self._index.add(indexed.id, embedding, indexed.content)
        return indexed

    def configure_compressor(self, compressor: ContextualCompressor) -> None:
        """Replace the active compressor at runtime (late-binding)."""
        self._compressor = compressor

    def configure_query_expander(self, expander: QueryExpander) -> None:
        """Replace the active query expander at runtime (late-binding)."""
        self._query_expander = expander

    def configure_top_k_with_llm(self, top_k: int) -> None:
        """Set the candidate pool size used when an LLM compressor is active."""
        self._top_k_with_llm = max(1, top_k)

    def retrieve(
        self,
        query: str,
        agent_id: str,
        policy: RetrievalPolicy = RetrievalPolicy.BALANCED,
        *,
        session_filter: SessionFilter | None = None,
        compress: bool = False,
        expand: bool = False,
    ) -> RetrievalResult:
        """Run the full pipeline: filter, vector search, expand, rerank."""
        analysis = self._analyzer.analyze(query)
        spec = BUILTIN_POLICIES[policy.value]
        policy_engine = self._plugins.get(HOOK_POLICY_ENGINE)
        if policy_engine:
            spec = policy_engine(analysis, hint=policy)
        elif hasattr(analysis, "estimated_top_k") and analysis.estimated_top_k != spec.top_k:
            # QueryClassifier drove a different top_k — use it, keep other spec fields
            spec = RetrievalPolicySpec(
                name        = spec.name,
                display_name= spec.display_name,
                description = spec.description,
                top_k       = analysis.estimated_top_k,
                graph_depth = (2 if getattr(analysis, "question_depth", 1) >= 3
                               else 1 if getattr(analysis, "question_depth", 1) == 2
                               else spec.graph_depth),
                rerank_depth    = spec.rerank_depth,
                min_confidence  = spec.min_confidence,
                enable_decomposition = spec.enable_decomposition,
            )

        self._sync_index(agent_id)

        # Compute allowed IDs before vector search so filters restrict scoring,
        # not just results. Without this, unrelated sessions dominate top-k and
        # filtered fragments are starved before they can be ranked.
        sf = session_filter or SessionFilter()
        is_filtered = any([sf.session_id, sf.session_prefix, sf.tags, sf.tier, sf.scope,
                           sf.block_id, sf.block_ids])
        list_kwargs: dict = dict(
            states=_RETRIEVABLE_STATES,
            session_id=sf.session_id,
            session_prefix=sf.session_prefix,
            tags=sf.tags,
            tier=sf.tier,
            scope=sf.scope,
        )
        # Older/custom stores may not support block filters — pass only when used.
        if sf.block_id is not None or sf.block_ids:
            list_kwargs["block_id"] = sf.block_id
            list_kwargs["block_ids"] = sf.block_ids
        allowed_ids = {
            fragment.id
            for fragment in self._store.list_by_agent(agent_id, **list_kwargs)
            if not fragment.is_expired()
        }

        # When an LLM compressor is active and compression is requested, fetch a
        # larger candidate pool so the compressor has more to work with before
        # distilling to what's truly relevant.  Without an LLM the policy's top_k
        # is used as-is (default fast=5, balanced=10, deep=15).
        has_llm = compress and not isinstance(self._compressor, NoopCompressor)
        effective_top_k = self._top_k_with_llm if has_llm else spec.top_k

        # Multi-query expansion: generate alternate phrasings, then merge scores by max.
        query_texts = (
            self._query_expander.expand(query)
            if expand and not isinstance(self._query_expander, NoopQueryExpander)
            else [query]
        )
        seed_scores: dict[str, float] = {}
        for q_text in query_texts:
            q_vec  = self._embedder.embed(q_text)
            hits   = self._index.search(q_vec, top_k=effective_top_k * 2,
                                        include_ids=allowed_ids if is_filtered else None)
            for hit in hits:
                # Use None-sentinel so score=0.0 hits are included on first query
                prev = seed_scores.get(hit.fragment_id)
                if prev is None or hit.score > prev:
                    seed_scores[hit.fragment_id] = hit.score

        if spec.graph_depth > 0 and seed_scores:
            self._expander.max_hops = spec.graph_depth
            direction = getattr(spec, "expansion_direction", "both")
            expanded_scores = self._expander.expand(seed_scores, agent_id, direction=direction)
            contradiction_ids = self._expander.contradiction_penalty_ids(list(seed_scores))
        else:
            expanded_scores = seed_scores
            contradiction_ids = set()

        candidates = [
            VectorSearchResult(fragment_id=fragment_id, score=score)
            for fragment_id, score in expanded_scores.items()
            if fragment_id in allowed_ids
        ]

        scorer = self._plugins.get(HOOK_SCORING)
        if scorer:
            ranked: list[RerankResult] = scorer(
                candidates,
                query=query,
                policy=policy,
                agent_id=agent_id,
                contradiction_ids=contradiction_ids,
            )
        else:
            ranked = self._reranker.rerank(query, candidates, effective_top_k, agent_id, contradiction_ids)

        fragments: list[MemoryFragment] = []
        scores: list[float] = []
        for result in ranked:
            fragment = self._store.get(result.fragment_id)
            if fragment is not None:
                fragments.append(fragment)
                scores.append(result.score)

        # Hebbian LTP: touch first so access_count is current when computing boost
        frag_id_set = {f.id for f in fragments}
        for fragment in fragments:
            if hasattr(self._store, "touch"):
                self._store.touch(fragment.id)
                # Re-fetch so on_access sees the incremented access_count
                refreshed = self._store.get(fragment.id) or fragment
            else:
                refreshed = fragment
            new_imp = self._hebbian.on_access(refreshed)
            if abs(new_imp - refreshed.importance) > 1e-9:
                if hasattr(self._store, "update_importance"):
                    self._store.update_importance(fragment.id, new_imp)

        # Hebbian co-activation: strengthen edges between co-retrieved fragments (undirected).
        # Each edge is strengthened at most once per retrieval regardless of how many
        # fragment pairs reference it.
        if len(fragments) > 1 and hasattr(self._store, "update_edge_weight"):
            seen_edges: set[str] = set()
            for fragment in fragments:
                for node in self._store.nodes_for_fragment(fragment.id):
                    # Outgoing edges
                    for edge in self._store.edges_from_node(node.id):
                        if edge.id in seen_edges:
                            continue
                        tgt = self._store.get_node(edge.target_id)
                        if tgt and tgt.fragment_id and tgt.fragment_id in frag_id_set:
                            new_w = self._coact.on_co_retrieval(edge.weight)
                            if abs(new_w - edge.weight) > 1e-9:
                                self._store.update_edge_weight(edge.id, new_w)
                            seen_edges.add(edge.id)
                    # Incoming edges — symmetric strengthening
                    if hasattr(self._store, "edges_to_node"):
                        for edge in self._store.edges_to_node(node.id):
                            if edge.id in seen_edges:
                                continue
                            src = self._store.get_node(edge.source_id)
                            if src and src.fragment_id and src.fragment_id in frag_id_set:
                                new_w = self._coact.on_co_retrieval(edge.weight)
                                if abs(new_w - edge.weight) > 1e-9:
                                    self._store.update_edge_weight(edge.id, new_w)
                                seen_edges.add(edge.id)

        top_score = scores[0] if scores else 0.0
        gap_report = GapReport.from_score(top_score, len(fragments), spec.min_confidence)
        if gap_report.should_search:
            hint = " ".join(analysis.keywords[:5]) if analysis.keywords else query
            gap_report = gap_report.model_copy(update={"search_query_hint": hint})

        # Optional contextual compression: filter/rewrite fragments before context injection.
        # Only runs when compress=True AND a non-Noop compressor is configured.
        was_compressed = False
        if compress and not isinstance(self._compressor, NoopCompressor):
            compressed = self._compressor.compress(query, fragments)
            if len(compressed) != len(fragments) or any(
                c.content != f.content for c, f in zip(compressed, fragments)
            ):
                # Align scores with the (possibly shorter) compressed fragment list
                id_to_score = dict(zip([f.id for f in fragments], scores))
                fragments = compressed
                scores = [id_to_score.get(f.id, 0.0) for f in fragments]
                was_compressed = True
                top_score = scores[0] if scores else 0.0
                gap_report = GapReport.from_score(top_score, len(fragments), spec.min_confidence)
                if gap_report.should_search:
                    hint = " ".join(analysis.keywords[:5]) if analysis.keywords else query
                    gap_report = gap_report.model_copy(update={"search_query_hint": hint})

        return RetrievalResult(
            query=query,
            fragments=fragments,
            scores=scores,
            policy_used=policy,
            spec_used=spec,
            gap_report=gap_report,
            analysis=analysis,
            was_compressed=was_compressed,
        )

    def _sync_index(self, agent_id: str) -> None:
        for fragment in self._store.list_by_agent(agent_id, states=_RETRIEVABLE_STATES):
            if fragment.embedding:
                if self._index.get(fragment.id) is None:
                    self._index.add(fragment.id, fragment.embedding, fragment.content)
            elif self._index.get(fragment.id) is None:
                # Fragment has no embedding (e.g. migrated from JSON store).
                # Compute and persist so it is only done once.
                embedding = self._embedder.embed(fragment.content)
                self._store.update_embedding(fragment.id, embedding)
                self._index.add(fragment.id, embedding, fragment.content)
