"""Per-agent memory facade: store, retrieve, observe, graph, and blocks."""
from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from typing import Any, Dict

from mark.embeddings import EmbeddingProvider
from mark.governance import ConsolidationGate, ConsolidationGateResult
from mark.intelligence import RetrievalPipeline, RetrievalPolicy, RetrievalResult, SessionFilter
from mark.memory.graph import GraphNeighborhood
from mark.intelligence.extractor import DeterministicExtractor, ExtractionMerger, LLMStructuredExtractor
from mark.memory.observe import ObserveEvent, ObserveResult
from mark.plugins import HOOK_OBSERVE_EVENT
from mark.store import LocalMemoryStore
from mark.skills.base import Skill
from mark.types import MemoryEdge, MemoryFragment, MemoryNode, MemoryScope, MemoryState, MemoryTier
from mark.types.graph import EdgeRelation, NodeType
from mark.types.llm import LLMProvider

# Aliases for node type strings that don't map directly to NodeType values.
# Lets callers use domain vocabulary ("episode", "person", "place") without
# knowing the canonical enum names.
_NODE_TYPE_ALIASES: dict[str, str] = {
    "episode": "scene",
    "shot":    "scene",
    "person":  "character",
    "place":   "location",
    "thing":   "object",
}


@dataclass(frozen=True)
class MemoryWriteRejection:
    """Details for the most recent guarded memory write."""

    fragment_id: str
    reason: str
    content: str


def _resolve_node_type(node_type: NodeType | str) -> NodeType:
    if isinstance(node_type, NodeType):
        return node_type
    normalized = node_type.lower()
    canonical  = _NODE_TYPE_ALIASES.get(normalized, normalized)
    try:
        return NodeType(canonical)
    except ValueError:
        return NodeType.ENTITY


class MarkMemory:
    """Per-agent memory facade backed by local SQLite, embeddings, and retrieval."""

    def __init__(
        self,
        *,
        agent_id: str,
        store: LocalMemoryStore,
        pipeline: RetrievalPipeline,
        embedder: EmbeddingProvider,
        executor: ThreadPoolExecutor,
        llm: LLMProvider | None = None,
        consolidation_gate: ConsolidationGate | None = None,
    ) -> None:
        self.agent_id = agent_id
        self._store = store
        self._pipeline = pipeline
        self._embedder = embedder
        self._executor = executor
        self._llm: LLMProvider | None = llm
        self._consolidation_gate = consolidation_gate or ConsolidationGate()
        self._last_rejection: MemoryWriteRejection | None = None

    async def store(
        self,
        content: str,
        *,
        importance: float = 0.5,
        scope: MemoryScope = MemoryScope.AGENT,
        tags: list[str] | None = None,
        source: str | None = None,
        state: MemoryState = MemoryState.UNVERIFIED,
        confidence: float = 1.0,
        session_id: str | None = None,
        ttl_seconds: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Persist the given value."""
        fragment, accepted = self._guarded_fragment(
            content,
            importance=importance,
            scope=scope,
            tags=tags,
            source=source,
            state=state,
            confidence=confidence,
            session_id=session_id,
            ttl_seconds=ttl_seconds,
            metadata=metadata,
        )
        loop = asyncio.get_running_loop()
        if not accepted:
            return await loop.run_in_executor(self._executor, self._store.store, fragment)
        indexed = await loop.run_in_executor(self._executor, self._pipeline.index_fragment, fragment)
        return indexed.id

    def store_sync(
        self,
        content: str,
        *,
        importance: float = 0.5,
        scope: MemoryScope = MemoryScope.AGENT,
        tags: list[str] | None = None,
        source: str | None = None,
        state: MemoryState = MemoryState.UNVERIFIED,
        confidence: float = 1.0,
        session_id: str | None = None,
        ttl_seconds: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Synchronous variant of store()."""
        fragment, accepted = self._guarded_fragment(
            content,
            importance=importance,
            scope=scope,
            tags=tags,
            source=source,
            state=state,
            confidence=confidence,
            session_id=session_id,
            ttl_seconds=ttl_seconds,
            metadata=metadata,
        )
        if not accepted:
            return self._store.store(fragment)
        return self._pipeline.index_fragment(fragment).id

    def quarantine_sync(
        self,
        content: str,
        *,
        reason: str,
        scope: MemoryScope = MemoryScope.AGENT,
        tags: list[str] | None = None,
        source: str | None = None,
        session_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Store a diagnostic artifact outside normal retrieval.

        Quarantined fragments remain inspectable through the store but are not
        indexed into retrieval and cannot become continuity/context by accident.
        """
        meta = {
            **(metadata or {}),
            "mark_guard": {
                "passed": False,
                "reason": reason,
                "quarantined_by": "quarantine_sync",
            },
        }
        fragment = self._fragment(
            content,
            importance=0.0,
            scope=scope,
            tags=[*(tags or []), "mark:quarantined"],
            source=source,
            state=MemoryState.QUARANTINED,
            confidence=0.0,
            session_id=session_id,
            ttl_seconds=None,
            metadata=meta,
        )
        return self._store.store(fragment)

    @property
    def last_rejection(self) -> MemoryWriteRejection | None:
        """Return details for the most recent automatic quarantine."""
        return self._last_rejection

    def quarantined(self, *, limit: int = 100) -> list[MemoryFragment]:
        """List quarantined fragments for this agent."""
        return self._store.list_by_agent(
            self.agent_id,
            states=[MemoryState.QUARANTINED],
            limit=limit,
        )

    def observe(
        self,
        text: str,
        *,
        session_id:  str | None = None,
        importance:  float = 0.5,
        tags:        list[str] | None = None,
        source:      str | None = None,
        metadata:    dict[str, Any] | None = None,
        memory_type: str | None = None,
    ) -> ObserveResult:
        """Store an observation and automatically structure it into memory.

        Always:
          • Stores the text as a fragment.
          • Runs the deterministic extractor (no LLM needed) to find named
            entities, relationships, and generate tags.
          • Creates graph nodes and edges for extracted entities.
          • Applies auto-generated tags to the fragment for filtered retrieval.

        If an LLM is configured (Mark.local(llm=...) or mark.configure_llm(...)):
          • Runs LLM-based extraction in addition (additive, skips duplicates).

        memory_type — override the context node type (default: "scene").
                      Use "episode", "session", "shot", or any NodeType value.
                      Applies to the context entity derived from session_id.

        Falls back gracefully — extraction failures never block the write.
        Returns ObserveResult with fragment_id, nodes, edges, and auto_tags.
        """
        # 1. Deterministic extraction (always runs, no LLM required)
        det_result = DeterministicExtractor().extract(
            text, session_id=session_id, memory_type=memory_type,
        )

        # 2. Optional LLM structured extraction — merges additively with deterministic
        llm_found = False
        if self._llm is not None:
            llm_struct = LLMStructuredExtractor(self._llm).extract_structured(text)
            merged     = ExtractionMerger().merge(det_result, llm_struct)
            llm_found  = bool(llm_struct.entities) or bool(llm_struct.relations)
        else:
            merged = det_result

        # 3. Store fragment — all merged tags (deterministic + LLM) applied at write time
        auto_tags   = merged.tags
        all_tags    = [*(tags or []), *auto_tags]
        fragment_id = self.store_sync(
            text,
            importance = importance,
            tags       = all_tags,
            source     = source or "observe",
            session_id = session_id,
            metadata   = metadata,
        )

        nodes: list[MemoryNode] = []
        edges: list[MemoryEdge] = []
        seen_labels: set[str]   = set()

        def _safe_node(type_str: str, label: str) -> MemoryNode | None:
            if label.lower() in seen_labels:
                return None
            try:
                n = self.node(label, type_str, fragment_id=fragment_id)
                seen_labels.add(label.lower())
                nodes.append(n)
                return n
            except Exception:
                return None

        def _safe_edge(a: str, rel: str, b: str) -> None:
            try:
                edges.append(self.link_nodes(a, b, rel))
            except Exception:
                pass

        # 4. Create nodes and edges from the merged extraction result
        for type_str, label in merged.entities:
            _safe_node(type_str, label)
        for a, rel, b in merged.relations:
            _safe_edge(a, rel, b)

        inferred = bool(merged.entities) or bool(merged.relations) or llm_found

        # 5. Emit observability event
        hook = self._pipeline._plugins.get(HOOK_OBSERVE_EVENT) if hasattr(self._pipeline, "_plugins") else None
        if hook is not None:
            try:
                hook(ObserveEvent(
                    agent_id    = self.agent_id,
                    fragment_id = fragment_id,
                    content     = text,
                    session_id  = session_id,
                    node_labels = [n.label for n in nodes],
                    auto_tags   = auto_tags,
                    edge_count  = len(edges),
                    inferred    = inferred,
                ))
            except Exception:
                pass

        return ObserveResult(
            fragment_id = fragment_id,
            content     = text,
            session_id  = session_id,
            nodes       = nodes,
            edges       = edges,
            auto_tags   = auto_tags,
            inferred    = inferred,
        )

    async def retrieve(
        self,
        query: str,
        policy: RetrievalPolicy = RetrievalPolicy.BALANCED,
        *,
        session_id: str | None = None,
        session_prefix: str | None = None,
        tags: list[str] | None = None,
        tier: MemoryTier | None = None,
        scope: MemoryScope | None = None,
        block_id: str | None = None,
        block_ids: list[str] | None = None,
        escalate_on_gap: bool = False,
        heal_gaps: bool = False,
        web_skill: Skill | None = None,
        compress: bool = False,
    ) -> RetrievalResult:
        """Retrieve memory relevant to the query."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor,
            lambda: self.retrieve_sync(
                query,
                policy,
                session_id=session_id,
                session_prefix=session_prefix,
                tags=tags,
                tier=tier,
                scope=scope,
                block_id=block_id,
                block_ids=block_ids,
                escalate_on_gap=escalate_on_gap,
                heal_gaps=heal_gaps,
                web_skill=web_skill,
                compress=compress,
            ),
        )

    def retrieve_sync(
        self,
        query: str,
        policy: RetrievalPolicy = RetrievalPolicy.BALANCED,
        *,
        session_id: str | None = None,
        session_prefix: str | None = None,
        tags: list[str] | None = None,
        tier: MemoryTier | None = None,
        scope: MemoryScope | None = None,
        block_id: str | None = None,
        block_ids: list[str] | None = None,
        escalate_on_gap: bool = False,
        heal_gaps: bool = False,
        web_skill: Skill | None = None,
        compress: bool = False,
        expand: bool = False,
    ) -> RetrievalResult:
        """Synchronous variant of retrieve()."""
        sf = SessionFilter(
            session_id=session_id, session_prefix=session_prefix,
            tags=tags, tier=tier, scope=scope,
            block_id=block_id, block_ids=block_ids,
        )
        path = ["session" if (session_id or session_prefix) else "agent_namespace"]
        result = replace(
            self._pipeline.retrieve(query, self.agent_id, policy,
                                    session_filter=sf, compress=compress, expand=expand),
            escalation_path=path,
        )

        should_escalate = escalate_on_gap or heal_gaps
        if (
            should_escalate
            and result.gap_report.should_search
            and (session_id or session_prefix)
        ):
            agent_filter = SessionFilter(tags=tags, tier=tier, scope=scope)
            agent_result = replace(
                self._pipeline.retrieve(
                    query,
                    self.agent_id,
                    policy,
                    session_filter=agent_filter,
                    compress=compress,
                ),
                escalation_path=[*path, "agent_namespace"],
            )
            if not agent_result.gap_report.should_search:
                return agent_result
            result = agent_result

        if heal_gaps and result.gap_report.should_search:
            healed = self._lookup_and_store_evidence(
                query,
                result,
                session_id=session_id,
                web_skill=web_skill,
            )
            if healed:
                retry_filter = SessionFilter(
                    session_id=session_id,
                    session_prefix=session_prefix,
                    tags=tags,
                    tier=tier,
                    scope=scope,
                )
                retry = replace(
                    self._pipeline.retrieve(
                        query,
                        self.agent_id,
                        policy,
                        session_filter=retry_filter,
                        compress=compress,
                    ),
                    escalation_path=[*result.escalation_path, "external_lookup", "session_retry"],
                    external_lookup_used=True,
                )
                if not retry.gap_report.should_search:
                    return retry

                agent_retry = replace(
                    self._pipeline.retrieve(
                        query,
                        self.agent_id,
                        policy,
                        session_filter=SessionFilter(tags=tags, tier=tier, scope=scope),
                        compress=compress,
                    ),
                    escalation_path=[*result.escalation_path, "external_lookup", "agent_namespace_retry"],
                    external_lookup_used=True,
                )
                return agent_retry

        return result

    def _lookup_and_store_evidence(
        self,
        query: str,
        result: RetrievalResult,
        *,
        session_id: str | None,
        web_skill: Skill | None,
    ) -> bool:
        """Run an explicit local web lookup and persist attributed evidence.

        Local lookup is intentionally basic. Smart browser automation belongs
        to MARK Cloud where an LLM, source policy, and observability can guide it.
        """
        skill = web_skill
        if skill is None:
            try:
                from mark.skills.web_search import WebSearchSkill
                skill = WebSearchSkill()
            except Exception:
                return False

        search_query = result.gap_report.search_query_hint or query
        lookup = skill.run_sync(
            {"query": search_query, "mode": "search"},
            {"agent_id": self.agent_id},
        )
        if not lookup.success or not lookup.fragments:
            return False

        stored = False
        for fragment in lookup.fragments:
            if not isinstance(fragment, MemoryFragment):
                continue
            metadata = dict(fragment.metadata)
            metadata.update(
                {
                    "mark_acquired_via": "gap_healing",
                    "mark_original_query": query,
                    "mark_search_query": search_query,
                }
            )
            tags = sorted(set([*fragment.tags, "external_evidence", "gap_healing"]))
            evidence = fragment.model_copy(
                update={
                    "agent_id": self.agent_id,
                    "session_id": session_id or fragment.session_id,
                    "scope": MemoryScope.AGENT,
                    "state": MemoryState.UNVERIFIED,
                    "tags": tags,
                    "metadata": metadata,
                }
            )
            self._pipeline.index_fragment(evidence)
            stored = True
        return stored

    def list(
        self,
        states: list[MemoryState] | None = None,
        limit: int = 100,
    ) -> list[MemoryFragment]:
        """Return this agent's fragments, optionally filtered by state."""
        return self._store.list_by_agent(self.agent_id, states=states, limit=limit)

    def list_sessions(self) -> list[str]:
        """Return all session_ids used by this agent, ordered by most recent activity."""
        return self._store.list_sessions(self.agent_id)

    def list_by_session(
        self,
        session_id: str,
        *,
        states: list[MemoryState] | None = None,
        limit: int = 200,
    ) -> list[MemoryFragment]:
        """Return all fragments for a specific session of this agent."""
        return self._store.list_by_agent(
            self.agent_id,
            states=states,
            session_id=session_id,
            limit=limit,
        )

    def forget(self, fragment_id: str) -> bool:
        """Delete a fragment by id."""
        self._pipeline._index.remove(fragment_id)
        return self._store.delete(fragment_id)

    def promote(self, fragment_id: str) -> None:
        """Promote a fragment toward PROMOTED state."""
        fragment = self._store.get(fragment_id)
        if fragment is None:
            raise KeyError(fragment_id)
        current = fragment
        if current.state == MemoryState.UNVERIFIED:
            current = current.transition_to(MemoryState.VERIFIED)
        if current.state == MemoryState.VERIFIED:
            current = current.transition_to(MemoryState.PROMOTED)
        self._store.store(current)

    # ── graph API ─────────────────────────────────────────────────────────────

    def node(
        self,
        label: str,
        node_type: NodeType | str = NodeType.CONCEPT,
        *,
        fragment_id: str | None = None,
        weight: float = 1.0,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryNode:
        """Get an existing node by label or create a new one.

        If the node already exists for this agent, its stored version is returned
        unchanged. Pass fragment_id to attach the node to a specific fragment
        so that graph expansion during retrieval can reach it.
        """
        nt = _resolve_node_type(node_type)
        existing = self._store.get_node_by_label(label, self.agent_id)
        if existing is not None:
            return existing
        new_node = MemoryNode(
            label       = label,
            node_type   = nt,
            agent_id    = self.agent_id,
            fragment_id = fragment_id,
            weight      = weight,
            metadata    = metadata or {},
        )
        self._store.store_node(new_node)
        return new_node

    def attach_node(
        self,
        fragment_id: str,
        label: str,
        node_type: NodeType | str = NodeType.ENTITY,
        *,
        weight: float = 1.0,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryNode:
        """Associate a named concept node with a fragment.

        If the node already exists, this creates a second node record pointing
        to the new fragment (one concept can have multiple supporting fragments).
        Use node() for pure get-or-create without a fragment attachment.
        """
        nt = _resolve_node_type(node_type)
        new_node = MemoryNode(
            label       = label,
            node_type   = nt,
            agent_id    = self.agent_id,
            fragment_id = fragment_id,
            weight      = weight,
            metadata    = metadata or {},
        )
        self._store.store_node(new_node)
        return new_node

    def link_nodes(
        self,
        label_a: str,
        label_b: str,
        relation: EdgeRelation | str,
        *,
        weight: float = 1.0,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryEdge:
        """Create a directed edge between two named nodes.

        Both nodes are created (via node()) if they do not already exist.
        Duplicate edges (same source, target, relation) are allowed — each
        write creates a new edge record. Use unlink() to remove existing edges
        first if you want idempotent linking.
        """
        rel    = EdgeRelation(relation) if isinstance(relation, str) else relation
        node_a = self.node(label_a)
        node_b = self.node(label_b)
        edge   = MemoryEdge(
            source_id = node_a.id,
            target_id = node_b.id,
            relation  = rel,
            agent_id  = self.agent_id,
            weight    = weight,
            metadata  = metadata or {},
        )
        self._store.store_edge(edge)
        return edge

    def unlink(
        self,
        label_a: str,
        label_b: str,
        relation: EdgeRelation | str | None = None,
    ) -> int:
        """Remove directed edges from label_a → label_b.

        relation=None removes ALL relations between the two nodes.
        Returns the number of edge rows deleted.
        """
        node_a = self._store.get_node_by_label(label_a, self.agent_id)
        node_b = self._store.get_node_by_label(label_b, self.agent_id)
        if node_a is None or node_b is None:
            return 0
        rel = (EdgeRelation(relation) if isinstance(relation, str) else relation)
        return self._store.delete_edges_between(node_a.id, node_b.id, rel)

    def neighborhood(self, label: str, *, depth: int = 2) -> GraphNeighborhood:
        """Return the graph neighbourhood around a named concept node.

        Traverses both outgoing and incoming edges up to depth hops and
        collects all reachable nodes, edges, and their attached fragments.
        Raises KeyError if the node is not found for this agent.
        """
        center = self._store.get_node_by_label(label, self.agent_id)
        if center is None:
            raise KeyError(f"Node not found: {label!r} for agent {self.agent_id!r}")

        visited_nodes: dict[str, MemoryNode] = {center.id: center}
        visited_edges: dict[str, MemoryEdge] = {}
        frontier: list[tuple[str, int]] = [(center.id, 0)]

        while frontier:
            node_id, d = frontier.pop(0)
            if d >= depth:
                continue
            # Outgoing edges
            for edge in self._store.edges_from_node(node_id):
                if edge.id not in visited_edges:
                    visited_edges[edge.id] = edge
                if edge.target_id not in visited_nodes:
                    tgt = self._store.get_node(edge.target_id)
                    if tgt and tgt.agent_id in {None, self.agent_id}:
                        visited_nodes[tgt.id] = tgt
                        frontier.append((tgt.id, d + 1))
            # Incoming edges (undirected neighbourhood)
            if hasattr(self._store, "edges_to_node"):
                for edge in self._store.edges_to_node(node_id):
                    if edge.id not in visited_edges:
                        visited_edges[edge.id] = edge
                    if edge.source_id not in visited_nodes:
                        src = self._store.get_node(edge.source_id)
                        if src and src.agent_id in {None, self.agent_id}:
                            visited_nodes[src.id] = src
                            frontier.append((src.id, d + 1))

        # Gather fragments attached to any node in the neighbourhood
        frags: list[MemoryFragment] = []
        seen_frag_ids: set[str] = set()
        for node in visited_nodes.values():
            if node.fragment_id and node.fragment_id not in seen_frag_ids:
                frag = self._store.get(node.fragment_id)
                if frag is not None:
                    frags.append(frag)
                    seen_frag_ids.add(node.fragment_id)

        return GraphNeighborhood(
            center    = center,
            nodes     = list(visited_nodes.values()),
            edges     = list(visited_edges.values()),
            fragments = frags,
            depth     = depth,
        )

    # ── Long-running agent helpers ────────────────────────────────────────────

    def character(self, name: str) -> "CharacterMemory":
        """Return a CharacterMemory scoped to a named person/agent."""
        from mark.media.entity import CharacterMemory
        return CharacterMemory(name, self)

    def location(self, name: str) -> "LocationMemory":
        """Return a LocationMemory scoped to a named place."""
        from mark.media.entity import LocationMemory
        return LocationMemory(name, self)

    def object(self, name: str) -> "ObjectMemory":
        """Return an ObjectMemory scoped to a named object or artifact."""
        from mark.media.entity import ObjectMemory
        return ObjectMemory(name, self)

    def session(self, session_id: str) -> "SessionMemory":
        """Return a SessionMemory scoped to a specific session context."""
        from mark.media.session_memory import SessionMemory
        return SessionMemory(session_id, self)

    def world_bible(self) -> "WorldBibleMemory":
        """Return a WorldBibleMemory for this agent's canonical facts."""
        from mark.media.world_bible import WorldBibleMemory
        return WorldBibleMemory(self)

    # ── memory blocks ─────────────────────────────────────────────────────────

    def blocks(self) -> "BlockGraph":
        """Return the BlockGraph facade for graph-scoped memory blocks.

        Blocks group nodes and the edges between them per topic, session, or
        world-bible scope, and connect to other blocks through forward and
        backward links::

            graph  = memory.blocks()
            block  = graph.create_block("ep01", session_id="season-01/ep-01")
            graph.add_node(block.id, node.id)
            graph.link(block.id, other.id, relation="follows")
        """
        from mark.memory.block_graph import BlockGraph
        return BlockGraph(self._store, self.agent_id)

    def chain(self) -> "BlockChain":
        """Return the BlockChain facade for sealing and verifying blocks.

        Sealing hashes a block's members and links it to the previously
        sealed block, producing a tamper-evident local chain::

            chain  = memory.chain()
            sealed = chain.seal(block.id)
            report = chain.verify(block.id)      # pinpoints corrupted members
            health = chain.verify_chain()        # whole-chain integrity
            chain.quarantine(block.id)           # isolate without side effects
        """
        from mark.memory.block_chain import BlockChain
        return BlockChain(self._store, self.agent_id)

    def clear(self) -> int:
        """Delete all of this agent's fragments and index entries."""
        fragments = self._store.list_by_agent(self.agent_id)
        for fragment in fragments:
            self._pipeline._index.remove(fragment.id)
        self._store.clear(self.agent_id)
        return len(fragments)

    def _fragment(
        self,
        content: str,
        *,
        importance: float,
        scope: MemoryScope,
        tags: list[str] | None,
        source: str | None,
        state: MemoryState,
        confidence: float,
        session_id: str | None,
        ttl_seconds: int | None,
        metadata: dict[str, Any] | None,
    ) -> MemoryFragment:
        return MemoryFragment(
            content=content,
            agent_id=self.agent_id,
            importance=importance,
            scope=scope,
            tags=tags or [],
            source=source,
            state=state,
            confidence=confidence,
            session_id=session_id,
            ttl_seconds=ttl_seconds,
            metadata=metadata or {},
        )

    def _guarded_fragment(
        self,
        content: str,
        *,
        importance: float,
        scope: MemoryScope,
        tags: list[str] | None,
        source: str | None,
        state: MemoryState,
        confidence: float,
        session_id: str | None,
        ttl_seconds: int | None,
        metadata: dict[str, Any] | None,
    ) -> tuple[MemoryFragment, bool]:
        if state == MemoryState.QUARANTINED:
            fragment = self._fragment(
                content,
                importance=0.0,
                scope=scope,
                tags=[*(tags or []), "mark:quarantined"],
                source=source,
                state=state,
                confidence=0.0,
                session_id=session_id,
                ttl_seconds=ttl_seconds,
                metadata=metadata,
            )
            return fragment, False

        gate_result: ConsolidationGateResult = self._consolidation_gate.check(
            content,
            confidence=confidence,
        )
        if gate_result.passed:
            self._last_rejection = None
            return self._fragment(
                gate_result.content,
                importance=importance,
                scope=scope,
                tags=tags,
                source=source,
                state=state,
                confidence=confidence,
                session_id=session_id,
                ttl_seconds=ttl_seconds,
                metadata=metadata,
            ), True

        meta = {
            **(metadata or {}),
            "mark_guard": {
                "passed": False,
                "reason": gate_result.reason,
                "original_importance": importance,
                "requested_state": state.value,
            },
        }
        fragment = self._fragment(
            gate_result.content,
            importance=0.0,
            scope=scope,
            tags=[*(tags or []), "mark:quarantined", "mark:failure"],
            source=source,
            state=MemoryState.QUARANTINED,
            confidence=0.0,
            session_id=session_id,
            ttl_seconds=ttl_seconds,
            metadata=meta,
        )
        self._last_rejection = MemoryWriteRejection(
            fragment_id=fragment.id,
            reason=gate_result.reason,
            content=gate_result.content,
        )
        return fragment, False
