"""Graph expansion: propagate retrieval scores through the local memory graph.

Seed fragments map to graph nodes; scores flow along expansive edges with a
per-hop decay so a neighbour can never outrank the hit that led to it.
Contradiction edges are reported separately so the reranker can penalise them.
"""
from __future__ import annotations

import heapq

from mark.intelligence.retrieval_policy import _GRAPH_HOP_DECAY
from mark.store.sqlite import LocalMemoryStore
from mark.types import EdgeRelation


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
