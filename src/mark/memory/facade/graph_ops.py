# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 MARK Contributors
"""Graph API of MarkMemory: concept nodes, edges, and neighbourhoods.

Nodes are looked up by label within this agent. ``node()`` is get-or-create,
``attach_node()`` always writes a new node record (one concept can be backed
by many fragments), and ``link_nodes()`` creates both ends on demand.
"""
from __future__ import annotations

from typing import Any

from mark.memory.facade.core import MemoryCore
from mark.memory.graph import GraphNeighborhood
from mark.types import MemoryEdge, MemoryFragment, MemoryNode
from mark.types.graph import EdgeRelation, NodeType

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


def _resolve_node_type(node_type: NodeType | str) -> NodeType:
    """Map an enum or (aliased, case-insensitive) string to a NodeType.

    Unknown strings fall back to ``NodeType.ENTITY`` instead of raising, so
    free-form extractor output can always be stored.
    """
    if isinstance(node_type, NodeType):
        return node_type
    normalized = node_type.lower()
    canonical  = _NODE_TYPE_ALIASES.get(normalized, normalized)
    try:
        return NodeType(canonical)
    except ValueError:
        return NodeType.ENTITY


class GraphMixin(MemoryCore):
    """Per-agent knowledge-graph operations."""

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

        # Breadth-first walk. Nodes owned by other agents are never entered
        # (shared nodes with agent_id=None are), keeping the view agent-local.
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
            # Incoming edges (undirected neighbourhood); optional store API.
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
