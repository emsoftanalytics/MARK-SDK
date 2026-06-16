# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 MARK Contributors
"""Graph neighborhood result types."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from mark.types import MemoryFragment
from mark.types.graph import EdgeRelation, MemoryEdge, MemoryNode, NodeType


@dataclass(frozen=True)
class GraphNeighborhood:
    """
    The local graph neighbourhood around a named concept node.

    Returned by MarkMemory.neighborhood(label, depth=N).

    center    — the node you queried
    nodes     — all nodes reachable within depth hops (includes center)
    edges     — all edges traversed to build the neighbourhood
    fragments — fragments attached to any node in the neighbourhood
    depth     — how many hops were explored

    as_context() formats the neighbourhood as readable text for LLM injection.
    """
    center:    MemoryNode
    nodes:     list[MemoryNode]      = field(default_factory=list)
    edges:     list[MemoryEdge]      = field(default_factory=list)
    fragments: list[MemoryFragment]  = field(default_factory=list)
    depth:     int                   = 2

    def as_context(self, max_chars: int = 4000) -> str:
        """Render the neighbourhood as a text block for LLM context injection."""
        node_map  = {n.id: n for n in self.nodes}
        lines: list[str] = [
            f"<graph_context center={self.center.label!r} "
            f"type={self.center.node_type.value!r} depth={self.depth}>",
        ]

        if self.fragments:
            lines.append("  <evidence>")
            for frag in self.fragments:
                lines.append(f"    - {frag.content}")
            lines.append("  </evidence>")

        if self.edges:
            lines.append("  <relationships>")
            for edge in self.edges:
                src  = node_map.get(edge.source_id)
                tgt  = node_map.get(edge.target_id)
                src_label = src.label  if src  else edge.source_id[:8]
                tgt_label = tgt.label  if tgt  else edge.target_id[:8]
                w = f"{edge.weight:.2f}" if edge.weight != 1.0 else ""
                wstr = f" (w={w})" if w else ""
                lines.append(f"    {src_label} ─[{edge.relation.value}]→ {tgt_label}{wstr}")
            lines.append("  </relationships>")

        other_nodes = [n for n in self.nodes if n.id != self.center.id]
        if other_nodes:
            lines.append("  <related_concepts>")
            for n in other_nodes:
                lines.append(f"    [{n.node_type.value}] {n.label}")
            lines.append("  </related_concepts>")

        lines.append("</graph_context>")
        text = "\n".join(lines)
        return text[:max_chars] if len(text) > max_chars else text

    def node_labels(self) -> list[str]:
        """Return labels of all nodes in the neighborhood."""
        return [n.label for n in self.nodes]

    def fragment_texts(self) -> list[str]:
        """Return contents of all attached fragments."""
        return [f.content for f in self.fragments]
