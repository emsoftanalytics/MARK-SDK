# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 Somtochukwu Owen Ikeanyi
"""Graph delegation for SimpleMemory.

The simple API exposes the knowledge-graph helpers of the reserved simple
agent's :class:`~mark.memory.mark_memory.MarkMemory` unchanged; this mixin
only forwards calls (with a narrower keyword surface).
"""
from __future__ import annotations

from typing import Any

from mark.memory.graph import GraphNeighborhood
from mark.types.graph import EdgeRelation, NodeType


class SimpleGraphMixin:
    """Forward graph operations to ``self._mark_memory``."""

    _mark_memory: Any

    def node(
        self,
        label: str,
        node_type: "NodeType | str" = NodeType.CONCEPT,
        *,
        fragment_id: str | None = None,
        weight: float = 1.0,
    ) -> "Any":
        """Get or create a named concept node in the simple agent's graph."""
        return self._mark_memory.node(label, node_type,
                                      fragment_id=fragment_id, weight=weight)

    def attach_node(
        self,
        fragment_id: str,
        label: str,
        node_type: "NodeType | str" = NodeType.ENTITY,
        *,
        weight: float = 1.0,
    ) -> "Any":
        """Associate a fragment with a named concept node."""
        return self._mark_memory.attach_node(fragment_id, label, node_type,
                                             weight=weight)

    def link_nodes(
        self,
        label_a: str,
        label_b: str,
        relation: "EdgeRelation | str",
        *,
        weight: float = 1.0,
    ) -> "Any":
        """Create a directed edge between two named nodes."""
        return self._mark_memory.link_nodes(label_a, label_b, relation, weight=weight)

    def unlink(
        self,
        label_a: str,
        label_b: str,
        relation: "EdgeRelation | str | None" = None,
    ) -> int:
        """Remove directed edges from label_a → label_b. Returns count removed."""
        return self._mark_memory.unlink(label_a, label_b, relation)

    def neighborhood(self, label: str, *, depth: int = 2) -> GraphNeighborhood:
        """Return the graph neighbourhood around a named concept node."""
        return self._mark_memory.neighborhood(label, depth=depth)
