# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 MARK Contributors
"""Which ids are protected by sealed blocks.

Maintenance jobs (deduplication, consolidation, pruning, working-memory
expiry) call :func:`sealed_membership` before deleting or rewriting memory,
so they never silently break a block's hash chain.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mark.memory.block_hashing import _MEMBER_HASHES_KEY
from mark.types import BlockStatus


@dataclass(frozen=True)
class SealedMembership:
    """Ids protected by sealed blocks (sealed, or quarantined after sealing).

    Maintenance jobs (deduplication, consolidation, pruning) must not delete
    or rewrite anything listed here, or they would silently break the chain.
    """

    block_ids: frozenset[str] = frozenset()
    fragment_ids: frozenset[str] = frozenset()
    node_ids: frozenset[str] = frozenset()
    edge_ids: frozenset[str] = frozenset()

    def __bool__(self) -> bool:
        return bool(self.block_ids)

    def protects_fragment(self, fragment: Any) -> bool:
        """True when the fragment is a sealed member or lives in a sealed block."""
        return (
            getattr(fragment, "id", None) in self.fragment_ids
            or getattr(fragment, "block_id", None) in self.block_ids
        )

    def protects_node(self, node: Any) -> bool:
        """True when the node is a member of a sealed block."""
        return getattr(node, "id", None) in self.node_ids

    def protects_edge(self, edge: Any) -> bool:
        """True when the edge is sealed or scoped to a sealed block."""
        return (
            getattr(edge, "id", None) in self.edge_ids
            or getattr(edge, "block_id", None) in self.block_ids
        )


def sealed_membership(store: Any, agent_id: str) -> SealedMembership:
    """Collect every block/fragment/node/edge id protected by a seal.

    Works with any store exposing ``list_blocks``; stores without block
    support return an empty membership.
    """
    list_blocks = getattr(store, "list_blocks", None)
    if list_blocks is None:
        return SealedMembership()
    try:
        blocks = list_blocks(agent_id)
    except Exception:
        return SealedMembership()
    block_ids: set[str] = set()
    fragment_ids: set[str] = set()
    node_ids: set[str] = set()
    edge_ids: set[str] = set()
    for block in blocks:
        if block.content_hash is None and block.status != BlockStatus.SEALED:
            continue
        block_ids.add(block.id)
        member_ids = set(dict(block.metadata.get(_MEMBER_HASHES_KEY, {})))
        for frag in _safe_call(store, "fragments_for_block", block.id):
            fragment_ids.add(frag.id)
        for node in _safe_call(store, "nodes_in_block", block.id):
            node_ids.add(node.id)
        for edge in _safe_call(store, "edges_in_block", block.id):
            edge_ids.add(edge.id)
        # Sealed members that have since moved/been removed stay protected.
        for member_id in member_ids:
            if member_id not in node_ids and member_id not in edge_ids:
                fragment_ids.add(member_id)
    return SealedMembership(
        block_ids=frozenset(block_ids),
        fragment_ids=frozenset(fragment_ids),
        node_ids=frozenset(node_ids),
        edge_ids=frozenset(edge_ids),
    )


def _safe_call(store: Any, name: str, *args: Any) -> list[Any]:
    method = getattr(store, name, None)
    if method is None:
        return []
    try:
        return list(method(*args))
    except Exception:
        return []
