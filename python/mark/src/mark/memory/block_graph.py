# SPDX-License-Identifier: MIT
# Copyright (c) 2025 MARK Contributors
"""Graph-scoped memory blocks.

A MemoryBlock groups nodes and the edges between them into one bounded,
inspectable unit. Blocks can represent a topic within a session, an entire
session, or a world-bible scope, and they connect to other blocks through
directed BlockLinks — traversable forward and backward exactly like nodes
are traversed through edges.

Hierarchy::

    fragment  → evidence
    node      → semantic anchor built from fragments
    block     → bounded set of nodes plus the edges between them
    chain     → blocks linked forward/backward per agent (see block_chain)

Node membership is many-to-many: a canonical node such as ``Elena`` may
belong to many blocks without being duplicated. Sealing, hash verification,
and quarantine live in :mod:`mark.memory.block_chain`.
"""

from __future__ import annotations

from typing import Any

from mark.store import LocalMemoryStore
from mark.types import BlockLink, BlockStatus, MemoryBlock, MemoryEdge, MemoryNode, MemoryScope
from mark.types.graph import EdgeRelation


class BlockGraph:
    """Per-agent facade for creating, populating, linking, and traversing blocks.

    Obtain one through :meth:`mark.memory.mark_memory.MarkMemory.blocks` or
    construct directly over a :class:`LocalMemoryStore`::

        graph = BlockGraph(store, agent_id="video-agent")
        block = graph.create_block("ep01-warehouse", session_id="season-01/ep-01")
        graph.add_node(block.id, node.id)
        graph.link(block.id, next_block.id, relation="follows")
    """

    def __init__(self, store: LocalMemoryStore, agent_id: str) -> None:
        self._store = store
        self.agent_id = agent_id

    # ── block lifecycle ───────────────────────────────────────────────────────

    def create_block(
        self,
        name: str,
        *,
        session_id: str | None = None,
        block_type: str = "ltm",
        scope: MemoryScope = MemoryScope.AGENT,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryBlock:
        """Create and persist a new OPEN block owned by this agent.

        block_type is a domain hint ("ltm", "working", "global", "world-bible",
        or any custom label); it does not change block behaviour locally.
        """
        block = MemoryBlock(
            name=name,
            agent_id=self.agent_id,
            session_id=session_id,
            block_type=block_type,
            scope=scope,
            metadata=metadata or {},
        )
        self._store.store_block(block)
        return block

    def get(self, block_id: str) -> MemoryBlock | None:
        """Fetch a block by id (any status)."""
        return self._store.get_block(block_id)

    def by_name(self, name: str) -> MemoryBlock | None:
        """Fetch this agent's block by name."""
        return self._store.get_block_by_name(name, self.agent_id)

    def list(self, *, status: BlockStatus | None = None) -> list[MemoryBlock]:
        """List this agent's blocks, optionally filtered by lifecycle status."""
        if status is not None:
            return self._store.list_blocks_by_status(self.agent_id, status)
        return self._store.list_blocks(self.agent_id)

    # ── membership ────────────────────────────────────────────────────────────

    def add_fragment(self, block_id: str, fragment_id: str) -> None:
        """Assign a fragment to a block (a fragment belongs to one block).

        Raises ValueError if the block is not OPEN — sealed blocks are
        immutable and quarantined blocks must be released first.
        """
        self._require_open(block_id)
        self._store.assign_fragment_block(fragment_id, block_id)

    def add_node(self, block_id: str, node_id: str) -> None:
        """Add a node to a block (idempotent; nodes may belong to many blocks)."""
        self._require_open(block_id)
        self._store.add_node_to_block(block_id, node_id)

    def add_nodes(self, block_id: str, node_ids: list[str]) -> None:
        """Add several nodes to a block in one call."""
        self._require_open(block_id)
        for node_id in node_ids:
            self._store.add_node_to_block(block_id, node_id)

    def remove_node(self, block_id: str, node_id: str) -> bool:
        """Remove a node's membership from an OPEN block."""
        self._require_open(block_id)
        return self._store.remove_node_from_block(block_id, node_id)

    def nodes(self, block_id: str) -> list[MemoryNode]:
        """Return the member nodes of a block."""
        return self._store.nodes_in_block(block_id)

    def blocks_for_node(self, node_id: str) -> list[MemoryBlock]:
        """Return every block a node belongs to."""
        blocks = []
        for block_id in self._store.block_ids_for_node(node_id):
            block = self._store.get_block(block_id)
            if block is not None:
                blocks.append(block)
        return blocks

    # ── block-scoped edges ────────────────────────────────────────────────────

    def add_edge(
        self,
        block_id: str,
        source_node_id: str,
        target_node_id: str,
        relation: EdgeRelation | str = EdgeRelation.RELATED_TO,
        *,
        weight: float = 1.0,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryEdge:
        """Create a directed edge between two member nodes, scoped to the block.

        Both endpoints must already be members of the block; this keeps a
        block's internal edge set self-contained so it can be hashed, verified,
        and quarantined as one unit.
        """
        self._require_open(block_id)
        members = {node.id for node in self._store.nodes_in_block(block_id)}
        missing = {source_node_id, target_node_id} - members
        if missing:
            raise ValueError(
                f"Nodes must be members of block {block_id!r} before linking: {sorted(missing)}"
            )
        rel = EdgeRelation(relation) if isinstance(relation, str) else relation
        edge = MemoryEdge(
            source_id=source_node_id,
            target_id=target_node_id,
            relation=rel,
            agent_id=self.agent_id,
            weight=weight,
            block_id=block_id,
            metadata=metadata or {},
        )
        self._store.store_edge(edge)
        return edge

    def edges(self, block_id: str) -> list[MemoryEdge]:
        """Return the edges scoped to a block."""
        return self._store.edges_in_block(block_id)

    # ── block links (inter-block graph) ───────────────────────────────────────

    def link(
        self,
        source_block_id: str,
        target_block_id: str,
        relation: str = "follows",
        *,
        weight: float = 1.0,
        metadata: dict[str, Any] | None = None,
    ) -> BlockLink:
        """Create a directed link from one block to another.

        Links work like edges between nodes: traverse() follows them forward
        (source → target) and backward (target → source).
        """
        if source_block_id == target_block_id:
            raise ValueError("A block cannot link to itself")
        for block_id in (source_block_id, target_block_id):
            if self._store.get_block(block_id) is None:
                raise KeyError(f"Block not found: {block_id!r}")
        link = BlockLink(
            source_block_id=source_block_id,
            target_block_id=target_block_id,
            relation=relation,
            agent_id=self.agent_id,
            weight=weight,
            metadata=metadata or {},
        )
        self._store.store_block_link(link)
        return link

    def links_forward(self, block_id: str) -> list[BlockLink]:
        """Outgoing links (this block → others)."""
        return self._store.links_from_block(block_id)

    def links_backward(self, block_id: str) -> list[BlockLink]:
        """Incoming links (others → this block)."""
        return self._store.links_to_block(block_id)

    def unlink(self, link_id: str) -> bool:
        """Delete a block link by id."""
        return self._store.delete_block_link(link_id)

    def traverse(
        self,
        block_id: str,
        *,
        direction: str = "forward",
        depth: int = 3,
        include_quarantined: bool = False,
    ) -> list[MemoryBlock]:
        """Walk block links breadth-first from a starting block.

        direction  "forward" (source → target), "backward", or "both"
        depth      maximum hops from the starting block
        include_quarantined
                   quarantined blocks are skipped by default so corrupted
                   memory does not travel through traversal results

        Returns reached blocks in BFS order, excluding the start block.
        """
        if direction not in {"forward", "backward", "both"}:
            raise ValueError("direction must be 'forward', 'backward', or 'both'")
        visited: set[str] = {block_id}
        ordered: list[MemoryBlock] = []
        frontier: list[tuple[str, int]] = [(block_id, 0)]
        while frontier:
            current_id, hops = frontier.pop(0)
            if hops >= depth:
                continue
            neighbor_ids: list[str] = []
            if direction in {"forward", "both"}:
                neighbor_ids += [l.target_block_id for l in self._store.links_from_block(current_id)]
            if direction in {"backward", "both"}:
                neighbor_ids += [l.source_block_id for l in self._store.links_to_block(current_id)]
            for neighbor_id in neighbor_ids:
                if neighbor_id in visited:
                    continue
                visited.add(neighbor_id)
                block = self._store.get_block(neighbor_id)
                if block is None:
                    continue
                if block.status == BlockStatus.QUARANTINED and not include_quarantined:
                    continue
                ordered.append(block)
                frontier.append((neighbor_id, hops + 1))
        return ordered

    # ── internal ──────────────────────────────────────────────────────────────

    def _require_open(self, block_id: str) -> MemoryBlock:
        block = self._store.get_block(block_id)
        if block is None:
            raise KeyError(f"Block not found: {block_id!r}")
        if block.status != BlockStatus.OPEN:
            raise ValueError(
                f"Block {block.name!r} is {block.status.value}; only OPEN blocks accept writes"
            )
        return block
