# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 Somtochukwu Owen Ikeanyi
"""Scoped helper views over MarkMemory: entities, sessions, blocks, chain.

Each accessor returns a lightweight object bound to this memory. Imports are
deferred to call time because the view modules import MarkMemory themselves
(importing them at module level would create a cycle).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from mark.memory.facade.core import MemoryCore

if TYPE_CHECKING:
    from mark.memory.block_chain import BlockChain
    from mark.memory.block_graph import BlockGraph
    from mark.embeddings.info import EmbedderInfo
    from mark.media.memory import ImageMemory
    from mark.middlewares.media_continuity.entity import CharacterMemory, LocationMemory, ObjectMemory
    from mark.middlewares.media_continuity.session_memory import SessionMemory
    from mark.middlewares.media_continuity.world_bible import WorldBibleMemory


class ViewsMixin(MemoryCore):
    """Factories for long-running-agent helpers and memory-block facades."""

    # ── Long-running agent helpers ────────────────────────────────────────────

    def character(self, name: str) -> "CharacterMemory":
        """Return a CharacterMemory scoped to a named person/agent."""
        from mark.middlewares.media_continuity.entity import CharacterMemory
        return CharacterMemory(name, self)

    def location(self, name: str) -> "LocationMemory":
        """Return a LocationMemory scoped to a named place."""
        from mark.middlewares.media_continuity.entity import LocationMemory
        return LocationMemory(name, self)

    def object(self, name: str) -> "ObjectMemory":
        """Return an ObjectMemory scoped to a named object or artifact."""
        from mark.middlewares.media_continuity.entity import ObjectMemory
        return ObjectMemory(name, self)

    def session(self, session_id: str) -> "SessionMemory":
        """Return a SessionMemory scoped to a specific session context."""
        from mark.middlewares.media_continuity.session_memory import SessionMemory
        return SessionMemory(session_id, self)

    def world_bible(self) -> "WorldBibleMemory":
        """Return a WorldBibleMemory for this agent's canonical facts."""
        from mark.middlewares.media_continuity.world_bible import WorldBibleMemory
        return WorldBibleMemory(self)

    @property
    def images(self) -> "ImageMemory":
        """Image memories (path + hash, captions, continuity checks) for this agent.

        A fresh view without an image embedder or describer; construct
        ``mark.media.ImageMemory(memory, embedder=..., describer=...)`` to add them.
        """
        from mark.media.memory import ImageMemory
        return ImageMemory(self)

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

    # ── embedder visibility ───────────────────────────────────────────────────

    def embedder_info(self) -> "EmbedderInfo":
        """Which embedder this memory uses (name, dim, semantic, source)."""
        from mark.embeddings.info import embedder_info

        return embedder_info(self._embedder)
