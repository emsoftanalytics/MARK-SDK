# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 MARK Contributors — v1.3 plasticity
#
# MemoryPruner — maintenance job that culls forgotten fragments and
# dissolves weak edges. Run nightly or after N agent interactions.
#
# Hook: HOOK_PLASTICITY activates a registered plasticity plugin.
"""Decay-driven pruning of fragments and dissolution of weak edges."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .decay   import ExponentialDecay, TemporalScope
from .hebbian import EdgeCoActivation


@dataclass
class PrunerStats:
    """Counts produced by one pruning pass."""
    fragments_pruned:   int = 0
    edges_dissolved:    int = 0
    importance_updated: int = 0


class MemoryPruner:
    """
    Applies decay, updates importances, prunes forgotten fragments,
    and dissolves weak edges.

    Steps per run(agent_id):
    1. List all fragments for the agent.
    2. Apply ExponentialDecay to each.
    3. Update importance in the store.
    4. Delete fragments at floor (should_prune=True) — except PROMOTED.
    5. Dissolve MemoryEdges below EDGE_FLOOR.

    Typical usage:
        pruner = MemoryPruner(store)
        stats  = pruner.run("my-agent")

    Media agents: run after each episode (not each scene) to preserve
    character/world facts while discarding stale scene notes.
    PROMOTED fragments (character bible, world rules) are always exempt.
    """

    def __init__(self, store: object,
                 decay: Optional[ExponentialDecay] = None,
                 hebbian: Optional[object] = None) -> None:
        self._store  = store
        self._decay  = decay or ExponentialDecay()
        self._coact  = EdgeCoActivation()

    def run(self, agent_id: str) -> PrunerStats:
        """Execute with the given input and context."""
        stats = PrunerStats()
        store = self._store

        # Fragment decay + pruning
        fragments = (store.list_by_agent(agent_id)  # type: ignore[union-attr]
                     if hasattr(store, "list_by_agent") else [])

        for frag in fragments:
            new_imp = self._decay.apply(frag)
            if abs(new_imp - float(getattr(frag, "importance", 0.5))) > 1e-6:
                if hasattr(store, "update_importance"):
                    store.update_importance(frag.id, new_imp)  # type: ignore[union-attr]
                elif hasattr(store, "update_state"):
                    pass  # importance update not supported; skip
                stats.importance_updated += 1

            if self._decay.should_prune(frag):
                from mark.types import MemoryState
                if getattr(frag, "state", None) != MemoryState.PROMOTED:
                    if hasattr(store, "delete"):
                        store.delete(frag.id)  # type: ignore[union-attr]
                    stats.fragments_pruned += 1

        # Edge dissolution
        if hasattr(store, "list_edges"):
            for edge in store.list_edges(agent_id=agent_id):  # type: ignore[union-attr]
                if self._coact.should_dissolve(float(getattr(edge, "weight", 1.0))):
                    if hasattr(store, "delete_edge"):
                        store.delete_edge(edge.id)  # type: ignore[union-attr]
                    stats.edges_dissolved += 1

        return stats
