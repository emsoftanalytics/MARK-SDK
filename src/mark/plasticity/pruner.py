# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 Somtochukwu Owen Ikeanyi — v1.3 plasticity
#
# MemoryPruner — maintenance job that culls forgotten fragments and
# dissolves weak edges. Run nightly or after N agent interactions.
#
# Hook: HOOK_PLASTICITY activates a registered plasticity plugin.
"""Decay-driven pruning of fragments and dissolution of weak edges."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional

from .decay   import ExponentialDecay, TemporalScope
from .hebbian import EdgeCoActivation


@dataclass
class PrunerStats:
    """Counts produced by one pruning pass."""
    fragments_pruned:   int = 0
    edges_dissolved:    int = 0
    importance_updated: int = 0
    protected:          int = 0


def _list_all(store: Any, agent_id: str) -> list[Any]:
    """List every fragment (no 200-row cap); tolerate stores without ``limit``."""
    try:
        return list(store.list_by_agent(agent_id, limit=None))
    except TypeError:
        return list(store.list_by_agent(agent_id))


class MemoryPruner:
    """
    Applies decay, updates importances, prunes forgotten fragments,
    and dissolves weak edges.

    Steps per run(agent_id):
    1. List all fragments for the agent (uncapped).
    2. Skip protected fragments: PROMOTED (forget gate), QUARANTINED and
       CONTRADICTED (audit evidence), and members of SEALED blocks.
    3. Apply ExponentialDecay relative to the last decay (idempotent).
    4. Persist importance plus ``mark_decay`` bookkeeping in metadata.
    5. Delete fragments at the floor.
    6. Dissolve MemoryEdges below the edge floor (never sealed edges).

    Typical usage:
        pruner = MemoryPruner(store)
        stats  = pruner.run("my-agent")

    ``hebbian`` may be an :class:`EdgeCoActivation` (or any object with
    ``should_dissolve(weight) -> bool``) to customise edge dissolution.

    Media agents: run after each episode (not each scene) to preserve
    character/world facts while discarding stale scene notes.
    PROMOTED fragments (character bible, world rules) are always exempt.
    """

    def __init__(self, store: object,
                 decay: Optional[ExponentialDecay] = None,
                 hebbian: Optional[object] = None) -> None:
        self._store  = store
        self._decay  = decay or ExponentialDecay()
        if hebbian is not None and not hasattr(hebbian, "should_dissolve"):
            raise TypeError("hebbian must provide should_dissolve(weight) -> bool")
        self._coact  = hebbian if hebbian is not None else EdgeCoActivation()

    def run(self, agent_id: str, *, now: Optional[datetime] = None) -> PrunerStats:
        """Run one maintenance pass for ``agent_id`` (``now`` defaults to the current time)."""
        from mark.memory.block_chain import sealed_membership
        from mark.types import MemoryState

        stats = PrunerStats()
        store = self._store
        now   = now or datetime.now(timezone.utc)
        sealed = sealed_membership(store, agent_id)
        protected_states = {MemoryState.PROMOTED, MemoryState.QUARANTINED, MemoryState.CONTRADICTED}

        # Fragment decay + pruning
        fragments = _list_all(store, agent_id) if hasattr(store, "list_by_agent") else []

        for frag in fragments:
            if getattr(frag, "state", None) in protected_states or sealed.protects_fragment(frag):
                stats.protected += 1
                continue

            new_imp = self._decay.apply(frag, now=now)
            if abs(new_imp - float(getattr(frag, "importance", 0.5))) > 1e-6:
                self._persist_decay(frag, new_imp, now)
                stats.importance_updated += 1

            if new_imp <= self._decay._floor:
                if hasattr(store, "delete"):
                    store.delete(frag.id)  # type: ignore[union-attr]
                stats.fragments_pruned += 1

        # Edge dissolution
        if hasattr(store, "list_edges"):
            for edge in store.list_edges(agent_id=agent_id):  # type: ignore[union-attr]
                if sealed.protects_edge(edge):
                    continue
                if self._coact.should_dissolve(float(getattr(edge, "weight", 1.0))):
                    if hasattr(store, "delete_edge"):
                        store.delete_edge(edge.id)  # type: ignore[union-attr]
                    stats.edges_dissolved += 1

        return stats

    def _persist_decay(self, frag: Any, new_imp: float, now: datetime) -> None:
        store = self._store
        # Full-fragment stores record decay bookkeeping so re-runs are idempotent.
        if hasattr(store, "store") and hasattr(frag, "model_copy"):
            updated = frag.model_copy(update={
                "importance": new_imp,
                "metadata": self._decay.decay_metadata(frag, now=now),
            })
            store.store(updated)  # type: ignore[union-attr]
        elif hasattr(store, "update_importance"):
            store.update_importance(frag.id, new_imp)  # type: ignore[union-attr]


__all__ = ["MemoryPruner", "PrunerStats"]
