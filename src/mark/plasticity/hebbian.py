# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 MARK Contributors — v1.3 plasticity
#
# HebbianReinforcement + EdgeCoActivation
#
# Biological analogy: LTP (long-term potentiation) and Hebbian learning —
# "neurons that fire together wire together."
#
# Hook: HOOK_PLASTICITY activates a registered plasticity plugin.
"""Hebbian reinforcement and edge co-activation arithmetic."""
from __future__ import annotations

import math

from .decay import REINFORCE_DELTA

COACTIVATION_DELTA = 0.03   # edge weight increment on co-retrieval
EDGE_FLOOR         = 0.10   # edge weight below this → dissolved


class HebbianReinforcement:
    """
    Implements Hebbian strengthening: on each fragment access, importance
    is boosted by a dampened delta.

    Boost formula: delta * (1 − exp(−access_count / 10))
    — dampening prevents runaway growth; heavily accessed fragments
    asymptotically approach importance=1.0.

    Does NOT mutate fragments. Returns the new importance value.
    Caller must persist: store.update_importance(frag.id, new_imp).

    Media agent use: characters/props accessed every scene naturally
    reinforce to high importance through this mechanism alone.
    """

    def __init__(self, delta: float = REINFORCE_DELTA) -> None:
        self._delta = delta

    def on_access(self, fragment: object) -> float:
        """Return the new importance for a fragment that was just accessed."""
        access_count = int(getattr(fragment, "metadata", {}).get("access_count", 0))
        importance   = float(getattr(fragment, "importance", 0.5))
        boost = self._delta * (1 - math.exp(-access_count / 10.0))
        return min(1.0, importance + boost)


class EdgeCoActivation:
    """
    Strengthens MemoryEdge weight when two nodes are co-retrieved in the same query.

    On co-retrieval: edge.weight += COACTIVATION_DELTA (capped at 1.0)
    Below EDGE_FLOOR: edge is a candidate for dissolution by MemoryPruner.

    Does NOT mutate edges. Returns the new weight.

    Media agent use: character→prop, character→location edges strengthen
    when co-retrieved in the same scene — the graph learns the story's
    visual language automatically.
    """

    def __init__(self, delta: float = COACTIVATION_DELTA,
                 floor: float = EDGE_FLOOR) -> None:
        self._delta = delta
        self._floor = floor

    def on_co_retrieval(self, current_weight: float) -> float:
        """Return the strengthened weight after co-retrieval."""
        return min(1.0, current_weight + self._delta)

    def should_dissolve(self, weight: float) -> bool:
        """Return True when the edge weight fell below the floor."""
        return weight < self._floor
