# SPDX-License-Identifier: MIT
# Copyright (c) 2025 MARK Contributors — v1.3 plasticity
#
# ExponentialDecay — importance decay for MemoryFragments.
#
# Biological analogy: LTD (long-term depression) — memories that are not
# accessed weaken over time until they fall below the pruning floor.
#
# Cloud hook: HOOK_PLASTICITY activates the cloud plasticity plugin.
"""Time-based importance decay with tier half-lives."""
from __future__ import annotations

import math
from datetime import datetime, timezone
from enum import Enum
from typing import ClassVar, Dict, Optional

DECAY_FLOOR     = 0.05   # fragments at or below this importance are pruned
REINFORCE_DELTA = 0.05   # per-access importance boost (see HebbianReinforcement)


class TemporalScope(str, Enum):
    """
    Maps conceptual memory lifetimes to exponential decay half-lives.

    SHORT_TERM   48h   — conversational scratchpad; decays fast
    MEDIUM_TERM  168h  — sprint/project facts; decays over a week
    LONG_TERM    720h  — stable knowledge; decays over a month
    PERMANENT    None  — core truths, never decays (forget gate)

    Media agent usage:
        Scene context  → SHORT_TERM
        Episode facts  → MEDIUM_TERM
        Character bible, world rules → PERMANENT (+ PROMOTED state)
    """
    SHORT_TERM  = "short_term"
    MEDIUM_TERM = "medium_term"
    LONG_TERM   = "long_term"
    PERMANENT   = "permanent"

    HALF_LIVES_HOURS: ClassVar[Dict["TemporalScope", Optional[float]]]


TemporalScope.HALF_LIVES_HOURS = {
    TemporalScope.SHORT_TERM:  48.0,
    TemporalScope.MEDIUM_TERM: 168.0,
    TemporalScope.LONG_TERM:   720.0,
    TemporalScope.PERMANENT:   None,
}


class ExponentialDecay:
    """
    Computes exponential importance decay for MemoryFragments.

    Formula: new_importance = max(DECAY_FLOOR, current * exp(−λ × age_hours))
    where λ = ln(2) / half_life_hours

    Forget gate: PROMOTED fragments are exempt from all decay.
    Does NOT mutate fragments — returns the new importance value.
    Caller (MemoryPruner or scheduled job) applies the result.

    Cloud replacement: HOOK_PLASTICITY activates the cloud plasticity plugin.
    """

    def __init__(self, temporal_scope: TemporalScope = TemporalScope.MEDIUM_TERM,
                 floor: float = DECAY_FLOOR) -> None:
        self._scope = temporal_scope
        self._floor = floor
        half_life   = TemporalScope.HALF_LIVES_HOURS[temporal_scope]
        self._lambda: Optional[float] = (
            math.log(2) / half_life if half_life is not None else None
        )

    def decay_factor(self, age_hours: float) -> float:
        """Multiplicative decay factor for a given age. Returns 1.0 for PERMANENT scope."""
        if self._lambda is None:
            return 1.0
        return math.exp(-self._lambda * max(0.0, age_hours))

    def apply(self, fragment: object) -> float:
        """
        Compute decayed importance for one fragment.
        PROMOTED state = forget gate: no decay applied.
        Returns float in [floor, 1.0]. Does not mutate the fragment.
        """
        from mark.types import MemoryState
        if getattr(fragment, "state", None) == MemoryState.PROMOTED:
            return float(getattr(fragment, "importance", 1.0))
        created_at = getattr(fragment, "created_at", None)
        if created_at is None:
            return float(getattr(fragment, "importance", 0.5))
        now       = datetime.now(timezone.utc)
        age_hours = (now - created_at).total_seconds() / 3600.0
        importance = float(getattr(fragment, "importance", 0.5))
        new_imp   = importance * self.decay_factor(age_hours)
        return max(self._floor, min(1.0, new_imp))

    def should_prune(self, fragment: object) -> bool:
        """Return True if decayed importance has reached the floor."""
        return self.apply(fragment) <= self._floor
