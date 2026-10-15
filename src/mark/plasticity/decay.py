# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 MARK Contributors — v1.3 plasticity
#
# ExponentialDecay — importance decay for MemoryFragments.
#
# Biological analogy: LTD (long-term depression) — memories that are not
# accessed weaken over time until they fall below the pruning floor.
#
# Hook: HOOK_PLASTICITY activates a registered plasticity plugin.
"""Time-based importance decay with tier half-lives."""
from __future__ import annotations

import math
from datetime import datetime, timezone
from enum import Enum
from typing import ClassVar, Dict, Optional

DECAY_FLOOR     = 0.05   # fragments at or below this importance are pruned
REINFORCE_DELTA = 0.05   # per-access importance boost (see HebbianReinforcement)
DECAY_METADATA_KEY = "mark_decay"  # {"base_importance": float, "last_decay_at": iso8601}


def _aware(value: datetime) -> datetime:
    """Treat naive datetimes as UTC so aware/naive arithmetic never crashes."""
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value


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

    Register HOOK_PLASTICITY to activate a custom plasticity plugin.
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

    def reference_time(self, fragment: object) -> Optional[datetime]:
        """When decay was last applied (``mark_decay.last_decay_at``), else created_at."""
        metadata = getattr(fragment, "metadata", None) or {}
        record = metadata.get(DECAY_METADATA_KEY) if isinstance(metadata, dict) else None
        if isinstance(record, dict) and record.get("last_decay_at"):
            try:
                return _aware(datetime.fromisoformat(str(record["last_decay_at"])))
            except ValueError:
                pass
        created_at = getattr(fragment, "created_at", None)
        return _aware(created_at) if isinstance(created_at, datetime) else None

    def apply(self, fragment: object, *, now: Optional[datetime] = None) -> float:
        """
        Compute decayed importance for one fragment.

        Decay is applied to the current importance for the time elapsed since
        decay was last applied (recorded in ``metadata["mark_decay"]``), or
        since creation for never-decayed fragments. Because exponential decay
        composes multiplicatively, repeated runs equal one run over the total
        interval, and re-running at the same instant changes nothing.

        PROMOTED state = forget gate: no decay applied.
        Returns float in [floor, 1.0]. Does not mutate the fragment.
        """
        from mark.types import MemoryState
        if getattr(fragment, "state", None) == MemoryState.PROMOTED:
            return float(getattr(fragment, "importance", 1.0))
        reference = self.reference_time(fragment)
        importance = float(getattr(fragment, "importance", 0.5))
        if reference is None:
            return importance
        current   = _aware(now) if now is not None else datetime.now(timezone.utc)
        age_hours = (current - reference).total_seconds() / 3600.0
        new_imp   = importance * self.decay_factor(age_hours)
        return max(self._floor, min(1.0, new_imp))

    def decay_metadata(self, fragment: object, *, now: datetime) -> Dict[str, object]:
        """Return fragment metadata updated with this decay step's bookkeeping."""
        metadata = dict(getattr(fragment, "metadata", None) or {})
        previous = metadata.get(DECAY_METADATA_KEY)
        base = (
            previous.get("base_importance")
            if isinstance(previous, dict) and "base_importance" in previous
            else float(getattr(fragment, "importance", 0.5))
        )
        metadata[DECAY_METADATA_KEY] = {
            "base_importance": base,
            "last_decay_at": _aware(now).isoformat(),
        }
        return metadata

    def should_prune(self, fragment: object, *, now: Optional[datetime] = None) -> bool:
        """Return True if decayed importance has reached the floor."""
        return self.apply(fragment, now=now) <= self._floor
