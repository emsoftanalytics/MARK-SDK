# SPDX-License-Identifier: MIT
# Copyright (c) 2025 MARK Contributors — v1.4 RL data pipeline
#
# MIT (open SDK):
#   RewardSignal  — structured record of one interaction's outcome
#   RewardComputer — trivial local placeholder (quality score only)
#
# MARK Core (MSAL) — registered via HOOK_FEEDBACK:
#   Cloud reward scoring plugin. Calibrated weights are not part of the
#   public SDK.
"""Transparent local reward scoring schema."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class RewardSignal:
    """
    Structured record of one agent interaction's outcome.

    Open-contract note:
    This MIT SDK type is only the public outcome envelope. Hosted reward
    models, calibrated scoring formulas, and training pipelines remain outside
    this package.

    Collected locally and fed to the HOOK_FEEDBACK plugin (when registered)
    which routes signals to MARK Cloud.
    """
    decision_id:           str
    agent_id:              str
    policy_used:           str           # "FAST" | "BALANCED" | "DEEP"
    result_count:          int
    latency_ms:            float
    evaluator_score:       float         # 0.0–1.0, quality of the retrieval
    hallucination_flagged: bool  = False
    governance_rejected:   bool  = False
    reward:                Optional[float] = None  # filled by RewardComputer


class RewardComputer:
    """
    Local placeholder reward computer for small projects and testing.

    Open-contract note:
    This class intentionally computes only a simple local fallback reward. It
    is not MARK Core's proprietary reward scorer or adaptive learning system.

    Uses only the evaluator_score — no weighted penalty formula.
    Sufficient for local development and SDK smoke tests.

    Cloud replacement: register HOOK_FEEDBACK to activate the cloud
    reward scoring plugin.
    """

    def compute(self, signal: RewardSignal) -> float:
        """Simple local reward: evaluator score clamped to [0, 1]."""
        return max(0.0, min(1.0, float(signal.evaluator_score)))
