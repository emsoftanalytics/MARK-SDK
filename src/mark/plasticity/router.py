# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 MARK Contributors — v1.3 plasticity
#
# Apache-2.0 (open SDK):
#   RoutingFeedback  — reward signal data structure
#   RouterDecision   — routing decision data structure
#   StaticRouter     — stateless router; wraps QueryClassifier
#
# HOOK_LEARNING_ROUTER can register an adaptive routing plugin.
# HOOK_RL_POLICY can register a learned memory policy plugin.
"""Static local routing defaults; hooks may supply learned routers."""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class RoutingFeedback:
    """
    Reward signal for a routing decision.

    Pass to the HOOK_LEARNING_ROUTER plugin when registered.
    """
    decision_id: str
    reward:      float   # 0.0–1.0


@dataclass
class RouterDecision:
    """
    A routing decision produced by StaticRouter or a registered router plugin.
    """
    id:         str   = field(default_factory=lambda: uuid.uuid4().hex[:12])
    policy:     str   = "BALANCED"   # FAST | BALANCED | DEEP
    confidence: float = 0.5


class StaticRouter:
    """
    Stateless router for local development — wraps QueryClassifier.

    Converts a query into a RouterDecision without any learning state.
    Suitable for small projects and local development.

    Register HOOK_LEARNING_ROUTER to activate an adaptive routing plugin.
    """

    def __init__(self) -> None:
        from mark.intelligence.classifier import QueryClassifier
        self._classifier = QueryClassifier()

    def route(self, query: str, features: Optional[Dict[str, Any]] = None) -> RouterDecision:
        """Return a routing decision based on query complexity."""
        analysis = self._classifier.classify(query)
        return RouterDecision(
            policy     = analysis.policy.value.upper(),
            confidence = analysis.complexity,
        )

    def record_feedback(self, feedback: RoutingFeedback) -> None:
        """No-op locally. Register HOOK_LEARNING_ROUTER to handle feedback."""
        pass
