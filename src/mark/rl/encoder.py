# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 MARK Contributors
"""Feature encoding for locally collected signals."""
from __future__ import annotations

from typing import Dict, List


class StateEncoder:
    """
    Encodes memory system state into a fixed-length float vector.

    Open-contract note:
    This Apache-2.0 SDK encoder is a transparent local feature-shape helper.
    Registered plugins may use richer private feature sets.

    Features encoded (each normalised to [0, 1]):
      query_complexity   — 0=simple, 0.5=medium, 0.9=complex
      result_count       — normalised by max_results (default 20)
      avg_importance     — already in [0, 1]
      edge_density       — already in [0, 1]
      routing_signal     — current routing policy signal

    The vector is L2-normalised to produce consistent input magnitudes.

    Register HOOK_FEEDBACK to activate a feedback plugin.
    """

    DEFAULT_FEATURES = [
        "query_complexity",
        "result_count",
        "avg_importance",
        "edge_density",
        "routing_signal",
    ]
    MAX_RESULTS = 20.0

    def __init__(self, state_dim: int = 64) -> None:
        self._state_dim = state_dim

    def encode(self, features: Dict[str, float]) -> List[float]:
        """Encode features into a vector."""
        raw = [
            float(features.get("query_complexity", 0.0)),
            float(features.get("result_count",      0.0)) / self.MAX_RESULTS,
            float(features.get("avg_importance",    0.0)),
            float(features.get("edge_density",      0.0)),
            float(features.get("routing_signal",    0.0)),
        ]
        padded = (raw + [0.0] * self._state_dim)[:self._state_dim]
        norm = sum(x ** 2 for x in padded) ** 0.5
        if norm > 1e-8:
            padded = [x / norm for x in padded]
        return padded
