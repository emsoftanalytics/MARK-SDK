# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 MARK Contributors — v1.4 RL data pipeline
"""Local experience buffer for plasticity/feedback signals."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import ClassVar, List, Optional


@dataclass
class ExperienceTuple:
    """
    One (state, action, reward, next_state, done) transition for RL training.

    Open-contract note:
    This Apache-2.0 SDK type is only a transparent telemetry/learning-signal envelope.
    It does not include training, policy optimization, model
    inference, or proprietary reward calibration.

    state       — encoded feature vector (List[float], length = StateEncoder.state_dim)
    action      — retrieval policy: "FAST" | "BALANCED" | "DEEP"
    reward      — scalar reward in [0, 1] from RewardComputer
    next_state  — state vector after action
    done        — True for the last interaction in an episode
    agent_id    — which agent produced this experience
    timestamp   — Unix timestamp
    """
    state:      List[float]
    action:     str
    reward:     float
    next_state: List[float]
    done:       bool
    agent_id:   str
    timestamp:  float = field(default_factory=time.time)

    def is_valid(self) -> bool:
        """Return True when the entry passes validation."""
        return (
            len(self.state) > 0
            and len(self.next_state) == len(self.state)
            and self.action in ("FAST", "BALANCED", "DEEP")
            and 0.0 <= self.reward <= 1.0
            and isinstance(self.agent_id, str)
            and len(self.agent_id) > 0
        )


class ReplayBuffer:
    """
    Fixed-capacity circular buffer of ExperienceTuples.

    Open-contract note:
    This local buffer is provided so developers can inspect and test the shape
    of learning signals. Dataset ingestion, retention, training, and model
    promotion are not implemented by the Apache-2.0 SDK.

    When capacity is reached, oldest experiences are overwritten (FIFO).
    Supports random sampling for mini-batch training.

    DatasetExporter.flush_to_parquet() can call ReplayBuffer.load_latest()
    without an explicit handle via the class-level _latest reference.
    """

    _latest: ClassVar[Optional["ReplayBuffer"]] = None

    def __init__(self, capacity: int = 10_000) -> None:
        self._capacity = capacity
        self._buffer:  List[ExperienceTuple] = []
        self._pos:     int = 0
        ReplayBuffer._latest = self

    def push(self, experience: ExperienceTuple) -> None:
        """Append an item."""
        if len(self._buffer) < self._capacity:
            self._buffer.append(experience)
        else:
            self._buffer[self._pos] = experience
        self._pos = (self._pos + 1) % self._capacity

    def sample(self, n: int) -> List[ExperienceTuple]:
        """Return up to n random items."""
        import random
        return random.choices(self._buffer, k=min(n, len(self._buffer)))

    def size(self) -> int:
        """Return the number of stored items."""
        return len(self._buffer)

    def all(self) -> List[ExperienceTuple]:
        """Return every stored item."""
        return list(self._buffer)

    @classmethod
    def load_latest(cls) -> "ReplayBuffer":
        """Load the most recent persisted buffer."""
        if cls._latest is None:
            raise RuntimeError("No ReplayBuffer has been created in this session")
        return cls._latest
