# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 MARK Contributors — v1.4 RL data pipeline
#
# Collectors store MINIMAL signals by default — no raw fragment content.
# Raw payloads require explicit opt-in (raw=True) and should only be
# enabled after reviewing your data governance policy.
"""Local collectors for retrieval and feedback signals."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class GovernanceSignal:
    """
    Supervised training label from the governance pipeline.

    Open-contract note:
    This Apache-2.0 SDK type records the public shape of a governance label.
    Governance models, audit ledgers, training datasets, and promotion systems
    are not implemented by this package.

    content_hash  — SHA-256 of fragment content (never raw text by default)
    label         — "approve" | "reject" | "quarantine"
    confidence    — gate confidence score
    context_hint  — non-sensitive context tag (e.g. "scope:agent, tier:episodic")
    """
    content_hash:  str
    label:         str
    confidence:    float
    context_hint:  str = ""


class RoutingDecisionLogger:
    """Logs routing decisions for audit and replay (no raw content stored).

    Open-contract note:
    This is a local transparency helper, not an adaptive router or model
    training implementation.
    """

    def __init__(self, runtime: Any = None) -> None:
        self._runtime = runtime
        self._log: List[Dict] = []

    def log(self, experience: Any, query_hash: str, policy: str) -> None:
        """Log a routing decision. Pass hash(query), not the raw query."""
        self._log.append({
            "agent_id":   getattr(experience, "agent_id", ""),
            "query_hash": query_hash,
            "policy":     policy,
            "reward":     getattr(experience, "reward", 0.0),
            "timestamp":  getattr(experience, "timestamp", 0.0),
        })

    def recent(self, limit: int = 50) -> List[Dict]:
        """Return the most recent entries."""
        return self._log[-limit:]


class GovernanceSignalCollector:
    """
    Accumulates governance gate decisions as supervised training labels.

    Open-contract note:
    This collector only stores transparent local signal records. Ingestion,
    consent checks, dataset lineage, and training pipelines are not implemented
    in the Apache-2.0 SDK.

    Raw fragment content is NEVER stored by default — only SHA-256 hashes.
    This prevents sensitive agent data from leaking through the training
    pipeline.

    To store raw content in a controlled deployment:
        collector = GovernanceSignalCollector(raw=True)
    """

    def __init__(self, raw: bool = False) -> None:
        self._raw     = raw
        self._signals: List[GovernanceSignal] = []

    def collect(
        self,
        fragment_content: str,
        label:            str,
        confidence:       float,
        context_hint:     str = "",
    ) -> None:
        """Record one signal."""
        content_hash = (
            fragment_content if self._raw
            else hashlib.sha256(fragment_content.encode()).hexdigest()
        )
        self._signals.append(GovernanceSignal(
            content_hash = content_hash,
            label        = label,
            confidence   = confidence,
            context_hint = context_hint,
        ))

    def all(self) -> List[GovernanceSignal]:
        """Return every stored item."""
        return list(self._signals)

    def count(self) -> int:
        """Return the number of stored entries."""
        return len(self._signals)
