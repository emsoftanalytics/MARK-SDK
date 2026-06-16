# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 MARK Contributors
"""Validation gates that screen content before it becomes memory."""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass


@dataclass
class GateResult:
    """Outcome of a single governance gate check."""
    passed:  bool
    reason:  str  = ""


# Patterns that strongly indicate content is an error/failure artifact, not memory.
_FAILURE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"Traceback \(most recent call last\)"),
    re.compile(r'^\s*\{\s*"error"\s*:', re.M),
    re.compile(r"^\s*Error:\s+\w", re.M),
    re.compile(r"<error>"),
    re.compile(r"\bHTTP\s+[45]\d{2}\b"),
    re.compile(r"\bRESOURCE_EXHAUSTED\b", re.I),
    re.compile(r"\bPAYMENT_REQUIRED\b", re.I),
    re.compile(r"\binsufficient balance\b", re.I),
    re.compile(r"\bno image\b", re.I),
    re.compile(r"\bmissing image\b", re.I),
    re.compile(r"\bimage artifact failed\b", re.I),
    re.compile(r"\bgeneration failed\b", re.I),
    re.compile(r"\breview error\b", re.I),
    re.compile(r"\bvideo assembly failed\b", re.I),
]


class ValidationGate:
    """
    Reject fragments that are empty, too short, or below a confidence floor.

    Registered governance extensions can add hallucination scoring and policy packs.
    """

    def __init__(
        self,
        *,
        min_length:     int   = 1,
        min_confidence: float = 0.0,
    ) -> None:
        self._min_length     = max(1, min_length)
        self._min_confidence = min_confidence

    def check(self, content: str, *, confidence: float = 1.0) -> GateResult:
        """Check the content and return the gate outcome."""
        stripped = content.strip()
        if len(stripped) < self._min_length:
            return GateResult(False, "content too short or empty")
        if confidence < self._min_confidence:
            return GateResult(
                False,
                f"confidence {confidence:.3f} below threshold {self._min_confidence:.3f}",
            )
        return GateResult(True)


class DuplicateHashGate:
    """
    Reject fragments whose exact content hash has already been seen.

    Hashes are stored in-process only; use DeduplicationConsolidator for
    persistent cross-session deduplication.
    """

    def __init__(self) -> None:
        self._seen: set[str] = set()

    def check(self, content: str) -> GateResult:
        """Check the content and return the gate outcome."""
        h = hashlib.sha256(content.encode()).hexdigest()
        if h in self._seen:
            return GateResult(False, "duplicate content hash")
        self._seen.add(h)
        return GateResult(True)

    def reset(self) -> None:
        """Reset internal state."""
        self._seen.clear()


class FailureFilter:
    """
    Reject content that looks like an error message or exception trace.

    Designed to prevent accidental storage of stack traces, HTTP error bodies,
    or raw exception strings as memory fragments.
    """

    def check(self, content: str) -> GateResult:
        """Check the content and return the gate outcome."""
        for p in _FAILURE_PATTERNS:
            if p.search(content):
                return GateResult(False, f"content matches failure pattern")
        return GateResult(True)
