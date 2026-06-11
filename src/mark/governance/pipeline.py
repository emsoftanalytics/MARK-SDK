# SPDX-License-Identifier: MIT
# Copyright (c) 2025 MARK Contributors
"""Consolidation gate pipeline orchestrating the local governance checks."""
from __future__ import annotations

from dataclasses import dataclass

from mark.governance.gates import DuplicateHashGate, FailureFilter, GateResult, ValidationGate
from mark.governance.sanitizer import ContentSanitizer


@dataclass
class ConsolidationGateResult:
    """Aggregate outcome of the consolidation gate pipeline."""
    passed:  bool
    content: str   # sanitized content (even when blocked)
    reason:  str = ""


class ConsolidationGate:
    """
    Orchestrates local governance gates before memory promotion.

    Pipeline order:
      1. ContentSanitizer    — redact PII, normalise length
      2. ValidationGate      — reject empty / low-confidence content
      3. FailureFilter       — reject stack traces and error artifacts
      4. DuplicateHashGate   — reject exact-duplicate content (optional)

    Cloud governance replaces this with LLM evaluation, policy packs, and
    immutable audit trails.
    """

    def __init__(
        self,
        *,
        sanitizer:  ContentSanitizer  | None = None,
        validation: ValidationGate    | None = None,
        failure:    FailureFilter     | None = None,
        dedup:      DuplicateHashGate | None = None,
    ) -> None:
        self._sanitizer  = sanitizer  or ContentSanitizer()
        self._validation = validation or ValidationGate()
        self._failure    = failure    or FailureFilter()
        self._dedup      = dedup      # optional — None means skip

    def check(
        self,
        content: str,
        *,
        confidence: float = 1.0,
    ) -> ConsolidationGateResult:
        """Check the content and return the gate outcome."""
        cleaned = self._sanitizer.sanitize(content)

        v = self._validation.check(cleaned, confidence=confidence)
        if not v.passed:
            return ConsolidationGateResult(False, cleaned, v.reason)

        f = self._failure.check(cleaned)
        if not f.passed:
            return ConsolidationGateResult(False, cleaned, f.reason)

        if self._dedup is not None:
            d = self._dedup.check(cleaned)
            if not d.passed:
                return ConsolidationGateResult(False, cleaned, d.reason)

        return ConsolidationGateResult(True, cleaned)
