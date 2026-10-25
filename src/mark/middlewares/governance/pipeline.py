# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 Somtochukwu Owen Ikeanyi
"""Consolidation gate pipeline orchestrating the local governance checks."""
from __future__ import annotations

from dataclasses import dataclass

from mark.middlewares.governance.gates import DuplicateHashGate, FailureFilter, GateResult, ValidationGate
from mark.middlewares.governance.sanitizer import ContentSanitizer


@dataclass
class ConsolidationGateResult:
    """Aggregate outcome of the consolidation gate pipeline."""
    passed:  bool
    content: str   # sanitized content (even when blocked)
    reason:  str = ""
    truncated:       bool = False
    original_length: int  = 0


class ConsolidationGate:
    """
    Orchestrates local governance gates before memory promotion.

    Pipeline order:
      1. ContentSanitizer    — redact secrets + PII, normalise whitespace
      2. ValidationGate      — reject empty / low-confidence content
      3. FailureFilter       — reject stack traces and error artifacts (opt-in)
      4. DuplicateHashGate   — reject exact-duplicate content (opt-in)

    Since 0.2.0a10 the FailureFilter is off by default: agents must be able
    to remember errors, stack traces and HTTP failures together with their
    fixes. Pass ``failure=FailureFilter()`` to restore the old quarantine.

    Registered governance extensions can add LLM evaluation, policy packs, and
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
        self._failure    = failure    # optional — None means errors are kept
        self._dedup      = dedup      # optional — None means skip

    def check(
        self,
        content: str,
        *,
        confidence: float = 1.0,
    ) -> ConsolidationGateResult:
        """Check the content and return the gate outcome."""
        if hasattr(self._sanitizer, "sanitize_report"):
            report = self._sanitizer.sanitize_report(content)
            cleaned, truncated, original_length = report.text, report.truncated, report.original_length
        else:  # custom sanitizer implementing only sanitize()
            cleaned, truncated, original_length = self._sanitizer.sanitize(content), False, len(content)

        def result(passed: bool, reason: str = "") -> ConsolidationGateResult:
            return ConsolidationGateResult(
                passed, cleaned, reason, truncated=truncated, original_length=original_length,
            )

        v = self._validation.check(cleaned, confidence=confidence)
        if not v.passed:
            return result(False, v.reason)

        if self._failure is not None:
            f = self._failure.check(cleaned)
            if not f.passed:
                return result(False, f.reason)

        if self._dedup is not None:
            d = self._dedup.check(cleaned)
            if not d.passed:
                return result(False, d.reason)

        return result(True)
