# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 MARK Contributors
"""Regex-based redaction of common secrets and PII for governance gates.

The sanitizer is the strict, explicit counterpart of the default write-time
redaction: it removes secrets *and* contact PII (emails, phones, IBANs,
SSNs). It never truncates unless ``max_length`` is set explicitly, and then
reports it (``SanitizeReport.truncated``) — content size limits belong to
the write path, not to a redaction step that reports ``passed=True``.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from mark.security.redaction import RedactionPolicy, redact
from mark.security.lazy_regex import lazy_compile

# Governance is opt-in and strict: secrets plus all contact PII.
STRICT_POLICY = RedactionPolicy(secrets=True, pii=True)

_INLINE_SPACE_RUN = lazy_compile(r"(?<=\S)[ \t]{2,}")


@dataclass(frozen=True)
class SanitizeReport:
    """Sanitized text plus what the sanitizer changed."""
    text:            str
    truncated:       bool = False
    original_length: int  = 0
    redacted:        bool = False


class ContentSanitizer:
    """
    Redact secrets and common PII patterns and normalise content length.

    Uses the shared pattern set from :mod:`mark.security.redaction`, so
    governance and sync redaction always agree. Newlines and leading
    indentation are preserved; only runs of spaces inside a line collapse.
    ``max_length`` defaults to None (no limit); when set, truncation is
    reported via :meth:`sanitize_report` and recorded in the memory metadata.
    ``policy`` overrides the strict default redaction policy. Used locally
    before memory promotion. No network calls.
    """

    def __init__(self, *, max_length: int | None = None, policy: RedactionPolicy | None = None) -> None:
        self._max_length = max_length
        self._policy = policy or STRICT_POLICY

    def sanitize_report(self, text: str) -> SanitizeReport:
        """Return the sanitized text together with redaction/truncation details."""
        original_length = len(text)
        redacted = redact(text, self._policy).text
        was_redacted = redacted != text
        lines = [_INLINE_SPACE_RUN.sub(" ", line).rstrip() for line in redacted.splitlines()]
        cleaned = "\n".join(lines).strip()
        truncated = self._max_length is not None and len(cleaned) > self._max_length
        if truncated:
            cleaned = cleaned[: self._max_length]
        return SanitizeReport(
            text=cleaned,
            truncated=truncated,
            original_length=original_length,
            redacted=was_redacted,
        )

    def sanitize(self, text: str) -> str:
        """Return the text with secrets/PII redacted."""
        return self.sanitize_report(text).text

    def has_pii(self, text: str) -> bool:
        """Return True when the text matches a known secret or PII pattern."""
        return redact(text, self._policy).changed
