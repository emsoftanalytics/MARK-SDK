# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 MARK Contributors
"""Regex-based redaction of common secrets and PII."""
from __future__ import annotations

import re
from dataclasses import dataclass

from mark.security.redaction import contains_sensitive, redact_text

_INLINE_SPACE_RUN = re.compile(r"(?<=\S)[ \t]{2,}")


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
    Truncation is reported via :meth:`sanitize_report` rather than happening
    silently. Used locally before memory promotion. No network calls.
    """

    def __init__(self, *, max_length: int = 10_000) -> None:
        self._max_length = max_length

    def sanitize_report(self, text: str) -> SanitizeReport:
        """Return the sanitized text together with redaction/truncation details."""
        original_length = len(text)
        redacted = redact_text(text, emails=True, pii=True)
        was_redacted = redacted != text
        lines = [_INLINE_SPACE_RUN.sub(" ", line).rstrip() for line in redacted.splitlines()]
        cleaned = "\n".join(lines).strip()
        truncated = len(cleaned) > self._max_length
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
        return contains_sensitive(text, emails=True, pii=True)
