# SPDX-License-Identifier: MIT
# Copyright (c) 2025 MARK Contributors
"""Regex-based redaction of common secrets and PII."""
from __future__ import annotations

import re


_PII_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),                                                  "[SSN]"),
    (re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b"),                                            "[CC]"),
    (re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"),                  "[EMAIL]"),
    (re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),               "[PHONE]"),
    (re.compile(r"\b(?:tok|key|secret|password|passwd|token|bearer)[_\-]?\w{8,}\b", re.I),  "[SECRET]"),
]


class ContentSanitizer:
    """
    Redact common PII patterns and normalise content length.

    Used locally before memory promotion. No network calls.
    Cloud governance adds ML-based PII detection and policy packs.
    """

    def __init__(self, *, max_length: int = 10_000) -> None:
        self._max_length = max_length

    def sanitize(self, text: str) -> str:
        """Return the text with secrets/PII redacted."""
        for pattern, replacement in _PII_PATTERNS:
            text = pattern.sub(replacement, text)
        text = " ".join(text.split())   # collapse whitespace
        if len(text) > self._max_length:
            text = text[: self._max_length]
        return text

    def has_pii(self, text: str) -> bool:
        """Return True when the text matches a known PII pattern."""
        return any(p.search(text) for p, _ in _PII_PATTERNS)
