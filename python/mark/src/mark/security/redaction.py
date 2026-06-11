"""Secret and PII redaction helpers."""
from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

SECRET_PATTERNS = [
    re.compile(r"(?i)(api[\s_-]?key|token|secret|password)\s*[:=]\s*['\"]?([^'\"\s]+)"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{8,}\b"),
]


def redact_secrets(text: str) -> str:
    """Return the text with API keys, tokens, and passwords replaced by [REDACTED]."""
    redacted = text
    for pattern in SECRET_PATTERNS:
        if pattern.groups >= 2:
            redacted = pattern.sub(lambda match: f"{match.group(1)}=[REDACTED]", redacted)
        else:
            redacted = pattern.sub("[REDACTED]", redacted)
    return redacted


def redact_sync_value(value: Any) -> Any:
    """Recursively redact strings inside JSON-like sync payload values."""
    if isinstance(value, str):
        return redact_secrets(value)
    if isinstance(value, Mapping):
        return {
            str(redact_secrets(str(key))): redact_sync_value(item)
            for key, item in value.items()
        }
    if isinstance(value, tuple):
        return tuple(redact_sync_value(item) for item in value)
    if isinstance(value, list):
        return [redact_sync_value(item) for item in value]
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return [redact_sync_value(item) for item in value]
    return value
