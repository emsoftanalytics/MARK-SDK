"""Secret and PII redaction helpers.

This module owns the single shared pattern set used by both sync redaction
(:func:`redact_secrets`, :func:`redact_sync_value`) and the governance
:class:`~mark.middlewares.governance.sanitizer.ContentSanitizer`.
"""
from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any, Callable

REDACTED = "[REDACTED]"

# Key names whose values are always secret, in free text (key=value) and in
# mapping keys. ``auth(?!ors?\b)`` keeps "author"/"authors" readable while
# still matching "auth", "authorization", "auth_token", ...
SENSITIVE_KEY_RE = re.compile(
    r"(?i)(pass(?:word)?|passwd|secret|token|api[\s_-]?key|auth(?!ors?\b)|cookie|credential|private[_-]?key)"
)
# Key names that contain a sensitive word but are routine counters/settings.
_BENIGN_KEY_RE = re.compile(
    r"(?i)^(?:max_?tokens?|num_?tokens?|tokens?_?(?:count|used|limit|budget)|"
    r"(?:prompt|completion|total|input|output)_tokens?|token_?type|tokenizer|auth_?type|auth_?method)$"
)


def is_sensitive_key(key: object) -> bool:
    """Return True when a mapping key / assignment name names a secret value."""
    text = str(key)
    return bool(SENSITIVE_KEY_RE.search(text)) and not _BENIGN_KEY_RE.match(text)


_Replacement = str | Callable[["re.Match[str]"], str]

# Order matters: whole-structure secrets (PEM blocks, URLs with credentials,
# bearer tokens) run before the generic key=value rule so a value is never
# half-redacted.
SECRET_PATTERNS: list[tuple[str, re.Pattern[str], _Replacement]] = [
    (
        "pem_private_key",
        re.compile(
            r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----[\s\S]*?(?:-----END [A-Z0-9 ]*PRIVATE KEY-----|\Z)"
        ),
        "[REDACTED_PRIVATE_KEY]",
    ),
    (
        "url_credentials",
        re.compile(r"(?i)\b([a-z][a-z0-9+.\-]*://)[^\s/:@]+:[^\s/@]+@"),
        lambda m: f"{m.group(1)}{REDACTED}@",
    ),
    (
        "bearer",
        re.compile(r"(?i)\b(Bearer|Basic|Token)\s+[A-Za-z0-9._~+/=\-]{8,}"),
        lambda m: f"{m.group(1)} {REDACTED}",
    ),
    ("jwt", re.compile(r"\beyJ[A-Za-z0-9_-]{5,}\.[A-Za-z0-9_-]{5,}\.[A-Za-z0-9_-]{5,}"), REDACTED),
    ("aws_access_key", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"), REDACTED),
    ("github_token", re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})"), REDACTED),
    ("slack_token", re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}"), REDACTED),
    ("anthropic_key", re.compile(r"\bsk-ant-[A-Za-z0-9_-]{8,}"), REDACTED),
    ("openai_key", re.compile(r"\bsk-[A-Za-z0-9_-]{8,}"), REDACTED),
    ("google_api_key", re.compile(r"\bAIza[0-9A-Za-z_-]{30,}"), REDACTED),
    ("huggingface_token", re.compile(r"\bhf_[A-Za-z0-9]{20,}"), REDACTED),
    (
        "prefixed_secret",
        re.compile(r"(?i)\b(?:tok|key|secret|password|passwd|token|bearer)[_\-]\w{8,}\b"),
        REDACTED,
    ),
]


def _redact_assignment(match: "re.Match[str]") -> str:
    key, sep, value = match.group(1), match.group(2), match.group(3)
    if not is_sensitive_key(key):
        return match.group(0)
    if value.startswith("[REDACTED") or value in {"Bearer", "Basic", "Token"}:
        return match.group(0)
    if len(value) >= 2 and value[0] in "\"'" and value[-1] == value[0]:
        return f"{key}{sep}{value[0]}{REDACTED}{value[0]}"
    return f"{key}{sep}{REDACTED}"


# key=value, key: value, "key": "value", key = 'value'
# Only keys that contain a sensitive word can match, so ordinary "label: text"
# (or "https://...") never consumes a later secret assignment.
ASSIGNMENT_PATTERN = re.compile(
    r"(?i)(?<![\w.\-])("                                       # key
    r"[\w.\-]*?(?:pass(?:word)?|passwd|secret|token|api[ \t_-]?key|auth(?!ors?\b)|cookie|"
    r"credential|private[_-]?key)[\w.\-]*)"
    r"([\"']?[ \t]*[:=](?!//)[ \t]*)"            # separator (optional closing quote of key)
    r"(\"[^\"\n]*\"|'[^'\n]*'|[^\s,;&\"'<>]+)"  # value
)

EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")

PII_PATTERNS: list[tuple[str, re.Pattern[str], str]] = [
    ("ssn", re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[SSN]"),
    ("credit_card", re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b"), "[CC]"),
    ("phone", re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"), "[PHONE]"),
]


def redact_text(text: str, *, emails: bool = True, pii: bool = False) -> str:
    """Redact secrets (always), emails (default), and other PII (opt-in)."""
    redacted = text
    for _name, pattern, replacement in SECRET_PATTERNS:
        redacted = pattern.sub(replacement, redacted)
    redacted = ASSIGNMENT_PATTERN.sub(_redact_assignment, redacted)
    if emails:
        redacted = EMAIL_PATTERN.sub("[EMAIL]", redacted)
    if pii:
        for _name, pattern, replacement in PII_PATTERNS:
            redacted = pattern.sub(replacement, redacted)
    return redacted


def contains_sensitive(text: str, *, emails: bool = True, pii: bool = False) -> bool:
    """Return True when :func:`redact_text` would change the text."""
    return redact_text(text, emails=emails, pii=pii) != text


def redact_secrets(text: str) -> str:
    """Return the text with credentials, tokens, keys, and emails replaced."""
    return redact_text(text, emails=True, pii=False)


def redact_sync_value(value: Any) -> Any:
    """Recursively redact strings inside JSON-like sync payload values.

    Values stored under sensitive keys (password, token, api_key, cookie, ...)
    are replaced wholesale, whatever their type.
    """
    if isinstance(value, str):
        return redact_secrets(value)
    if isinstance(value, Mapping):
        out: dict[str, Any] = {}
        for key, item in value.items():
            if is_sensitive_key(key) and item is not None and not isinstance(item, bool):
                out[str(redact_secrets(str(key)))] = _redact_whole(item)
            else:
                out[str(redact_secrets(str(key)))] = redact_sync_value(item)
        return out
    if isinstance(value, tuple):
        return tuple(redact_sync_value(item) for item in value)
    if isinstance(value, list):
        return [redact_sync_value(item) for item in value]
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return [redact_sync_value(item) for item in value]
    return value


def _redact_whole(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _redact_whole(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_redact_whole(v) for v in value]
    if value is None or isinstance(value, bool):
        return value
    return REDACTED


__all__ = [
    "ASSIGNMENT_PATTERN",
    "EMAIL_PATTERN",
    "PII_PATTERNS",
    "REDACTED",
    "SECRET_PATTERNS",
    "SENSITIVE_KEY_RE",
    "contains_sensitive",
    "is_sensitive_key",
    "redact_secrets",
    "redact_sync_value",
    "redact_text",
]
