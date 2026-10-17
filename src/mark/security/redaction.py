# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 MARK Contributors
"""Secret and PII redaction: the policy object, the engine, legacy helpers.

This module owns the single shared rule set used by write-time redaction
(:mod:`mark.security.write_policy`), the governance
:class:`~mark.middlewares.governance.sanitizer.ContentSanitizer` and sync
redaction (:func:`redact_secrets`, :func:`redact_sync_value`).

Defaults (``RedactionPolicy()``): credentials are redacted (API keys,
tokens, private keys, passwords, card numbers, PINs); emails, phone numbers,
IBANs and SSNs are kept unless ``pii=True``. Secrets give an attacker
access; contact identifiers are what agents are asked to remember.
"""
from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from mark.security.pii_rules import EMAIL_PATTERN, EMAIL_RULE, OTHER_PII_RULES, PHONE_RULES
from mark.security.rules import Rule
from mark.security.secret_rules import SECRET_ASSIGNMENT
from mark.security.secret_rules_prose import ALL_SECRET_RULES

REDACTED = "[REDACTED]"

# Every placeholder MARK writes; fact extraction must never store one as a value.
PLACEHOLDER_RE = re.compile(r"\[(?:SECRET:[a-z0-9_]+|REDACTED[A-Z_]*|EMAIL|PHONE|IBAN|SSN|CC)\]")


@dataclass(frozen=True)
class RedactionPolicy:
    """What write-time redaction removes.

    ``secrets``: credentials (default on). ``pii``: emails, phones, IBANs,
    SSNs (default off). ``emails`` / ``phones`` override ``pii`` for that
    one class when not None. ``extra_patterns``: ``(kind, regex)`` pairs
    redacted as ``[SECRET:<kind>]``.
    """

    secrets: bool = True
    pii: bool = False
    emails: bool | None = None
    phones: bool | None = None
    extra_patterns: tuple[tuple[str, "re.Pattern[str]"], ...] = field(default=())

    def __post_init__(self) -> None:
        # Accept a list for convenience but keep the policy hashable.
        object.__setattr__(self, "extra_patterns", tuple(tuple(p) for p in self.extra_patterns))

    def rules(self) -> list[Rule]:
        """Return the ordered rules this policy applies."""
        out: list[Rule] = []
        if self.secrets:
            out.extend(ALL_SECRET_RULES)
        out.extend(Rule(kind, pattern) for kind, pattern in self.extra_patterns)
        if self.emails if self.emails is not None else self.pii:
            out.append(EMAIL_RULE)
        if self.pii:
            out.extend(OTHER_PII_RULES)  # IBAN before phones: IBAN digits look phone-like
        if self.phones if self.phones is not None else self.pii:
            out.extend(PHONE_RULES)
        return out


@dataclass(frozen=True)
class RedactionResult:
    """Redacted text plus the sorted, de-duplicated kinds that were removed."""

    text: str
    kinds: tuple[str, ...] = ()

    @property
    def changed(self) -> bool:
        return bool(self.kinds)


DEFAULT_POLICY = RedactionPolicy()


@dataclass(frozen=True)
class _Compiled:
    """A policy's rules plus every hint keyword, checked once per text."""

    rules: tuple[Rule, ...]
    hints: tuple[str, ...]


_RULE_CACHE: dict[RedactionPolicy, _Compiled] = {}


def _compiled(policy: RedactionPolicy) -> _Compiled:
    compiled = _RULE_CACHE.get(policy)
    if compiled is None:
        rules = tuple(policy.rules())
        hints = tuple(sorted({h for rule in rules for h in rule.hint}))
        compiled = _RULE_CACHE.setdefault(policy, _Compiled(rules, hints))
    return compiled


def redact(text: str, policy: RedactionPolicy | None = None) -> RedactionResult:
    """Apply *policy* (default: secrets only) to *text*.

    Deterministic and idempotent: placeholders never match a rule, so
    redacting twice gives the same text.
    """
    compiled = _compiled(policy or DEFAULT_POLICY)
    if not text:
        return RedactionResult(text or "")
    found: list[str] = []
    lowered = text.lower()
    # One pass over all hint keywords; a rule whose hints are all absent is
    # skipped without touching its regex (the common case for most memories).
    present = {hint for hint in compiled.hints if hint in lowered}
    out = text
    for rule in compiled.rules:
        if rule.hint and present.isdisjoint(rule.hint):
            continue
        out = rule.run(out, found)
    return RedactionResult(out, tuple(sorted(set(found))))


# ── legacy helpers (sync, governance) ─────────────────────────────────────────

def redact_text(text: str, *, emails: bool = True, pii: bool = False) -> str:
    """Redact secrets (always), emails (default), and other PII (opt-in)."""
    return redact(text, RedactionPolicy(pii=pii, emails=emails)).text


def contains_sensitive(text: str, *, emails: bool = True, pii: bool = False) -> bool:
    """Return True when :func:`redact_text` would change the text."""
    return redact_text(text, emails=emails, pii=pii) != text


def redact_secrets(text: str) -> str:
    """Return the text with credentials, tokens, keys, and emails replaced."""
    return redact_text(text, emails=True, pii=False)


# Key names whose values are always secret in mappings. ``auth(?!ors?\b)``
# keeps "author"/"authors" readable while matching "auth", "authorization".
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


# Deprecated aliases kept for 0.2.0a9 imports.
SECRET_PATTERNS = [(rule.kind, rule.pattern) for rule in ALL_SECRET_RULES]
PII_PATTERNS = [(rule.kind, rule.pattern) for rule in [*OTHER_PII_RULES, *PHONE_RULES]]
ASSIGNMENT_PATTERN = SECRET_ASSIGNMENT

__all__ = [
    "ASSIGNMENT_PATTERN",
    "DEFAULT_POLICY",
    "EMAIL_PATTERN",
    "PII_PATTERNS",
    "PLACEHOLDER_RE",
    "REDACTED",
    "RedactionPolicy",
    "RedactionResult",
    "SECRET_PATTERNS",
    "SENSITIVE_KEY_RE",
    "contains_sensitive",
    "is_sensitive_key",
    "redact",
    "redact_secrets",
    "redact_sync_value",
    "redact_text",
]
