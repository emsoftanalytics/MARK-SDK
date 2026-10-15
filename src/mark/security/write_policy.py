# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 MARK Contributors
"""The write-time security policy applied to every stored memory.

:class:`WritePolicy` bundles redaction and prompt-injection screening and is
applied by ``GuardMixin._guarded_fragment`` (memory/facade/write_guard.py)
before a fragment is built, so store, batch store, observe, gap healing and
explicit quarantine all share it. Defaults:

* redaction **on** with :class:`RedactionPolicy` defaults (credentials out,
  contact PII kept) — opt out with ``redact=False``;
* injection screening ``"tag"``: never blocks, adds ``mark:untrusted`` plus
  ``metadata["mark_security"]["injection"]`` reasons; ``"quarantine"``
  stores the memory QUARANTINED instead; ``"off"`` disables screening.

Findings are recorded under ``metadata["mark_security"]`` only when
something happened, so clean memories are stored exactly as given.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Union

from mark.security.injection import UNTRUSTED_TAG, screen_injection
from mark.security.redaction import RedactionPolicy, redact

InjectionMode = Literal["tag", "quarantine", "off"]
_MODES = ("tag", "quarantine", "off")


@dataclass(frozen=True)
class WriteDecision:
    """What the write path must store after applying a :class:`WritePolicy`."""

    content: str
    metadata: dict[str, Any] | None
    tags: list[str] | None
    quarantine: bool = False
    reason: str = ""


@dataclass(frozen=True)
class WritePolicy:
    """Redaction + injection screening applied at write time."""

    redaction: RedactionPolicy | None = field(default_factory=RedactionPolicy)
    injection: InjectionMode = "tag"

    def __post_init__(self) -> None:
        if self.injection not in _MODES:
            raise ValueError(f"injection must be one of {_MODES}, got {self.injection!r}")

    def apply(
        self,
        content: str,
        *,
        metadata: dict[str, Any] | None = None,
        tags: list[str] | None = None,
    ) -> WriteDecision:
        """Return the content/metadata/tags to store and whether to quarantine.

        Inputs are never mutated.
        """
        findings: dict[str, Any] = {}
        if self.redaction is not None:
            result = redact(content, self.redaction)
            content = result.text
            kinds = set(result.kinds)
            if metadata:
                metadata, meta_kinds = _redact_metadata(metadata, self.redaction)
                kinds |= meta_kinds
            if kinds:
                findings["redacted"] = sorted(kinds)

        quarantine, reason = False, ""
        if self.injection != "off":
            report = screen_injection(content)
            if report.flagged:
                findings["untrusted"] = True
                findings["injection"] = list(report.reasons)
                tags = [*(tags or []), UNTRUSTED_TAG] if UNTRUSTED_TAG not in (tags or []) else tags
                if self.injection == "quarantine":
                    quarantine = True
                    reason = "prompt injection: " + ", ".join(report.reasons)
                    findings["quarantined"] = True

        if findings:
            previous = dict((metadata or {}).get("mark_security") or {})
            metadata = {**(metadata or {}), "mark_security": {**previous, **findings}}
        return WriteDecision(content, metadata, tags, quarantine, reason)


def _redact_metadata(value: Any, policy: RedactionPolicy) -> tuple[Any, set[str]]:
    """Redact string values (recursively) in caller metadata."""
    kinds: set[str] = set()

    def walk(item: Any) -> Any:
        if isinstance(item, str):
            result = redact(item, policy)
            kinds.update(result.kinds)
            return result.text
        if isinstance(item, dict):
            return {key: walk(val) for key, val in item.items()}
        if isinstance(item, (list, tuple)):
            return type(item)(walk(val) for val in item)
        return item

    return walk(value), kinds


DEFAULT_WRITE_POLICY = WritePolicy()
RedactOption = Union[bool, None, RedactionPolicy, WritePolicy]


def resolve_write_policy(redact: RedactOption = True, injection: InjectionMode = "tag") -> WritePolicy:
    """Turn the public ``redact=`` / ``injection=`` options into a WritePolicy.

    ``redact``: True/None (default policy), False (no redaction), a
    :class:`RedactionPolicy` (custom rules) or a full :class:`WritePolicy`
    (returned as-is; ``injection`` is then ignored).
    """
    if isinstance(redact, WritePolicy):
        return redact
    if isinstance(redact, RedactionPolicy):
        return WritePolicy(redaction=redact, injection=injection)
    if redact is None or redact is True:
        if injection == "tag":
            return DEFAULT_WRITE_POLICY
        return WritePolicy(injection=injection)
    if redact is False:
        return WritePolicy(redaction=None, injection=injection)
    raise TypeError(f"redact must be a bool, RedactionPolicy or WritePolicy, got {type(redact).__name__}")


__all__ = [
    "DEFAULT_WRITE_POLICY",
    "InjectionMode",
    "RedactOption",
    "WriteDecision",
    "WritePolicy",
    "resolve_write_policy",
]
