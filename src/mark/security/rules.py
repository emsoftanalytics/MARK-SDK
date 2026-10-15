# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 MARK Contributors
"""The redaction rule type and the engine that applies one rule to text.

A :class:`Rule` is a compiled regex plus an optional validator. Rules redact
either the whole match or only the named value group (``v``, ``vq`` or
``vs``), so ``password=…`` keeps its key and ``postgres://admin:…@host`` keeps
the user and host: the memory still says *which* credential existed.
Placeholders are deterministic (``[SECRET:<kind>]``) so redaction is
idempotent and recall can still answer "is there a staging Stripe key?".
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Union

# A validator returns False to keep the text, True to redact the value, or an
# int to redact only the first N characters of the whole match (used when a
# greedy match swallowed trailing words, e.g. an IBAN followed by "EUR").
Validator = Callable[["re.Match[str]"], Union[bool, int]]
KindFn = Callable[["re.Match[str]"], str]

_VALUE_GROUPS = ("v", "vq", "vs")


def secret_placeholder(kind: str) -> str:
    """Return the placeholder that replaces a secret of *kind*."""
    return f"[SECRET:{kind}]"


@dataclass(frozen=True)
class Rule:
    """One redaction rule: pattern, kind label, placeholder and validator."""

    kind: str
    pattern: "re.Pattern[str]"
    placeholder: str = ""          # "" means [SECRET:<kind>]
    validate: Validator | None = None
    kind_of: KindFn | None = None  # dynamic kind (e.g. from the key name)
    # Cheap pre-check: lower-case substrings, at least one must occur in the
    # text for the regex to run. Most memories contain none, so most rules
    # cost one C-level substring scan instead of a regex pass.
    hint: str | tuple[str, ...] = ()
    # Cheap regex pre-check for rules without a keyword (card numbers): a
    # pattern that starts with a literal class lets re skip ahead in C.
    prefilter: "re.Pattern[str] | None" = None
    # Require a non-word character before the match. Done here rather than
    # with a leading ``\b`` because a pattern that starts with a literal
    # ("sk-ant-", "AKIA") lets re jump between candidates in C: ~30x faster.
    word_start: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.hint, str):
            object.__setattr__(self, "hint", (self.hint,) if self.hint else ())

    def apply(self, text: str, found: list[str], lowered: str | None = None) -> str:
        """Return *text* with this rule applied; append kinds to *found*.

        *lowered* is the lower-cased original text, computed once per call
        by the engine so hint checks stay O(rules) string searches.
        """
        if self.hint:
            low = lowered if lowered is not None else text.lower()
            if not any(h in low for h in self.hint):
                return text  # fast path: most rules cannot match most memories
        return self.run(text, found)

    def run(self, text: str, found: list[str]) -> str:
        """Apply the rule without the keyword pre-check (caller did it)."""
        if self.prefilter is not None and not self.prefilter.search(text):
            return text
        return self.pattern.sub(lambda m: self._replace(m, found), text)

    def _replace(self, match: "re.Match[str]", found: list[str]) -> str:
        start = match.start()
        if self.word_start and start and (match.string[start - 1].isalnum() or match.string[start - 1] == "_"):
            return match.group(0)
        verdict = True if self.validate is None else self.validate(match)
        if verdict is False:
            return match.group(0)
        kind = self.kind_of(match) if self.kind_of is not None else self.kind
        token = self.placeholder or secret_placeholder(kind)
        found.append(kind)
        whole = match.group(0)
        if verdict is not True and isinstance(verdict, int):
            return token + whole[verdict:]
        names = match.re.groupindex
        for name in _VALUE_GROUPS:
            if name in names and match.group(name) is not None:
                start = match.start(name) - match.start()
                end = match.end(name) - match.start()
                return whole[:start] + token + whole[end:]
        return token


def value_of(match: "re.Match[str]") -> str:
    """Return the captured value group of an assignment-style match."""
    for name in _VALUE_GROUPS:
        if name in match.re.groupindex and match.group(name) is not None:
            return match.group(name)
    return match.group(0)


__all__ = ["KindFn", "Rule", "Validator", "secret_placeholder", "value_of"]
