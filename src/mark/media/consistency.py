# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 MARK Contributors
"""Compare what a describer observed in an image with what the facts expect.

The describer is pluggable (``describe(path) -> dict | str``): in production a
vision model, in tests a deterministic fake. This module only *compares*; it
never fills gaps. Rules, in order:

1. an attribute the describer did not mention (or answered with "unknown",
   "unclear", ...) is ``unverified`` -- never a match;
2. values match when equal after normalisation (case, punctuation, articles),
   when the caller's synonym groups put them together, or when every word of
   the expected value appears in a non-negated observation ("dark red" covers
   "red"; "not red" does not);
3. anything else is a ``mismatch`` and carries the observed text verbatim.

Dict observations are preferred (``{"scarf": "blue"}`` or nested per subject
``{"Elena": {"scarf": "blue"}}``). Free text is supported conservatively: an
attribute counts as observed only in a clause that names it, and a mismatch
reports that clause as the observation.
"""
from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from typing import Any
from mark.security.lazy_regex import lazy_compile

Synonyms = Mapping[str, Iterable[str]] | Iterable[Iterable[str]] | None

_WORD = lazy_compile(r"[a-z0-9]+")
_ARTICLES = frozenset({"a", "an", "the"})
_NEGATIONS = frozenset({"not", "no", "without", "never", "isn", "isnt", "nor"})
_UNKNOWN = frozenset({
    "", "unknown", "unclear", "not visible", "n a", "na", "none visible", "cannot tell",
    "can t tell", "not shown", "not sure", "unsure", "unspecified", "null", "none",
})
_CLAUSE = lazy_compile(r"[.;\n]+|,(?=\s)")


def normalize(text: Any) -> str:
    """Lower-case words without punctuation or articles: 'The Red  Scarf!' -> 'red scarf'."""
    words = _WORD.findall(str(text).lower().replace("_", " "))
    return " ".join(w for w in words if w not in _ARTICLES)


def synonym_groups(synonyms: Synonyms) -> list[frozenset[str]]:
    """Normalise the caller's synonyms into groups of interchangeable phrases.

    Accepts ``{"red": ["crimson", "scarlet"]}`` or ``[["red", "crimson"], ...]``.
    """
    if not synonyms:
        return []
    groups: list[frozenset[str]] = []
    if isinstance(synonyms, Mapping):
        pairs = [[key, *values] for key, values in synonyms.items()]
    else:
        pairs = [list(group) for group in synonyms]
    for group in pairs:
        cleaned = frozenset(normalize(item) for item in group if normalize(item))
        if len(cleaned) > 1:
            groups.append(cleaned)
    return groups


def _equivalents(phrase: str, groups: list[frozenset[str]]) -> set[str]:
    found = {phrase}
    for group in groups:
        if phrase in group:
            found |= group
    return found


def values_match(expected: Any, observed: Any, groups: list[frozenset[str]]) -> bool:
    """True when the observation supports the expected value (see module rules)."""
    want, seen = normalize(expected), normalize(observed)
    if not want or not seen:
        return False
    seen_words = seen.split()
    negated = any(word in _NEGATIONS for word in seen_words)
    for candidate in _equivalents(want, groups):
        if candidate == seen:
            return True
        # Containment covers "dark red" for "red" and synonyms inside phrases
        # ("crimson scarf" for "red" when red~crimson), but never "not red".
        if not negated and set(candidate.split()) <= set(seen_words):
            return True
    return False


def is_unknown(observed: Any) -> bool:
    """True for empty or explicitly-unknown observations."""
    return observed is None or normalize(observed) in _UNKNOWN


def observations_for(observed: Any, subject: str) -> Mapping[str, Any] | str:
    """Pick the part of a describer's answer that concerns ``subject``.

    A nested dict keyed by subject name is narrowed to that subject; a dict
    that names other subjects but not this one yields ``{}`` (nothing observed).
    """
    if isinstance(observed, Mapping):
        nested = {normalize(k): v for k, v in observed.items() if isinstance(v, Mapping)}
        if nested:
            return nested.get(normalize(subject), {})
        return observed
    return "" if observed is None else str(observed)


def lookup(
    observations: Mapping[str, Any] | str,
    attribute: str,
    groups: list[frozenset[str]],
    subject: str = "",
) -> Any:
    """Return what was observed for ``attribute``, or None when nothing was."""
    names = {name for name in _equivalents(normalize(attribute), groups) if name}
    if isinstance(observations, Mapping):
        for key, value in observations.items():
            if normalize(key) in names:
                return None if is_unknown(value) else value
        return None
    # Free text: only clauses that literally name the attribute (or a synonym).
    # Clauses that also name the subject win, so "Elena: red scarf, Marco:
    # blue scarf" is read per subject.
    naming = [c for c in _CLAUSE.split(observations) if any(f" {n} " in f" {normalize(c)} " for n in names)]
    who = normalize(subject)
    naming.sort(key=lambda c: 0 if who and f" {who} " in f" {normalize(c)} " else 1)
    for clause in naming:
        head, sep, tail = clause.partition(":")
        # "scarf: blue" -> "blue"; any other phrasing keeps the whole clause.
        value = tail.strip() if sep and normalize(head) in names else clause.strip()
        return None if is_unknown(value) else value
    return None


def compare(
    expected: Mapping[str, Any],
    observed: Any,
    subject: str,
    *,
    synonyms: Synonyms = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Return ``(matches, mismatches, unverified)`` for one subject's attributes."""
    groups = synonym_groups(synonyms)
    observations = observations_for(observed, subject)
    matches: list[dict[str, Any]] = []
    mismatches: list[dict[str, Any]] = []
    unverified: list[dict[str, Any]] = []
    for attribute, want in expected.items():
        seen = lookup(observations, attribute, groups, subject)
        entry = {"attribute": attribute, "expected": want, "observed": seen}
        if seen is None:
            unverified.append({**entry, "status": "unverified"})
        elif values_match(want, seen, groups):
            matches.append({**entry, "status": "match"})
        else:
            mismatches.append({**entry, "status": "mismatch"})
    return matches, mismatches, unverified
