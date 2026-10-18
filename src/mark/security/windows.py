# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 MARK Contributors
"""Run a regex only around the places a literal hint occurs.

Python's ``re`` tries an alternation at every position of the text, so a
screen with a few dozen phrase patterns costs seconds on a 1 MB document even
when nothing matches. Every screening pattern has *hints*: literals at least
one of which occurs inside any match, and matches have bounded length. On
long text we therefore find the hints with ``str.find`` (C speed), merge
``±radius`` windows around them and search only those windows. Short text
(the common case for memories) is searched whole, exactly as before.
"""
from __future__ import annotations

import re
from collections.abc import Iterable, Iterator

RADIUS = 600   # longer than any screening match (gaps are bounded well below this)


def _spans(lowered: str, hints: Iterable[str], radius: int) -> list[tuple[int, int]]:
    size, spans = len(lowered), []
    for hint in hints:
        at = lowered.find(hint)
        while at != -1:
            spans.append((max(0, at - radius), min(size, at + len(hint) + radius)))
            at = lowered.find(hint, at + len(hint))
    spans.sort()
    merged: list[tuple[int, int]] = []
    for start, end in spans:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def finditer_near(pattern: "re.Pattern[str]", text: str, hints: Iterable[str], lowered: str | None = None,
                  radius: int = RADIUS) -> Iterator["re.Match[str]"]:
    """``pattern.finditer(text)``, restricted to windows around *hints* when *text* is long.

    *lowered* is ``text.lower()`` (computed once by the caller); when lowering
    changed the length (rare Unicode case folds) the whole text is searched.
    """
    lowered = text.lower() if lowered is None else lowered
    hints = tuple(hints)
    # Short text, case folds that changed the length, or hints so dense that
    # the windows would cover the text anyway: search it whole.
    if len(text) <= 4 * radius or len(lowered) != len(text) \
            or sum(lowered.count(hint) for hint in hints) * radius >= len(text):
        yield from pattern.finditer(text)
        return
    for start, end in _spans(lowered, hints, radius):
        yield from pattern.finditer(text, start, end)


def search_near(pattern: "re.Pattern[str]", text: str, hints: Iterable[str], lowered: str | None = None,
                radius: int = RADIUS) -> "re.Match[str] | None":
    """The first match of :func:`finditer_near`, or None."""
    return next(finditer_near(pattern, text, hints, lowered, radius), None)


__all__ = ["RADIUS", "finditer_near", "search_near"]
