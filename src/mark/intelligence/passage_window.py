"""Best-passage windowing: show the part of a long memory that answers the query.

Rendering used to drop a whole result when it did not fit the budget, so a
3,500-word document holding the answer produced an empty context. Now a
long memory is cut to a window of at most ``max_chars`` placed where the
query's most informative terms meet (stemmed, like retrieval), snapped to sentence
or line boundaries when one is near, and marked with "…" where text was
left out. A memory with no query term in it shows its beginning.
"""
from __future__ import annotations

import re

from mark.intelligence.query_understanding import _TOKEN_RE, _content_terms, _stem

_ELLIPSIS = "…"
# How far (in chars) a window edge may move to land on a sentence boundary.
_SNAP = 80
_BOUNDARY_RE = re.compile(r"(?<=[.!?;:])\s+|\n+")


def window(content: str, query: str, max_chars: int) -> tuple[str, bool]:
    """(text, windowed): *content* itself when it fits, else its best window."""
    if max_chars <= 0:
        return "", bool(content)
    if len(content) <= max_chars:
        return content, False
    body = max(1, max_chars - 2)  # room for the two ellipsis marks
    start = _best_start(content, set(_content_terms(query)), body)
    start, end = _snap(content, start, min(len(content), start + body))
    text = content[start:end].strip()
    return f"{_ELLIPSIS if start > 0 else ''}{text}{_ELLIPSIS if end < len(content) else ''}", True


def _best_start(content: str, terms: set[str], size: int) -> int:
    """Start offset of the *size*-char span covering the most informative query terms.

    A span scores the sum of its *distinct* query terms, each weighted by
    rarity in this memory (1 / occurrences): a name repeated on every line
    of a story says little; the one sentence with "brass key" says a lot.
    """
    if not terms:
        return 0
    hits = [(m.start(), stem) for m in _TOKEN_RE.finditer(content)
            if (stem := _stem(m.group().lower())) in terms]
    if not hits:
        return 0
    occurrences: dict[str, int] = {}
    for _, stem in hits:
        occurrences[stem] = occurrences.get(stem, 0) + 1
    weight = {stem: 1.0 / n for stem, n in occurrences.items()}
    inside: dict[str, int] = {}
    score, best_score, best = 0.0, -1.0, (0, 0)
    j = 0
    for i, (first, _) in enumerate(hits):
        while j < len(hits) and hits[j][0] < first + size:
            stem = hits[j][1]
            inside[stem] = inside.get(stem, 0) + 1
            if inside[stem] == 1:
                score += weight[stem]
            j += 1
        if score > best_score:
            best_score, best = score, (i, j - 1)
        stem = hits[i][1]
        inside[stem] -= 1
        if inside[stem] == 0:
            score -= weight[stem]
    # Centre the window on the rarest terms of the best span (the sentence
    # that is about the question), not on where the span happens to start.
    span = hits[best[0]: best[1] + 1]
    top = max(weight[stem] for _, stem in span)
    focus = [pos for pos, stem in span if weight[stem] >= top / 2]
    centre = (min(focus) + max(focus)) // 2
    return max(0, min(centre - size // 2, len(content) - size))


def _snap(content: str, start: int, end: int) -> tuple[int, int]:
    """Move window edges onto nearby sentence / line boundaries (never growing it)."""
    if start > 0:
        boundary = _BOUNDARY_RE.search(content, start, min(len(content), start + _SNAP))
        if boundary is not None and boundary.end() < end:
            start = boundary.end()
    if end < len(content):
        tail = content[max(start, end - _SNAP):end]
        cuts = list(_BOUNDARY_RE.finditer(tail))
        if cuts:
            candidate = end - len(tail) + cuts[-1].start()
            if candidate > start:
                end = candidate
    return start, end
