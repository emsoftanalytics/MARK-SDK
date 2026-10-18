"""Find the known fact subjects a text names, fast enough for ten thousand subjects.

``_matching_facts`` asks on every recall which known subjects the query (or
a top memory) mentions. Testing each subject with a whole-token regex cost
one compiled pattern per subject: 1-3 s per recall with ~10k subjects.

:class:`SubjectIndex` normalises the subjects once (cached per subject
list) and indexes them by their first token. A text is tokenised once and
only subjects whose first token occurs in it are compared, token by token:
the same whole-token rule as ``_contains_anchor``. Subjects or texts with
CJK characters, where words are not space-separated, use
``_contains_anchor`` itself, behind a cheap substring test.
"""
from __future__ import annotations

from functools import lru_cache

from mark.intelligence.query_anchors import _anchor_norm, _contains_anchor


class SubjectIndex:
    """Normalised subjects indexed by first token (see module docstring)."""

    __slots__ = ("_by_first", "_scan")

    def __init__(self, subjects: tuple[str, ...]) -> None:
        self._by_first: dict[str, list[tuple[tuple[str, ...], str]]] = {}
        # Non-ASCII subjects (CJK names) are checked with _contains_anchor.
        self._scan: list[tuple[str, str]] = []
        for subject in subjects:
            norm = _anchor_norm(subject)
            if not norm:
                continue
            if norm.isascii():
                tokens = tuple(norm.split())
                self._by_first.setdefault(tokens[0], []).append((tokens, subject))
            else:
                self._scan.append((norm, subject))

    def named_in(self, text_norm: str) -> list[str]:
        """Subjects that occur as whole tokens in *text_norm* (an ``_anchor_norm`` text), in subject order."""
        found: dict[str, None] = {}
        tokens = text_norm.split()
        for i, token in enumerate(tokens):
            for subject_tokens, subject in self._by_first.get(token, ()):
                if tuple(tokens[i:i + len(subject_tokens)]) == subject_tokens:
                    found[subject] = None
        if not text_norm.isascii():
            # "我在acme工作": an ASCII subject glued to CJK text is not a separate token.
            for first, entries in self._by_first.items():
                if first in text_norm:
                    for subject_tokens, subject in entries:
                        if subject not in found and _contains_anchor(text_norm, " ".join(subject_tokens)):
                            found[subject] = None
        for norm, subject in self._scan:
            if norm in text_norm and _contains_anchor(text_norm, norm):
                found[subject] = None
        return list(found)


@lru_cache(maxsize=8)
def subject_index(subjects: tuple[str, ...]) -> SubjectIndex:
    """The index for one list of known subjects (rebuilt only when the list changes)."""
    return SubjectIndex(subjects)
