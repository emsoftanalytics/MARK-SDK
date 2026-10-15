"""Pick the facts about a subject that a query actually asks for.

``RetrievalResult.facts`` used to fall back to *every* fact about a subject
whenever no attribute name matched the query, so "Which region is atlas-core
in now?" returned its port and owner. Selection per subject, in order:

1. attributes named in the query (stemmed: "ports" -> port, "owns" -> owner)
   are returned, and only those;
2. otherwise, if the question has a head noun ("Which *region* ...", "What
   *port* does ...", "What is the *region* of ...") the subject's facts are
   kept only when their value mentions it ("Which team ..." -> lead =
   "platform team"); a head noun matching nothing means the facts hold no
   answer, so none are returned (the memories still answer it);
3. otherwise (open questions: "Tell me about atlas-core", "What is
   atlas-core?") every fact about the subject is kept.

Only the subject's real attribute names and values are matched, through the
same stemmer as retrieval, so no attribute vocabulary is hard-coded.
"""
from __future__ import annotations

import re
from typing import Any, Sequence

from mark.intelligence.classifier import _STOP_WORDS
from mark.intelligence.query_understanding import _TEMPORAL_WORDS, _content_stems, _content_terms, _stem

_AUX = r"(?:is|are|was|were|does|do|did|has|have|had|will|can|could|should|would)"
# Question head nouns. "what" needs a following auxiliary (or "is the ...")
# so "What happened to X?" is not read as asking for a "happen" attribute.
_HEAD_RES = (
    re.compile(r"\bwhich\s+(\w+)", re.I),
    re.compile(r"\bwhat(?:\s+(?:is|are|was|were)|['’]s)\s+(?:the|its|their)\s+(\w+)", re.I),
    re.compile(rf"\bwhat\s+(\w+)\s+{_AUX}\b", re.I),
)


def question_head(query: str, subject: str = "") -> str | None:
    """Stem of the noun a which/what question asks for, or None.

    Stopwords, temporal words and words of the subject itself ("What is
    atlas-core ...") are not heads.
    """
    subject_stems = _content_stems(subject)
    for pattern in _HEAD_RES:
        match = pattern.search(query)
        if match is None:
            continue
        word = match.group(1).lower()
        if word in _STOP_WORDS or word in _TEMPORAL_WORDS or word.isdigit():
            return None
        stem = _stem(word)
        return None if stem in subject_stems else stem
    return None


def select_facts(query: str, subject: str, rows: Sequence[Any]) -> list[Any]:
    """Filter one subject's current fact rows (``attribute``/``value`` keys) for *query*."""
    terms = set(_content_terms(query))
    named = [row for row in rows if set(_content_terms(row["attribute"])) & terms]
    if named:
        return named
    head = question_head(query, subject)
    if head is None:
        return list(rows)
    return [row for row in rows if head in _content_stems(str(row["value"]))]
