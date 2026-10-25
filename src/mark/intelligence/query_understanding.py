"""Query understanding: temporal intent, anchors, stemming and lexical evidence.

Pure functions over strings, shared by retrieval, temporal ordering and the
context watch:

- ``detect_temporal_intent`` decides whether a query asks about the present
  ("current port", "the last bug") or the beginning ("original owner",
  "what did we use previously");
- ``query_anchors`` / ``gap_anchors`` (module ``query_anchors``, re-exported
  here) extract distinctive identifiers and names whose absence from memory
  is a real gap, not just a weak match;
- ``_stem`` / ``_content_terms`` / ``_relevance`` produce the calibrated
  query-evidence score that drives gap detection and reinforcement.
"""
from __future__ import annotations

import re
from enum import Enum

from mark.intelligence.date_parsing import find_dates
from mark.intelligence.term_evidence import (  # noqa: F401  (historical import location)
    _IRREGULAR,
    _TEMPORAL_WORDS,
    _camel_parts,
    _content_stems,
    _content_terms,
    _coverage,
    _literal_share,
    _long_numbers,
    _relevance,
    _stem,
    _stem_once,
)
from mark.intelligence.query_anchors import (  # noqa: F401  (historical import location)
    _BARE_NUMBER_RE,
    _COMPOUND_RE,
    _DIGIT_TOKEN_RE,
    _NAME_LEAD_WORDS,
    _NAME_RE,
    _NUMBER_WITH_UNIT_RE,
    _POSSESSIVE_RE,
    _TOKEN_RE,
    _anchor_norm,
    _contains_anchor,
    _is_identifier,
    anchor_present,
    gap_anchors,
    query_anchors,
)

# Words that express *when* the caller cares about, not *what*: they drive
# temporal ordering and must not count as content terms. "last" alone means
# the latest ("the last bug"); "last week" is a date range and is removed
# before intent detection.
_CURRENT_RE = re.compile(
    r"\b(now|current(?:ly)?|latest|newest|most\s+recent(?:ly)?|recent(?:ly)?|today|nowadays|these days"
    r"|as of (?:now|today)|still|any\s?more|last(?!\s*(?:[?.!]|$|for\b|long)))\b",
    re.I,
)
_ORIGINAL_RE = re.compile(
    r"\b(original(?:ly)?|initial(?:ly)?|first|earliest|oldest|at first|before any|previous(?:ly)?"
    r"|formerly|to begin with|in the beginning|(?:['’]s|\bmy|\bhis|\bher|\btheir|\bour)\s+old"
    r"|(?<!\bis )(?<!\bare )(?<!\bwas )(?<!\bwere )(?<!\bbe )(?<!\bget )used to"
    r"|before(?:\s+(?:that|then|the\s+(?:change|switch|move|migration)))?(?=\s*[?.!]*\s*$))\b",
    re.I,
)
# Phrases that contain a temporal word without asking about time:
# "first name", "first to respond", "max current of the PSU", "last time".
_FALSE_TEMPORAL_RE = re.compile(
    r"\b(?:first|last)\s+(?:names?|times?|steps?|place|line|page|item|of\s+all|to|call)\b"
    r"|\bcurrent\s+(?:draw|rating|limit|density|flow|source|sensor|consumption|of\s+\d)"
    r"|\bmax(?:imum)?\s+current\b",
    re.I,
)


class TemporalIntent(str, Enum):
    """What point in time a query asks about."""
    NONE = "none"
    CURRENT = "current"
    ORIGINAL = "original"


def detect_temporal_intent(query: str) -> TemporalIntent:
    """CURRENT for "now/current/latest/last" queries, ORIGINAL for "original/first/previously", else NONE.

    Date expressions ("last week", "next Tuesday") are ranges, not an ask for
    the latest record, so they are removed first. When a query mentions both
    ("was the original port changed? what is it now?") the present-tense ask
    wins: it is what the caller acts on.
    """
    cleaned = query
    for span in reversed(find_dates(query)):
        if span.text.lower() != "today":
            cleaned = cleaned[: span.pos[0]] + " " + cleaned[span.pos[1]:]
    cleaned = _FALSE_TEMPORAL_RE.sub(" ", cleaned)
    if _CURRENT_RE.search(cleaned):
        return TemporalIntent.CURRENT
    if _ORIGINAL_RE.search(cleaned):
        return TemporalIntent.ORIGINAL
    return TemporalIntent.NONE


def is_blank_query(query: str) -> bool:
    """True for empty, whitespace-only or punctuation-only queries (nothing to search for).

    Emoji and other non-ASCII symbols are searchable content ("🦉"), not punctuation.
    """
    text = query or ""
    if _TOKEN_RE.search(text):
        return False
    from mark.embeddings.segments import is_symbol

    return not any(is_symbol(char) for char in text)
