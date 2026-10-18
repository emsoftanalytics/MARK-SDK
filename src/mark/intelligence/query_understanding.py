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
from functools import lru_cache

from mark.intelligence.classifier import _STOP_WORDS
from mark.intelligence.date_parsing import find_dates
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
_TEMPORAL_WORDS = {"now", "current", "currently", "latest", "newest", "recent", "recently", "today",
                   "nowadays", "original", "originally", "initial", "initially", "first", "earliest",
                   "oldest", "last", "previous", "previously", "formerly", "still"}


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


# Everyday irregular past forms, so "How much rent do I pay?" matches "paid on the 1st".
_IRREGULAR = {"paid": "pay", "bought": "buy", "sold": "sell", "met": "meet", "went": "go", "gone": "go",
              "ran": "run", "taught": "teach", "brought": "bring", "kept": "keep",
              "lost": "lose", "spent": "spend", "sent": "send", "built": "build", "wrote": "write",
              "written": "write", "chose": "choose", "chosen": "choose", "began": "begin", "begun": "begin"}


@lru_cache(maxsize=65536)
def _stem(token: str) -> str:
    """Tiny suffix stemmer applied to a fixpoint, then final-e removal.

    owns/owned/owner → own, filter/filtered → filt, rate/rating → rat,
    database/databases → databas, boxes → box, policies → policy.
    Deliberately conservative (no prefix matching): "titan" never matches
    "titanium". Query and content are stemmed identically, so an occasional
    over-merge only widens a match.
    """
    if token in _IRREGULAR:
        return _stem(_IRREGULAR[token])
    if len(token) < 4 or not token.isalpha():
        return token
    previous = None
    for _ in range(3):
        if token == previous:
            break
        previous, token = token, _stem_once(token)
    if len(token) >= 4 and token.endswith("e"):
        token = token[:-1]
    return token


def _stem_once(token: str) -> str:
    """Strip one suffix (plural, -ies/-ied, -ment(s), -ing, -er(s), -ed); see _stem."""
    if len(token) < 4:
        return token
    if token.endswith(("ies", "ied")) and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith("es") and token[:-2].endswith(("ss", "x", "z", "ch", "sh")):
        return token[:-2]
    if token.endswith("s") and not token.endswith(("ss", "us", "is")):
        return token[:-1]
    for suffix in ("ments", "ment", "ing", "ers", "er", "ed"):
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            stem = token[: -len(suffix)]
            if len(stem) >= 4 and stem[-1] == stem[-2] and stem[-1] not in "lsz":
                stem = stem[:-1]  # running → runn → run
            return stem
    return token


def _content_terms(text: str) -> list[str]:
    """Salient query terms: lowercase stems without stopwords or temporal words."""
    terms: dict[str, None] = {}  # ordered set: linear even for long memories
    for token in _TOKEN_RE.findall(text.lower()):
        if token.endswith("'s"):
            token = token[:-2]
        if token in _STOP_WORDS or token in _TEMPORAL_WORDS or (len(token) < 2 and not token.isdigit()):
            continue
        terms.setdefault(_stem(token), None)
    if not text.isascii():
        # Emoji and other symbols are searchable terms ("🦉"), matched literally.
        from mark.embeddings.segments import is_symbol

        for char in text:
            if is_symbol(char):
                terms.setdefault(char, None)
    return list(terms)


@lru_cache(maxsize=1024)
def _content_stems(content: str) -> frozenset[str]:
    """Every stemmed token of a memory (stopwords kept: they cannot match query terms).

    Cached by content: one retrieval scores the same candidates in several
    stages (fusion, hops, ordering, facts).
    """
    tokens = set(_TOKEN_RE.findall(content.lower()))
    # snake_case identifiers also count as their words: "code_length" answers "code length".
    tokens |= {part for token in tokens if "_" in token for part in token.split("_") if len(part) > 1}
    return frozenset(_stem(token) for token in tokens)


def _coverage(query_terms: list[str], content: str) -> float:
    """Fraction of the query's content terms that occur (stemmed) in the memory.

    Terms an English stemmer cannot split are matched literally: a symbol
    ("🦉") as a substring, and a CJK run ("我的中文老师是谁") by the share of its
    character bigrams found in the memory, since CJK text has no spaces and
    the run is rarely written out whole.
    """
    if not query_terms:
        return 0.0
    stems = _content_stems(content)
    lowered: str | None = None
    found = 0.0
    for term in query_terms:
        if term in stems:
            found += 1
        elif not term.isascii():
            lowered = content.lower() if lowered is None else lowered
            found += _literal_share(term, lowered)
    return found / len(query_terms)


def _literal_share(term: str, content: str) -> float:
    """1 when *term* occurs in *content*; for a CJK run, the share of its bigrams that do."""
    if term in content:
        return 1.0
    from mark.embeddings.segments import CJK_RUN, cjk_bigrams

    if len(term) < 3 or CJK_RUN.search(term) is None:
        return 0.0
    grams = cjk_bigrams(term)
    return sum(1 for gram in grams if gram in content) / len(grams)


def _relevance(query_terms: list[str], content: str, cosine: float) -> float:
    """Calibrated query evidence in [0, 1]: max of vector similarity and term coverage.

    Either signal alone can be decisive: a paraphrase scores on cosine, an
    exact identifier on coverage. Taking the max (not a blend) keeps the
    value comparable to ``RetrievalPolicySpec.min_confidence`` regardless of
    which channel found the memory.
    """
    return max(0.0, min(1.0, max(cosine, _coverage(query_terms, content))))
