"""Query understanding: temporal intent, anchors, stemming and lexical evidence.

Pure functions over strings, shared by retrieval, temporal ordering and the
context watch:

- ``detect_temporal_intent`` decides whether a query asks about the present
  ("current port") or the beginning ("original owner");
- ``query_anchors`` extracts distinctive identifiers and names whose absence
  from memory is a real gap, not just a weak match;
- ``_stem`` / ``_content_terms`` / ``_relevance`` produce the calibrated
  query-evidence score that drives gap detection and reinforcement.
"""
from __future__ import annotations

import re
from enum import Enum

from mark.intelligence.classifier import _STOP_WORDS

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)

# Words that express *when* the caller cares about, not *what*: they drive
# temporal ordering and must not count as content terms.
_CURRENT_RE = re.compile(r"\b(now|current(?:ly)?|latest|most recent|today|nowadays|these days|as of (?:now|today))\b", re.I)
_ORIGINAL_RE = re.compile(r"\b(original(?:ly)?|initial(?:ly)?|first|earliest|at first|before any)\b", re.I)
# Phrases that contain a temporal word without asking about time:
# "first name", "first to respond", "max current of the PSU".
_FALSE_TEMPORAL_RE = re.compile(
    r"\b(?:first|last)\s+(?:names?|times?|steps?|place|line|page|item|of\s+all|to)\b"
    r"|\bcurrent\s+(?:draw|rating|limit|density|flow|source|sensor|consumption|of\s+\d)"
    r"|\bmax(?:imum)?\s+current\b",
    re.I,
)
_TEMPORAL_WORDS = {"now", "current", "currently", "latest", "recent", "today", "nowadays",
                   "original", "originally", "initial", "initially", "first", "earliest"}

# Distinctive anchors: compound identifiers (atlas-core, HAL-1234, eu-west-1,
# v2.3.1), tokens with digits, and multi-word capitalised names.
_COMPOUND_RE = re.compile(r"\b\w+(?:[-_.:/]\w+)+\b")
_DIGIT_TOKEN_RE = re.compile(r"\b(?=\w*\d)\w{3,}\b")
# Bare numbers, decimals, dates and times ("8080", "2025", "1.5", "2026-03-01",
# "10:30") are values the query mentions, not subjects: a port or a year missing
# from memory is not a gap. Dotted quads (IP addresses) have four groups and stay.
_BARE_NUMBER_RE = re.compile(r"\d+(?:[.,:/-]\d+){0,2}")
_NUMBER_WITH_UNIT_RE = re.compile(r"\d+(?:s|st|nd|rd|th|ms|us|ns|k|kb|mb|gb|tb|m|h|x|px|pc|pct|am|pm)", re.I)
# Title-case words only ("Mei Lin", "Liam O'Connor"); ALL-CAPS log labels such
# as "TICKET HAL" are not names.
_NAME_RE = re.compile(r"\b([A-Z][a-z][\w'’]*(?:\s+[A-Z][a-z'’][\w'’]*)+)\b")
# Leading words of a title-cased run that are not part of a name
# ("Has Alice Smith moved?", "Tell Me About billing", "The Platform Team").
_NAME_LEAD_WORDS = {
    "who", "what", "which", "when", "where", "why", "how", "is", "are", "was", "were", "does", "do",
    "did", "has", "have", "had", "can", "could", "should", "would", "will", "shall", "may", "might",
    "must", "tell", "show", "give", "list", "find", "check", "please", "the", "a", "an", "in", "on",
    "at", "for", "from", "to", "of", "my", "our", "your", "their", "his", "her", "its", "me", "us",
    "about", "and", "or", "but", "so", "then", "if", "per", "also", "dr", "mr", "mrs", "ms",
}
_POSSESSIVE_RE = re.compile(r"(?<=\w)['’]s\b")


class TemporalIntent(str, Enum):
    """What point in time a query asks about."""
    NONE = "none"
    CURRENT = "current"
    ORIGINAL = "original"


def detect_temporal_intent(query: str) -> TemporalIntent:
    """CURRENT for "now/current/latest" queries, ORIGINAL for "original/first", else NONE.

    When a query mentions both ("was the original port changed? what is it
    now?") the present-tense ask wins: it is what the caller acts on.
    """
    cleaned = _FALSE_TEMPORAL_RE.sub(" ", query)
    if _CURRENT_RE.search(cleaned):
        return TemporalIntent.CURRENT
    if _ORIGINAL_RE.search(cleaned):
        return TemporalIntent.ORIGINAL
    return TemporalIntent.NONE


def query_anchors(query: str) -> list[str]:
    """Distinctive identifiers and names in a query, normalised to lowercase.

    Anchors are what the question is *about* (atlas-core, HAL-1234, Mei Lin).
    Three heuristics, in order: compound identifiers, tokens containing
    digits (minus quantities like "10ms" and bare numbers such as ports,
    years and dates, which are values rather than subjects), and runs of
    title-case words with leading/trailing question words stripped ("Has Alice Smith" -> "alice
    smith"). Possessives are dropped first so "Mei Lin's" matches "Mei Lin".
    """
    query = _POSSESSIVE_RE.sub("", query)
    anchors: list[str] = []
    for match in _COMPOUND_RE.findall(query):
        if _is_identifier(match) and not _BARE_NUMBER_RE.fullmatch(match):
            anchors.append(match)
    for match in _DIGIT_TOKEN_RE.findall(query):
        if _NUMBER_WITH_UNIT_RE.fullmatch(match) or _BARE_NUMBER_RE.fullmatch(match):
            continue  # "502s", "10ms", "3rd", "8080", "2025" are quantities, not identifiers
        if not any(match in compound for compound in anchors):
            anchors.append(match)
    for match in _NAME_RE.findall(query):
        words = match.split()
        while words and words[0].lower().rstrip(".") in _NAME_LEAD_WORDS:
            words = words[1:]
        while words and words[-1].lower() in _NAME_LEAD_WORDS:
            words = words[:-1]
        if len(words) >= 2:
            anchors.append(" ".join(words))
    seen: list[str] = []
    for anchor in anchors:
        norm = _anchor_norm(anchor)
        if norm and norm not in seen:
            seen.append(norm)
    return seen


def _is_identifier(compound: str) -> bool:
    """Keep 'atlas-core', 'HAL-1234', 'eu-west-1'; drop 'and/or', 'e.g', 'i.e'."""
    if any(ch.isdigit() for ch in compound):
        return True
    parts = [p.lower() for p in re.split(r"[-_.:/]+", compound) if p]
    return all(len(p) >= 2 for p in parts) and any(p not in _STOP_WORDS for p in parts)


def _anchor_norm(text: str) -> str:
    """Normalise identifiers so 'Atlas-Core', 'atlas core', 'atlas_core' compare equal."""
    return " ".join(_TOKEN_RE.findall(text.lower().replace("’", "'").replace("_", " ")))


def _contains_anchor(content_norm: str, anchor: str) -> bool:
    """Whole-token match of a normalised anchor ("atlas core" never matches "atlas cored")."""
    return re.search(rf"(?<!\w){re.escape(anchor)}(?!\w)", content_norm) is not None


def _stem(token: str) -> str:
    """Tiny suffix stemmer applied to a fixpoint, then final-e removal.

    owns/owned/owner → own, filter/filtered → filt, rate/rating → rat,
    database/databases → databas, boxes → box, policies → policy.
    Deliberately conservative (no prefix matching): "titan" never matches
    "titanium". Query and content are stemmed identically, so an occasional
    over-merge only widens a match.
    """
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
    terms: list[str] = []
    for token in _TOKEN_RE.findall(text.lower()):
        if token.endswith("'s"):
            token = token[:-2]
        if token in _STOP_WORDS or token in _TEMPORAL_WORDS or (len(token) < 2 and not token.isdigit()):
            continue
        stem = _stem(token)
        if stem not in terms:
            terms.append(stem)
    return terms


def _content_stems(content: str) -> set[str]:
    """Every stemmed token of a memory (stopwords kept: they cannot match query terms)."""
    return {_stem(token) for token in _TOKEN_RE.findall(content.lower())}


def _coverage(query_terms: list[str], content: str) -> float:
    """Fraction of the query's content terms that occur (stemmed) in the memory."""
    if not query_terms:
        return 0.0
    stems = _content_stems(content)
    return sum(1 for term in query_terms if term in stems) / len(query_terms)


def _relevance(query_terms: list[str], content: str, cosine: float) -> float:
    """Calibrated query evidence in [0, 1]: max of vector similarity and term coverage.

    Either signal alone can be decisive: a paraphrase scores on cosine, an
    exact identifier on coverage. Taking the max (not a blend) keeps the
    value comparable to ``RetrievalPolicySpec.min_confidence`` regardless of
    which channel found the memory.
    """
    return max(0.0, min(1.0, max(cosine, _coverage(query_terms, content))))
