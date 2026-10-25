"""Query anchors: the distinctive identifiers and names a question is about.

An anchor that appears nowhere in memory is a real gap ("not in memory"),
not just a weak match, and anchors steer fusion weights, temporal bands and
fact subjects. Two levels:

- ``query_anchors`` — identifiers and multi-word names: compound ids
  (atlas-core, HAL-1234, eu-west-1, TodoStore._migrate), snake_case
  (load_config, SESSION_TIMEOUT_MIN), CamelCase (TaskQueue, debounceSearch),
  tokens with digits, and runs of title-case words ("Mei Lin");
- ``gap_anchors`` — those plus single capitalised words inside a sentence
  ("How is Redis configured?", "Who does Priya report to?"). They only feed
  gap detection: a lone capitalised word is a likely proper noun, but too
  weak a signal to steer ranking.
"""
from __future__ import annotations

import re

from mark.intelligence._lazy_re import lazy  # compiled on first use (cold start)
from functools import lru_cache

from mark.intelligence.classifier import _STOP_WORDS
from mark.intelligence.date_parsing import find_dates
from mark.embeddings.segments import CJK_RUN

_TOKEN_RE = lazy(r"\w+", re.UNICODE)

# Distinctive anchors: compound identifiers (atlas-core, HAL-1234, eu-west-1,
# v2.3.1), tokens with digits, and multi-word capitalised names.
_COMPOUND_RE = lazy(r"\b\w+(?:[-_.:/]\w+)+\b")
_DIGIT_TOKEN_RE = lazy(r"\b(?=\w*\d)\w{3,}\b")
# Code identifiers: snake_case (a "_" between word characters, so "_private"
# alone is not one) and CamelCase / camelCase (a lower-to-upper transition).
_SNAKE_RE = lazy(r"\b_*[A-Za-z][A-Za-z0-9]*(?:_+[A-Za-z0-9]+)+\b")
_CAMEL_RE = lazy(r"\b[A-Za-z]*[a-z][A-Z][A-Za-z0-9]*\b")
# Bare numbers, decimals, dates and times ("8080", "2025", "1.5", "2026-03-01",
# "10:30") are values the query mentions, not subjects: a port or a year missing
# from memory is not a gap. Dotted quads (IP addresses) have four groups and stay.
_BARE_NUMBER_RE = lazy(r"\d+(?:[.,:/-]\d+){0,2}")
_NUMBER_WITH_UNIT_RE = lazy(r"\d+(?:s|st|nd|rd|th|ms|us|ns|k|kb|mb|gb|tb|m|h|x|px|pc|pct|am|pm)", re.I)
# Title-case words only ("Mei Lin", "Liam O'Connor"); ALL-CAPS log labels such
# as "TICKET HAL" are not names.
_NAME_RE = lazy(r"\b([A-Z][a-z][\w'’]*(?:\s+[A-Z][a-z'’][\w'’]*)+)\b")
# A single title-case word (gap anchors only), and what may precede it for it
# to count as "inside a sentence" rather than capitalised by position.
_TITLE_WORD_RE = lazy(r"(?<![\w'’.-])([A-Z][a-z][a-z'’]+)\b(?![-_.:/]\w)")
_SENTENCE_START_RE = lazy(r"(?:^|[.!?:;\"“(\[]\s*|\n\s*)$")
# Leading words of a title-cased run that are not part of a name
# ("Has Alice Smith moved?", "Tell Me About billing", "The Platform Team").
_NAME_LEAD_WORDS = {
    "who", "what", "which", "when", "where", "why", "how", "is", "are", "was", "were", "does", "do",
    "did", "has", "have", "had", "can", "could", "should", "would", "will", "shall", "may", "might",
    "must", "tell", "show", "give", "list", "find", "check", "please", "the", "a", "an", "in", "on",
    "at", "for", "from", "to", "of", "my", "our", "your", "their", "his", "her", "its", "me", "us",
    "about", "and", "or", "but", "so", "then", "if", "per", "also", "dr", "mr", "mrs", "ms",
}
# Capitalised words that are not proper nouns about memory content: calendar
# words (dates are parsed, not looked up), languages and nationalities (a
# memory written *in* Chinese rarely contains the word "Chinese").
_NOT_PROPER = {
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    "january", "february", "march", "april", "may", "june", "july", "august", "september",
    "october", "november", "december", "english", "spanish", "french", "german", "portuguese",
    "italian", "chinese", "japanese", "korean", "arabic", "hindi", "russian", "dutch",
    "yoruba", "igbo", "hausa", "swahili", "european", "american", "british", "african", "asian",
    "christmas", "easter", "ok", "okay", "yes", "no", "hi", "hello", "thanks", "please",
}
_POSSESSIVE_RE = lazy(r"(?<=\w)['’]s\b")
# Entities are read from the head of a memory: enough for statements, bounded for documents.
_ENTITY_SCAN_CHARS = 2000


def query_anchors(query: str) -> list[str]:
    """Distinctive identifiers and names in a query, normalised to lowercase.

    Anchors are what the question is *about* (atlas-core, HAL-1234, Mei Lin,
    load_config, TaskQueue). Heuristics, in order: compound identifiers,
    snake_case and CamelCase identifiers, tokens containing digits (minus
    quantities like "10ms" and bare numbers such as ports, years and dates,
    which are values rather than subjects), and runs of title-case words
    with leading/trailing question words stripped ("Has Alice Smith" ->
    "alice smith"). Possessives are dropped first so "Mei Lin's" matches
    "Mei Lin", and date expressions ("third-quarter", "3rd of May") are
    dates, not names.
    """
    return list(_query_anchors(query))


@lru_cache(maxsize=4096)
def _query_anchors(query: str) -> tuple[str, ...]:
    # Cached: one retrieval asks for the same query's anchors in several stages.
    query = _POSSESSIVE_RE.sub("", query)
    for span in reversed(find_dates(query)):
        query = query[: span.pos[0]] + " " + query[span.pos[1]:]
    anchors: list[str] = []
    for match in _COMPOUND_RE.findall(query):
        if _is_identifier(match) and not _BARE_NUMBER_RE.fullmatch(match) and not _is_descriptive(match):
            anchors.append(match)
    for pattern in (_SNAKE_RE, _CAMEL_RE):
        for match in pattern.findall(query):
            if not any(match in compound for compound in anchors):
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
    return tuple(_unique_norms(anchors))


def gap_anchors(query: str) -> list[str]:
    """``query_anchors`` plus single capitalised words inside a sentence (likely names).

    "How is Redis configured?" -> ["redis"]; "Redis is down" -> [] (the
    capital may be positional); weekdays, months and language names are
    never anchors.
    """
    anchors = query_anchors(query)
    stripped = _POSSESSIVE_RE.sub("", query)
    for match in _TITLE_WORD_RE.finditer(stripped):
        word = match.group(1)
        lower = word.lower()
        if lower in _NOT_PROPER or lower in _NAME_LEAD_WORDS or lower in _STOP_WORDS:
            continue
        if _SENTENCE_START_RE.search(stripped[: match.start()]):
            continue
        if any(lower in anchor.split() for anchor in anchors):
            continue
        anchors.append(word)
    return _unique_norms(anchors)


def content_entities(content: str) -> list[str]:
    """Normalised names and identifiers in a memory, in order of appearance.

    Like ``gap_anchors`` but for stored text: a capitalised word counts even
    at the start of a sentence ("Tomasz owns the ledger-sync service."),
    since memories are statements rather than questions.
    """
    text = _POSSESSIVE_RE.sub("", content[:_ENTITY_SCAN_CHARS])
    found = list(query_anchors(text))
    for match in _TITLE_WORD_RE.finditer(text):
        word = match.group(1).lower()
        if word in _STOP_WORDS or word in _NAME_LEAD_WORDS or word in _NOT_PROPER:
            continue
        if not any(word in anchor.split() for anchor in found):
            found.append(word)
    return found


def _unique_norms(anchors: list[str]) -> list[str]:
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


# Last parts of hyphenated adjectives ("well-known", "dairy-free", "4-year-old").
_ADJECTIVE_TAILS = {"old", "known", "like", "free", "based", "sized", "size", "long", "wide", "term", "time",
                    "friendly", "proof", "made", "aged", "style", "looking", "level", "only", "way", "end"}


def _is_descriptive(compound: str) -> bool:
    """Hyphenated adjectives and quantities are descriptions, not identifiers.

    "4-year-old", "3-day", "10-minute" (a number then lowercase words) and
    lowercase compounds ending in a participle or a common adjective tail
    ("paper-lined", "well-known", "dairy-free") were read as ids and then
    reported "not in memory" when a user introduced them.
    """
    if not re.fullmatch(r"[0-9a-z]+(?:-[a-z]+)+", compound):
        return False
    parts = compound.split("-")
    if parts[0].isdigit():
        return True
    tail = parts[-1]
    return tail in _ADJECTIVE_TAILS or (len(tail) >= 4 and tail.endswith("ed"))


def _anchor_norm(text: str) -> str:
    """Normalise identifiers so 'Atlas-Core', 'atlas core', 'atlas_core' compare equal."""
    return " ".join(_TOKEN_RE.findall(text.lower().replace("’", "'").replace("_", " ")))


def _contains_anchor(content_norm: str, anchor: str) -> bool:
    """Whole-token match of a normalised anchor ("atlas core" never matches "atlas cored").

    CJK-aware: Chinese and Japanese are written without spaces (and Korean
    attaches particles to names), so an anchor edge that is a CJK character
    needs no boundary ("王小明" is in "王小明搬到了上海"), and a CJK character
    next to a Latin edge counts as one ("acme" is in "我在acme工作").
    """
    if not anchor or (anchor.isascii() and content_norm.isascii()):
        return re.search(rf"(?<!\w){re.escape(anchor)}(?!\w)", content_norm) is not None
    # A scan instead of a regex: a pattern with the CJK class costs ~3 ms to
    # compile, and every query and timeline brings new anchors.
    free_left, free_right = _is_cjk(anchor[0]), _is_cjk(anchor[-1])
    start = content_norm.find(anchor)
    while start != -1:
        end = start + len(anchor)
        if (free_left or start == 0 or not _extends_word(content_norm[start - 1])) and (
                free_right or end == len(content_norm) or not _extends_word(content_norm[end])):
            return True
        start = content_norm.find(anchor, start + 1)
    return False


def _is_cjk(char: str) -> bool:
    return CJK_RUN.match(char) is not None


def _extends_word(char: str) -> bool:
    """A word character (as ``\\w``) that is not CJK: it would continue a token across the edge."""
    return (char.isalnum() or char == "_") and not _is_cjk(char)


def anchor_present(content_norm: str, anchor: str) -> bool:
    """Anchor match for gap detection: the exact phrase, or every part of an identifier.

    "TodoStore._migrate" is present in "TodoStore.__init__ calls
    self._migrate()": both parts occur as whole tokens in one memory, which
    is evidence the question is covered even though the dotted path is not
    written out.
    """
    if _contains_anchor(content_norm, anchor):
        return True
    parts = anchor.split()
    return len(parts) > 1 and all(_contains_anchor(content_norm, part) for part in parts)


# Capitalised words that may precede a name without being part of it ("Yesterday Joao called").
_NOT_NAME_PREFIX = {"yesterday", "today", "tomorrow", "also", "then", "now", "dear", "hi", "hey", "friend"}


def name_mentioned(content: str, name: str) -> bool:
    """Whole-name match: *name* (normalised) occurs in *content* as that name, not inside a longer one.

    "Joao" is not mentioned by "nurse at Sao Joao hospital" (a capitalised
    word right before it makes it part of another name), while "Sao Joao" is;
    "Joao Silva" still mentions "Joao" (given names come first). Matching is
    case-insensitive; a lowercase neighbour never extends a name. CJK names
    fall back to :func:`_contains_anchor`.
    """
    tokens = name.split()
    if not tokens:
        return False
    if not name.isascii():
        return _contains_anchor(_anchor_norm(content), name)
    pattern = r"(?<![\w-])" + r"[\s\-_.'’]+".join(re.escape(t) for t in tokens) + r"(?![\w-])"
    for match in re.finditer(pattern, content, re.I):
        before = re.search(r"([^\W\d_][\w'’]*)[\s]+$", content[max(0, match.start() - 40):match.start()])
        if before is None or not before.group(1)[0].isupper():
            return True
        word = before.group(1).lower()
        if word in _STOP_WORDS or word in _NAME_LEAD_WORDS or word in _NOT_PROPER or word in _NOT_NAME_PREFIX:
            return True
    return False


def surface_form(query: str, anchor: str) -> str:
    """The anchor as the user wrote it ("quuxinator 9000" -> "Quuxinator-9000"), else *anchor*.

    Gap reports quote what was not found; the normalised form ("hal 1234")
    reads like a different thing than the "HAL-1234" the user asked about.
    """
    text = _POSSESSIVE_RE.sub("", query)
    spans = [m.span() for m in _TOKEN_RE.finditer(text)]
    size = len(anchor.split())
    for i in range(len(spans) - size + 1):
        start, end = spans[i][0], spans[i + size - 1][1]
        if _anchor_norm(text[start:end]) == anchor:
            return text[start:end]
    return anchor
