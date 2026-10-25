"""Term evidence: the stemmer, query terms, and how much of a query a memory covers.

``_content_terms`` turns a query into salient stems, ``_content_stems`` a
memory into the stems it contains, and ``_coverage`` / ``_relevance`` give
the calibrated lexical evidence (0-1) that gap detection, reinforcement and
score fusion share. Split from ``query_understanding`` (which re-exports
every name) to keep both modules small.

Two rules keep the evidence honest:

- a dotted number with a long part ("0.30000000000000004") is one term, so
  a float artefact never matches a version such as "0.3.0" piece by piece;
- accents are folded on both sides ("Oyinbo" covers "Ọ̀yìnbó");
- a term matched only through an "-ing" form ("live" vs "living room")
  earns ``INFLECTION_CREDIT``, not a full match: the gerund is often a
  different word, and a full match showed unrelated memories at 1.000.
"""
from __future__ import annotations

from functools import lru_cache

from mark.intelligence._lazy_re import lazy
from mark.intelligence.classifier import _STOP_WORDS
from mark.intelligence.query_anchors import _TOKEN_RE

_TEMPORAL_WORDS = {"now", "current", "currently", "latest", "newest", "recent", "recently", "today",
                   "nowadays", "original", "originally", "initial", "initially", "first", "earliest",
                   "oldest", "last", "previous", "previously", "formerly", "still"}

# Credit for a query word found only as an "-ing" form of it (or vice versa).
INFLECTION_CREDIT = 0.8


class _Term(str):
    """A query stem that remembers the word it came from (for exact-form credit)."""
    __slots__ = ("surface",)

_CAMEL_WORD_RE = lazy(r"\b[A-Za-z]*[a-z][A-Z][A-Za-z]*\b")
# Dotted numbers; one with a long part ("0.30000000000000004") is a single
# term, never the version-like pieces "0" and "3..." that match "0.3.0".
_DOTTED_NUMBER_RE = lazy(r"(?<![\w.])\d+(?:\.\d+)+(?![\w])")
_LONG_PART = 5
_CAMEL_PART_RE = lazy(r"[A-Z]?[a-z]{2,}")
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
    original = text
    text = fold_accents(text)  # "Ọ̀yìnbó" and "Oyinbo" are one term
    longs = _long_numbers(text)
    for number in longs:
        terms.setdefault(number, None)
        text = text.replace(number, " ")
    for token in _TOKEN_RE.findall(text.lower()):
        if token.endswith("'s"):
            token = token[:-2]
        if token in _STOP_WORDS or token in _TEMPORAL_WORDS or (len(token) < 2 and not token.isdigit()):
            continue
        stem = _stem(token)
        if stem not in terms:
            term = _Term(stem)
            term.surface = token
            terms[term] = None
    if not original.isascii():
        # Emoji and other symbols are searchable terms ("🦉"), matched literally.
        from mark.embeddings.segments import is_symbol

        for char in original:
            if is_symbol(char):
                terms.setdefault(char, None)
    return list(terms)


@lru_cache(maxsize=1024)
def _content_stems(content: str) -> frozenset[str]:
    """Every stemmed token of a memory (stopwords kept: they cannot match query terms).

    Cached by content: one retrieval scores the same candidates in several
    stages (fusion, hops, ordering, facts).
    """
    return frozenset(_content_forms(content))


@lru_cache(maxsize=1024)
def _content_forms(content: str) -> dict[str, frozenset[str]]:
    """Stem -> the surface words of *content* that have it (see ``_content_stems``)."""
    tokens = set(_TOKEN_RE.findall(content.lower()))
    if not content.isascii():
        tokens |= set(_TOKEN_RE.findall(fold_accents(content).lower()))
    # snake_case identifiers also count as their words: "code_length" answers "code length".
    tokens |= {part for token in tokens if "_" in token for part in token.split("_") if len(part) > 1}
    # So do CamelCase ones: "TypeError" answers "what error ...".
    tokens |= {part.lower() for part in _camel_parts(content)}
    forms: dict[str, set[str]] = {}
    for token in tokens:
        forms.setdefault(_stem(token), set()).add(token)
    for number in _long_numbers(content):  # whole, as _content_terms keeps them
        forms.setdefault(number, set()).add(number)
    return {stem: frozenset(words) for stem, words in forms.items()}


def fold_accents(text: str) -> str:
    """*text* with accents removed from Latin, Greek and Cyrillic letters ("Ọ̀yìnbó" -> "Oyinbo").

    Stacked marks (a letter with a dot below plus a tone mark) are stripped
    too. CJK and Hangul are left alone: decomposing a Hangul syllable would
    turn it into jamo that no longer match the literal text.
    """
    if text.isascii():
        return text
    import unicodedata

    out = []
    for char in text:
        code = ord(char)
        if 0x300 <= code <= 0x36F:
            continue  # a combining mark on its own
        if 0x80 <= code < 0x2000:
            out.append("".join(c for c in unicodedata.normalize("NFKD", char) if not unicodedata.combining(c)))
        else:
            out.append(char)
    return "".join(out)


def _long_numbers(text: str) -> list[str]:
    """Dotted numbers with a part of 5+ digits (float artefacts, long decimals), in text order."""
    if "." not in text:
        return []
    return [match for match in _DOTTED_NUMBER_RE.findall(text)
            if any(len(part) >= _LONG_PART for part in match.split("."))]


def _camel_parts(content: str) -> list[str]:
    """Words inside CamelCase identifiers ("OperationalError" -> Operational, Error)."""
    if content.islower():
        return []
    return [part for word in _CAMEL_WORD_RE.findall(content) for part in _CAMEL_PART_RE.findall(word)]


def _coverage(query_terms: list[str], content: str) -> float:
    """Fraction of the query's content terms that occur (stemmed) in the memory.

    Terms an English stemmer cannot split are matched literally: a symbol
    ("🦉") as a substring, and a CJK run ("我的中文老师是谁") by the share of its
    character bigrams found in the memory, since CJK text has no spaces and
    the run is rarely written out whole.
    """
    if not query_terms:
        return 0.0
    forms = _content_forms(content)
    lowered: str | None = None
    found = 0.0
    for term in query_terms:
        if term in forms:
            found += _form_credit(getattr(term, "surface", None), forms[term])
        elif not term.isascii():
            lowered = content.lower() if lowered is None else lowered
            found += _literal_share(term, lowered)
    return found / len(query_terms)


def _form_credit(surface: str | None, words: frozenset[str]) -> float:
    """1 for the query's word (or its plural / possessive); less when only an "-ing" form matches."""
    if surface is None or surface in words:
        return 1.0
    gerund = surface.endswith("ing")
    if any(word.endswith("ing") == gerund for word in words):
        return 1.0
    return INFLECTION_CREDIT


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
