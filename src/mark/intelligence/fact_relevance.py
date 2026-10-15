"""Pick the facts about a subject that a query actually asks for.

``RetrievalResult.facts`` used to fall back to *every* fact about a subject
whenever no attribute name matched the query, so "Which region is atlas-core
in now?" returned its port and owner. Selection per subject, in order:

1. attributes named in the query (stemmed: "ports" -> port, "owns" -> owner)
   are returned, and only those;
2. otherwise, a who/whom question with a verb ("Who owns X?", "Who is
   assigned to X?") asks for the agent noun of that verb (own -> owner,
   assign -> assignee, manage -> manager, lead -> lead): only facts with that
   attribute are kept, and none when the subject has no such fact;
3. otherwise, if the question has a head noun ("Which *region* ...", "What
   *port* does ...", "What is the *region* of ...") the subject's facts are
   kept only when their value mentions it ("Which team ..." -> lead =
   "platform team"); a head noun matching nothing means the facts hold no
   answer, so none are returned (the memories still answer it);
4. otherwise (open questions: "Tell me about atlas-core", "What is
   atlas-core?", "Who is Mei Lin?") the subject's facts are kept, unless
   the subject was only inferred (``allow_open=False``: the user subject of
   an "I/my" question, or a subject found in the matched memories).

At most ``cap`` facts per subject are returned (``DEFAULT_FACT_CAP``), the
ones sharing most words with the query first, then the most recent; asking
for everything ("tell me about", "everything", "overview") lifts the cap.

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

# "Who [is|does] [now] <verb> ...": the verb names the role asked for.
_WHO_RE = re.compile(rf"\bwhom?\s+(?:{_AUX}\s+)?(?:(?:now|currently|still)\s+)?(\w+)", re.I)


def who_verb(query: str, subject: str = "") -> str | None:
    """Stem of the verb a who/whom question asks about ("Who owns X" -> "own"), or None.

    "Who is Mei Lin?" has no verb (the next word is the subject), and "Who is
    the owner ..." names the attribute itself, so both return None.
    """
    match = _WHO_RE.search(query)
    if match is None:
        return None
    word = match.group(1).lower()
    if word in _STOP_WORDS or word in _TEMPORAL_WORDS or word.isdigit():
        return None
    stem = _stem(word)
    return None if stem in _content_stems(subject) else stem


def _agent_stem(stem: str) -> str:
    """Fold an agent-noun stem onto its verb stem: "assigne" (assignee) -> "assign".

    The stemmer already maps owner/own, manager/manage and lead/leads together;
    only the "-ee" agent nouns keep an extra "e".
    """
    return stem[:-1] if len(stem) > 4 and stem.endswith("e") else stem


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


DEFAULT_FACT_CAP = 3
_EVERYTHING_RE = re.compile(
    r"\b(?:tell\s+me\s+(?:about|everything)|everything|all\s+(?:the\s+)?(?:facts|details|info\w*)"
    r"|overview|summar\w*|profile|what\s+do\s+(?:you|we|i)\s+know)\b",
    re.I,
)


def wants_everything(query: str) -> bool:
    """True when the query asks for all there is about a subject (no fact cap)."""
    return _EVERYTHING_RE.search(query) is not None


# Everyday question words -> the attribute names extraction uses. Small and
# generic on purpose: "Where do I live?" must reach a ``location`` fact.
_SYNONYM_GROUPS = (
    "live lives living reside resides based home location city town moved",
    "work works working job employer company occupation role",
    "eat eats diet food vegetarian vegan",
    "born birthday birth age old",
    "contact email phone number call reach",
    "like likes love prefer prefers favourite favorite hate dislike preference",
)


def _expand(terms: set[str]) -> set[str]:
    """Add the synonym group of every query term (normalised like the terms)."""
    out = set(terms)
    for group in _SYNONYM_GROUPS:
        words = set(_content_terms(group))
        if words & terms:
            out |= words
    return out


def select_facts(
    query: str,
    subject: str,
    rows: Sequence[Any],
    *,
    allow_open: bool = True,
    cap: int | None = DEFAULT_FACT_CAP,
) -> list[Any]:
    """Filter one subject's current fact rows (``attribute``/``value`` keys) for *query*."""
    terms = set(_content_terms(query))
    named = [row for row in rows if set(_content_terms(_attribute_text(row))) & _expand(terms)]
    if named:
        selected = named
    else:
        verb = who_verb(query, subject)
        head = question_head(query, subject) if verb is None else None
        if verb is not None:
            selected = [row for row in rows
                        if verb in {_agent_stem(stem) for stem in _content_stems(_attribute_text(row))}]
        elif head is not None:
            selected = [row for row in rows if head in _content_stems(str(row["value"]))]
        else:
            selected = list(rows) if allow_open else []
    if cap is None or len(selected) <= cap or wants_everything(query):
        return selected
    return _ranked(selected, terms)[:cap]


def attribute_named(query: str, rows: Sequence[Any]) -> list[Any]:
    """Rows whose attribute is fully named by the query ("test command" -> test_command)."""
    terms = set(_content_terms(query))
    return [row for row in rows if (attr := set(_content_terms(_attribute_text(row)))) and attr <= terms]


def _attribute_text(row: Any) -> str:
    """An attribute key as words: "lives_in" -> "lives in", "test-command" -> "test command"."""
    return re.sub(r"[_\-.]+", " ", str(row["attribute"]))


def _ranked(rows: Sequence[Any], terms: set[str]) -> list[Any]:
    """Most query words in attribute + value first, then the most recently valid."""
    def key(row: Any) -> tuple[int, str]:
        stems = _content_stems(f"{_attribute_text(row)} {row['value']}")
        return len(stems & terms), str(row.get("valid_from") or "") if hasattr(row, "get") else ""
    return sorted(rows, key=key, reverse=True)
