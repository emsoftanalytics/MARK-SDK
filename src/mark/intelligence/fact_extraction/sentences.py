"""Sentence preparation for :class:`PatternFactExtractor`.

Splits text into sentences (newlines and ``. ! ?`` + space, but not after
"Dr." or "e.g."), reads a leading date stamp and upper-case log labels
("[2026-03-30] CHANGE: ...", "HAL-1: ..."), and holds the guards that make a
sentence or clause unfit for extraction: questions, conditionals, intents
and negations. Pure functions over strings; no extraction logic here.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Iterator

from mark.intelligence.fact_extraction.labels import is_id_label, label_kind
from mark.security.lazy_regex import lazy_compile

# Abbreviations whose full stop does not end a sentence ("Dr. Bello").
_ABBREVIATIONS = ("Dr", "Mr", "Mrs", "Ms", "Mx", "Prof", "Sr", "Jr", "St", "Mt", "Capt", "Gen", "Col", "Lt",
                  "Rev", "Hon", "vs", "approx", "e.g", "i.e", "cf", "No", "Fig", "Inc", "Ltd", "Co", "Corp")
_SENTENCE_SPLIT = lazy_compile(
    r"\n+|" + "".join(rf"(?<!\b{re.escape(a)}\.)" for a in _ABBREVIATIONS) + r"(?<=[.!?])\s+")
_STAMP = lazy_compile(r"^\s*\[?(\d{4}-\d{2}-\d{2})(?:[T ](\d{2}:\d{2}(?::\d{2})?)Z?)?\]?\s*[:\-–—]?\s*")
# Upper-case log labels: "CHANGE:", "HR:", "DECISION ADR-7:" (all-caps tokens only).
_LABEL = lazy_compile(r"^(?P<label>[A-Z][A-Z0-9_/&-]*(?:\s+[A-Z][A-Z0-9_/&-]*){0,2})\s*:\s+")
LEADING_FILLER = lazy_compile(
    r"^(?:today|yesterday|tonight|this\s+(?:morning|afternoon|evening|week)|earlier\s+today|now|then"
    r"|finally|also|meanwhile|later|afterwards|after\s+that|subsequently|eventually|update|fyi|note|btw"
    r"|by\s+the\s+way|oh|ok|okay|so|well)\b[\s,:-]*",
    re.IGNORECASE)
_QUESTION_START = lazy_compile(
    r"^(?:did|does|do|is|are|was|were|will|would|should|can|could|has|have|had|shall|may|might"
    r"|when|what|which|who|whom|whose|why|how|where)\b", re.IGNORECASE)
CONDITIONAL = lazy_compile(
    r"\b(?:if|unless|whether|suppose|supposing|assuming|in\s+case|hypothetically|imagine"
    r"|would|could|might|should|maybe|perhaps|probably)\b", re.IGNORECASE)
NEGATION = lazy_compile(
    r"\b(?:not|never|no\s+longer|no\s+more|neither|nor|cannot|failed\s+to|refused\s+to|used\s+to)\b|n['’]t\b",
    re.IGNORECASE)
INTENT = lazy_compile(
    r"\b(?:will|shall|going\s+to|plan(?:s|ned)?\s+to|planning|intends?\s+to|expected\s+to|expects?\s+to"
    r"|scheduled\s+to|about\s+to|propose[sd]?|proposal|wants?\s+to|would\s+like|to\s+be|soon|someday"
    r"|one\s+day|hope\s+to|hoping\s+to|thinking\s+(?:of|about)|consider(?:s|ing)?)\b|['’]d\s+(?:like|love)\b",
    re.IGNORECASE)
INLINE_DATE = lazy_compile(r"\b(?:on|effective|as\s+of|since)\s+(\d{4}-\d{2}-\d{2})\b", re.IGNORECASE)


@dataclass
class Sentence:
    """One sentence ready for clause matching, with what its prefix said."""
    text: str
    when: datetime | None
    kinds: list[str] = field(default_factory=list)
    #: A ticket-like label ("HAL-1:") naming the subject of "key: value" clauses.
    record_subject: str | None = None
    #: The body after a TODO/DONE/FIX label, for ticket status facts.
    labelled_body: str | None = None
    #: A ticket id written inside a kind label ("TODO HAL-12: ..." -> "HAL-12").
    label_id: str | None = None


def iter_sentences(text: str, occurred_at: datetime | None) -> Iterator[Sentence]:
    """Sentences of *text* with stamps and labels read off (see module docstring).

    The first stamp is the memory's own event time, which the caller
    resolves (explicit ``occurred_at`` wins). A later line's stamp dates that
    line and the ones after it (a multi-line log in one memory). A label
    applies to the rest of its line.
    """
    when, first = occurred_at, True
    for line in re.split(r"\n+", text or ""):
        kinds: list[str] = []
        record: str | None = None
        body: str | None = None
        label_id: str | None = None
        for index, raw in enumerate(_SENTENCE_SPLIT.split(line)):
            sentence = raw.strip()
            if not sentence:
                continue
            stamp = _STAMP.match(sentence)
            if stamp:
                if not first:
                    when = parse_stamp(stamp.group(1), stamp.group(2)) or when
                sentence = sentence[stamp.end():]
            first = False
            while (label := _LABEL.match(sentence)) is not None:
                name = label.group("label")
                kind = label_kind(name)
                if kind and kind not in kinds:
                    kinds.append(kind)
                    body = sentence[label.end():]
                    rest = name.split(None, 1)[1] if len(name.split()) > 1 else ""
                    label_id = rest if rest and is_id_label(rest) else label_id
                if is_id_label(name):
                    record = name
                sentence = sentence[label.end():]
            yield Sentence(sentence, when, list(kinds), record, body if index == 0 else None,
                           label_id if index == 0 else None)


def is_question(sentence: str) -> bool:
    """A question states nothing ("Did X move to Y?", "Is X on port 1")."""
    return sentence.rstrip().endswith("?") or bool(_QUESTION_START.match(sentence))


def parse_stamp(day: str, clock: str | None) -> datetime | None:
    """"2026-03-30" (+ optional "12:00") as an aware UTC datetime, or None."""
    try:
        parsed = datetime.fromisoformat(day + (f"T{clock}" if clock else ""))
    except ValueError:
        return None
    return parsed.replace(tzinfo=timezone.utc)
