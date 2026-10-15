"""Clause splitting with subject carry-over for :class:`PatternFactExtractor`.

A first-mention line often lists several facts about one subject:

    service atlas-core is owned by Dana, listens on port 6430, runs in eu-west-1 and uses Redis

Rule packs match one clause at a time and need the subject at the clause
start, so a clause that *begins with a verb* ("listens on ...", "and uses
...", "working from ...") borrows the subject of the clause before it. A
clause that names its own subject resets the carried one, and nothing is
carried across sentences.
"""
from __future__ import annotations

import re

from mark.intelligence.fact_extraction.base import FactCandidate
from mark.intelligence.fact_extraction.rules import _TOKEN

# Verbs that can open a subject-less clause continuing a description. Kept to
# the verb families the built-in packs understand, so a carried subject only
# ever meets a pattern written for it.
_CONTINUATION_VERBS = (
    r"(?:now\s+|also\s+|currently\s+)?(?:is|are|was|were|has|have|listens|runs|serves|uses|relies|reports"
    r"|works|working|lives|living|based|located|hosted|deployed|owned|maintained|managed|assigned"
    r"|reassigned|moved|switched|migrated|transferred|relocated|joined|owns|maintains|leads"
    # Base forms continue a first-person clause: "I live in Lagos and work at Seefar".
    r"|work|live|own|lead|use|prefer|like|love|hate|enjoy|speak|drive)\b"
)
# Clause boundaries: ";" and ", " (an "and" after the comma is dropped), plus a
# bare "and" when a continuation verb follows ("... in Berlin and reports to Ana").
_CLAUSE_SPLIT = re.compile(rf";\s*|,\s+(?:and\s+)?|\s+and\s+(?={_CONTINUATION_VERBS})", re.IGNORECASE)
_CONTINUATION = re.compile(rf"^{_CONTINUATION_VERBS}", re.IGNORECASE)
# Grammatical subject of a clause: 1-4 tokens before an auxiliary or a
# lower-case "-ed"/"-s" verb ("TICKET INC-4 opened", "Mei Lin joined"). The
# lower-case requirement keeps capitalised surnames ("James Jones") whole.
_LEAD_SUBJECT = re.compile(
    rf"^(?P<subject>{_TOKEN}(?:\s+{_TOKEN}){{0,3}}?)\s+"
    r"(?=(?:is|are|was|were|has|have|had|now|does|did)\b|(?-i:[a-z][a-z]*(?:ed|s))(?=[\s:(]|$))",
    re.IGNORECASE)
_NOT_A_SUBJECT = frozenset(
    "it its he she they them we i you this that these those which who there here someone everyone the a an".split())


def split_clauses(sentence: str) -> list[str]:
    """Split one sentence into clauses (see module docstring)."""
    return [part for part in _CLAUSE_SPLIT.split(sentence) if part and part.strip()]


def is_continuation(clause: str) -> bool:
    """True when *clause* starts with a verb and so has no subject of its own."""
    return bool(_CONTINUATION.match(clause.strip()))


def carried_subject(clause: str, candidates: list[FactCandidate]) -> str | None:
    """The subject later verb-first clauses of this sentence should borrow.

    Prefer the subject of a fact found in *clause* when it is written before
    its value (so "Dana owns X" carries "Dana", not "X"); otherwise use the
    grammatical subject, which also covers clauses that yielded no fact
    ("TICKET INC-4 opened ..., assigned to Lee").
    """
    text = clause.lower()
    for candidate in candidates:
        at_subject, at_value = text.find(candidate.subject.lower()), text.find(candidate.value.lower())
        if 0 <= at_subject and (at_value < 0 or at_subject < at_value):
            return candidate.subject
    lead = _LEAD_SUBJECT.match(clause.strip())
    if lead is None:
        return None
    subject = lead.group("subject")
    return None if subject.lower() in _NOT_A_SUBJECT else subject
