"""Current state without a named old value: unnamed updates and conflicting statements.

``supersession`` handles updates that name what they replace ("moved from
Lisbon to Porto", "not Ana"). People often don't: "I moved to Porto last
month", "These days I now live in Braga", "Joao changed his number, the new
one is ...", "Biscuit's vet appointment moved to 21 October". And sometimes
two statements simply disagree ("Grace's birthday is 14 March" / "... is
21 March") with no change wording at all.

This stage is driven by the question. Its *band* is the retrieved memories
about the asked subject (the names the question uses, matched as whole
names, or the user for "I/my" questions) that also state the asked
attribute (every attribute word of the question, or an everyday synonym of
it: live ~ moved to ~ city, number ~ phone, job ~ work). When the band holds
two or more memories and

- its newest memory is worded as a change ("moved", "changed", "now", "new
  one") -> the band is ordered newest first and the older members are
  *outdated*;
- otherwise, members state different values (numbers, dates or names) ->
  newest first, and the disagreeing members are *conflicts* of each other.

"Original/before" questions get the band oldest first instead. Members keep
their slots in the result list (others never move) and scores stay
non-increasing.
"""
from __future__ import annotations

import re
from functools import cache

from mark.intelligence.query_anchors import gap_anchors, name_mentioned
from mark.intelligence.query_understanding import TemporalIntent, _content_stems, _content_terms
from mark.intelligence.supersession import is_update_statement
from mark.intelligence.temporal_order import _order_band
from mark.types import MemoryFragment

# Everyday attribute synonyms: (stems, phrase pattern). A query term from a
# group is satisfied by any stem of the group or by the phrase.
_GROUPS: tuple[tuple[str, str], ...] = (
    ("live lives living reside resides residence based relocated city town hometown",
     r"\b(?:mov(?:ed|ing)\s+(?:back\s+)?(?:to|from|into)|relocat\w*)\b"),
    ("phone phones mobile cell telephone number", r"\bnew\s+one\s+is\b"),
    ("job jobs work works working employer employed occupation role position", r"\bnow\s+an?\s+\w+"),
    ("appointment appointments meeting booking reservation", r"\b(?:moved|rescheduled|postponed)\s+to\b"),
)
# Words that say nothing about which attribute is asked ("What's my sister's name?").
_WEAK = {"name", "call", "called", "much", "many", "old", "new", "former", "previous"}
_FIRST_PERSON = r"\b(?:i|i'm|im|my|me|mine|myself)\b"
_CHANGE_EXTRA = r"\b(?:new\s+one|now|these\s+days|nowadays)\b"
_MIN_TOPIC_SHARE = 0.5


@cache
def _patterns() -> tuple[re.Pattern[str], re.Pattern[str], re.Pattern[str]]:
    # Compiled on first use (cold-import cost).
    return (re.compile(_FIRST_PERSON, re.I), re.compile(_CHANGE_EXTRA, re.I),
            re.compile(r"\b(?:\d[\w:.,/+-]*|[A-Z][a-z]+)\b"))


@cache
def _groups() -> tuple[tuple[frozenset[str], re.Pattern[str]], ...]:
    return tuple((frozenset(_content_terms(stems)), re.compile(phrase, re.I)) for stems, phrase in _GROUPS)


def _satisfied(term: str, content: str, stems: frozenset[str]) -> bool:
    if term in stems:
        return True
    return any(term in words and (words & stems or phrase.search(content)) for words, phrase in _groups())


def _values(content: str, skip: set[str]) -> set[str]:
    """Value-like tokens: numbers, dates, and capitalised words that are not the subject."""
    return {token.lower() for token in _patterns()[2].findall(content)} - skip


class StateBand:
    """Result of the band stage: new order plus per-id flags (see module docstring)."""

    __slots__ = ("order", "outdated", "conflicts")

    def __init__(self, order: list[int], outdated: set[str], conflicts: dict[str, list[str]]) -> None:
        self.order, self.outdated, self.conflicts = order, outdated, conflicts


def state_band(query: str, intent: TemporalIntent, fragments: list[MemoryFragment]) -> StateBand | None:
    """Reorder the asked subject+attribute band when it holds an update or a conflict."""
    if len(fragments) < 2:
        return None
    first_person, change_re, _ = _patterns()
    names = [n for n in gap_anchors(query) if not n.isdigit()]
    if not names and not first_person.search(query):
        return None
    name_words = {word for name in names for word in name.split()}
    terms = [t for t in _content_terms(query) if t not in name_words and t not in _WEAK
             and not t.isdigit() and t.isascii()]
    if not terms:
        return None
    band: list[int] = []
    for i, fragment in enumerate(fragments):
        content = fragment.content
        subject = (all(name_mentioned(content, n) for n in names) if names
                   else first_person.search(content) is not None)
        if subject:
            stems = _content_stems(content)
            if all(_satisfied(t, content, stems) for t in terms):
                band.append(i)
    if len(band) < 2:
        return None
    newest_first = _order_band(fragments, band, newest_first=True)
    newest = fragments[newest_first[0]]
    outdated: set[str] = set()
    conflicts: dict[str, list[str]] = {}
    if is_update_statement(newest.content) or change_re.search(newest.content):
        outdated = {fragments[i].id for i in newest_first[1:]}
    else:
        conflicts = _conflicts([fragments[i] for i in newest_first], name_words)
        if not conflicts:
            return None
    ordered = list(reversed(newest_first)) if intent == TemporalIntent.ORIGINAL else newest_first
    if intent == TemporalIntent.ORIGINAL:
        outdated = set()
    slots = sorted(band)
    order = list(range(len(fragments)))
    for slot, index in zip(slots, ordered):
        order[slot] = index
    return StateBand(order, outdated, conflicts)


def _conflicts(members: list[MemoryFragment], subject_words: set[str]) -> dict[str, list[str]]:
    """Pairs of near-identical statements (same topic) that give different values."""
    found: dict[str, list[str]] = {}
    for i, a in enumerate(members):
        for b in members[i + 1:]:
            ta, tb = set(_content_terms(a.content)), set(_content_terms(b.content))
            if len(ta & tb) < _MIN_TOPIC_SHARE * min(len(ta), len(tb)):
                continue
            va, vb = _values(a.content, subject_words), _values(b.content, subject_words)
            if va and vb and va != vb and not (va <= vb or vb <= va):
                found.setdefault(a.id, []).append(b.id)
                found.setdefault(b.id, []).append(a.id)
    return found


def apply_band(band: StateBand, fragments, scores, relevances):
    """Permute by the band order; scores keep their slots (non-increasing).

    Relevances move with their memories; the pipeline's display pass then
    gives each slot a non-increasing value (see ``retrieval_finish``).
    """
    return [fragments[i] for i in band.order], list(scores), [relevances[i] for i in band.order]
