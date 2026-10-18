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

"Original/before" questions get the band oldest first instead. The band
moves as a block to the slot of its best member (others keep their order)
and ranking scores stay non-increasing.
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
    ("manager managers boss supervisor report reports", r"\breport(?:s|ing)?\s+to\b"),
    ("use uses using", r"\b(?:switched|moved|migrated|changed)\s+(?:over\s+)?to\b"),
    # "What time is standup?": a clock time states the time.
    ("time times", r"\b\d{1,2}[:.]\d{2}\b|\b\d{1,2}\s*[ap]\.?m\b"),
)
# Words that find unnamed updates of a group's attribute in the keyword index.
_CHANGE_LOOKUPS = (("moved", "relocated", "based"), ("number",), ("job", "jobs"), ("moved", "rescheduled"),
                   ("report", "reports"), ("switched", "migrated"), ("moved", "rescheduled"))
# Words that say nothing about which attribute is asked ("What's my sister's name?").
_WEAK = {"name", "call", "called", "much", "many", "old", "new", "former", "previous"}
# First person singular or plural ("we use Jira"; "us" is left out: the US).
_FIRST_PERSON = r"\b(?:i|i'm|im|my|me|mine|myself|we|we're|our|ours)\b"
_CHANGE_EXTRA = r"\b(?:new\s+one|now|these\s+days|nowadays)\b"
_MIN_TOPIC_SHARE = 0.5
_LOW_TRUST = frozenset({"mark:assistant", "mark:question"})
# Marker "subject" for questions about a thing named by plain words (see _subject_and_terms).
TOPIC = "\x00topic"
# Words too generic to make a topic on their own ("What time is it?").
_GENERIC = {"time", "use", "thing"}


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


def _subject_and_terms(query: str) -> tuple[list[str], list[str]] | None:
    """(names asked about, attribute terms), or None when the question has no subject.

    No names means the user ("I/my") is the subject."""
    first_person = _patterns()[0]
    anchors = gap_anchors(query)
    names = [n for n in anchors if not n.isdigit()]
    name_words = {word for name in names for word in name.split()}
    terms = [t for t in _content_terms(query) if t not in name_words and t not in _WEAK
             and not t.isdigit() and t.isascii()]
    if not names and not first_person.search(query):
        # No name and no "I/we": the topic words are the subject ("What time is
        # standup?", "What is the wifi SSID?"); every one must be stated.
        # A question naming a number or id ("ticket 4521") asks for that one record.
        return ([TOPIC], terms) if not anchors and not any(c.isdigit() for c in query) and set(terms) - _GENERIC else None
    return (names, terms) if terms else None


def _member(content: str, names: list[str], terms: list[str]) -> bool:
    """Whether a memory is about the asked subject and states every asked attribute term."""
    if names == [TOPIC]:
        pass  # topic questions: the attribute terms below are the whole test
    elif names:
        if not all(name_mentioned(content, n) for n in names):
            return False
    elif _patterns()[0].search(content) is None:
        return False
    stems = _content_stems(content)
    return all(_satisfied(t, content, stems) for t in terms)


def update_lookups(query: str) -> tuple[list[str], object] | None:
    """Keyword lookups that find unnamed updates of what *query* asks, plus a membership test.

    "Where do I live?" -> (["moved", "relocated"], test): "I moved to Porto
    last month" shares no word with the question or with "I live in Lagos".
    """
    parsed = _subject_and_terms(query)
    if parsed is None:
        return None
    names, terms = parsed
    words: list[str] = []
    for (stems, _), lookups in zip(_GROUPS, _CHANGE_LOOKUPS):
        if set(_content_terms(stems)) & set(terms):
            words += [w for w in lookups if w not in words]
    if not words:
        return None
    return words, lambda content: _member(content, names, terms)


class StateBand:
    """Result of the band stage: new order plus per-id flags (see module docstring)."""

    __slots__ = ("order", "outdated", "conflicts")

    def __init__(self, order: list[int], outdated: set[str], conflicts: dict[str, list[str]]) -> None:
        self.order, self.outdated, self.conflicts = order, outdated, conflicts


def state_band(query: str, intent: TemporalIntent, fragments: list[MemoryFragment]) -> StateBand | None:
    """Reorder the asked subject+attribute band when it holds an update or a conflict."""
    if len(fragments) < 2:
        return None
    parsed = _subject_and_terms(query)
    if parsed is None:
        return None
    names, terms = parsed
    name_words = {word for name in names if name != TOPIC for word in name.split()}
    change_re = _patterns()[1]
    # The model's own replies and stored questions never outrank what the user stated.
    band = [i for i, f in enumerate(fragments)
            if _member(f.content, names, terms) and not _LOW_TRUST.intersection(f.tags or ())]
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
    # The band moves together to its best slot: swapping members across the
    # list would make every unrelated result between them show the evidence
    # of the member moved down (displayed scores are non-increasing).
    first = min(band)
    rest = [i for i in range(len(fragments)) if i not in set(band)]
    at = sum(1 for i in rest if i < first)
    order = rest[:at] + ordered + rest[at:]
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
