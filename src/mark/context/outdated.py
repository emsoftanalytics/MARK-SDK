"""Which briefing lines are outdated: retired by a revision, or replaced by a later statement.

Small models read two lines about the same thing as two truths ("vet on 14
October" next to "vet moved to 21 October"; a10 small-model run, root cause
2), so a briefing must say which one is current. Two signals are used:

1. **Retired.** ``memory.supersede()`` / ``resolve()`` tag the old memory
   ``mark:superseded`` / ``mark:resolved`` and set ``metadata["lifecycle"]``.
2. **Replaced by a later statement.** A newer trusted line that

   * names the old value as old: "moved *from Porto*", "Ines, *not Ana*",
     "*no longer at* Sao Joao", "Postgres *instead of SQLite*", and the older
     line contains that value; or
   * is a change statement ("changed", "moved", "the new one", "now") that
     shares at least two content words with the older line ("Joao changed
     his number" vs "Joao's phone number is ...").

Only lines that are both in the candidate pool are compared, and only an
older line is ever marked. Only the first ``_WINDOW`` characters of a memory
are read (a change statement leads its memory; a briefing line shows little
more), each window is analysed once (cached), and only pairs that can match
are compared — 12 x 16 KB memories took 1-6 s per briefing (a10 re-test 4). Callers drop outdated lines for small models and
label them "(outdated)" otherwise (``mark.context.watch_render``).
"""
from __future__ import annotations

import re
from functools import lru_cache
from typing import Any, Callable, Iterable, NamedTuple

from mark.context.relevance import content_words

RETIRED_TAGS = frozenset({"mark:superseded", "mark:resolved"})
_RETIRED_LIFECYCLE = frozenset({"superseded", "resolved"})
_CHANGE = re.compile(
    r"\b(moved|changed|changes|switched|no longer|actually|now|new|instead|stopped|replaced|renamed|"
    r"correction|updated|rescheduled|postponed|any ?more)\b", re.IGNORECASE)
#: "from Porto", "not Ana", "no longer at Sao Joao", "instead of SQLite": the old value.
_OLD_VALUE = re.compile(
    r"\b(moved from|switched from|changed from|not|no longer(?: at| in| with| a| an| the)?|"
    r"instead of|rather than|replac\w+)\s+((?:[A-Z0-9][\w'.:+-]*)(?:\s+(?:de |da |do )?[A-Z0-9][\w'.:+-]*){0,3})")
#: Words that signal a change but say nothing about the subject.
_CHANGE_WORDS = frozenset(content_words(
    "moved changed switched longer actually now new instead stopped replaced renamed correction "
    "updated rescheduled postponed anymore same time big news one"))


def _meta(fragment: Any) -> dict[str, Any]:
    meta = getattr(fragment, "metadata", None)
    return meta if isinstance(meta, dict) else {}


def retired(fragment: Any) -> bool:
    """True when a revision retired the memory (superseded or resolved)."""
    tags = set(getattr(fragment, "tags", None) or ())
    meta = _meta(fragment)
    return bool(tags & RETIRED_TAGS or str(meta.get("lifecycle") or "").lower() in _RETIRED_LIFECYCLE
                or meta.get("superseded_by"))


def resolved(fragment: Any) -> bool:
    """True when ``resolve()`` closed the memory (a fixed bug, a done TODO)."""
    tags = set(getattr(fragment, "tags", None) or ())
    return "mark:resolved" in tags or str(_meta(fragment).get("lifecycle") or "").lower() == "resolved"


def _old_values(text: str) -> list[tuple[str, str]]:
    """(negating phrase, old value) pairs: ("moved from", "Porto"), ("not", "Ana")."""
    return [(m.group(1).lower(), m.group(2).rstrip(".,;:'")) for m in _OLD_VALUE.finditer(text or "")]


def _contains_value(text: str, value: str) -> bool:
    return bool(value) and re.search(r"(?<!\w)" + re.escape(value) + r"(?!\w)", text or "", re.IGNORECASE) is not None


#: Characters of a memory read for replacement (see the module docstring).
_WINDOW = 600


class _Profile(NamedTuple):
    text: str                          # the window
    words: frozenset[str]              # content words minus change words
    old_values: tuple[tuple[str, str], ...]
    change: bool                       # has a change word ("moved", "now", ...)


@lru_cache(maxsize=1024)   # keyed by the window text: an edited memory is a new key
def _profile_of(text: str) -> _Profile:
    return _Profile(text, content_words(text) - _CHANGE_WORDS, tuple(_old_values(text)),
                    _CHANGE.search(text) is not None)


def _profile(fragment: Any) -> _Profile:
    return _profile_of((getattr(fragment, "content", "") or "")[:_WINDOW])


def replaces(newer: Any, older: Any) -> bool:
    """Does the *newer* statement replace what the *older* one said?"""
    return _replaces(_profile(newer), _profile(older))


def _replaces(new: _Profile, old: _Profile) -> bool:
    shared = new.words & old.words
    for negator, value in new.old_values:
        # A bare "not X" also needs a shared topic word ("sister"): "not Monday" alone is too weak.
        if len(value) > 1 and _contains_value(old.text, value) \
                and (negator != "not" or shared - content_words(value)):
            return True
    if not new.change:
        return False
    return len({w for w in shared if not w.isdigit()}) >= 2


def replaced_by(fragments: Iterable[Any], key: Callable[[Any], Any]) -> dict[str, str]:
    """``{old id: newer id}`` for every line of *fragments* a later line replaces.

    *key* orders fragments in time (oldest lowest); an equal key never replaces.
    Instead of every pair, an older line is compared only with newer lines
    that can replace anything (a change word or an old value) and share a
    content word with it, or that name an old value ("moved from Porto"):
    no other pair can match (see ``_replaces``).
    """
    pool = {getattr(f, "id", None): f for f in fragments if getattr(f, "id", None)}
    timed = sorted(pool.values(), key=key)
    keys = [key(f) for f in timed]
    profiles = [_profile(f) for f in timed]
    by_word: dict[str, list[int]] = {}
    for position, profile in enumerate(profiles):
        if not (profile.change or profile.old_values):
            continue   # a plain statement never replaces another
        for word in profile.words:
            by_word.setdefault(word, []).append(position)
    naming = {i for i, p in enumerate(profiles) if any(n != "not" for n, _ in p.old_values)}
    out: dict[str, str] = {}
    for index, older in enumerate(timed):
        candidates = {j for w in profiles[index].words for j in by_word.get(w, ()) if j > index}
        candidates.update(j for j in naming if j > index)
        for j in sorted(candidates, reverse=True):   # the newest replacement wins
            if keys[j] != keys[index] and _replaces(profiles[j], profiles[index]):
                out[older.id] = timed[j].id
                break
    return out


def outdated_ids(fragments: Iterable[Any], key: Callable[[Any], Any]) -> dict[str, str | None]:
    """``{id: replacing id or None}`` for retired lines and lines replaced in the pool."""
    fragments = list(fragments)
    out: dict[str, str | None] = {f.id: None for f in fragments if getattr(f, "id", None) and retired(f)}
    out.update(replaced_by(fragments, key))
    return out
