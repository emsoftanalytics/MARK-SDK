"""Line-level helpers for briefings: dense facts, record filters and de-duplication.

A briefing line costs the same tokens whether it helps or not, so every
line is checked before it becomes a budget candidate:

* **Provenance.** The model's own earlier replies (``mark:assistant``) are
  never shown as notes about the user; pure questions (``mark:question``)
  carry no information and are skipped (see ``mark.context.provenance``).
* **No echo.** A stored copy of the message being answered right now is not
  "memory" and is dropped (usage test M7: the current question came back as
  a remembered fact).
* **No near-duplicates.** Two lines whose words are (almost) the same are
  shown once; the first, higher-priority one wins.
* **Dense facts.** A fact is one short line: ``subject attribute = value
  (since date)``.
"""
from __future__ import annotations

import re
from typing import Any, Iterable

from mark.context.provenance import is_assistant, is_question_record

_WORD = re.compile(r"\w+")
_ROLE_PREFIX = re.compile(r"^\s*(user said|user|assistant replied|assistant said)\s*:\s*", re.IGNORECASE)
#: Word-set overlap above which two lines count as the same line.
_NEAR_DUPLICATE = 0.85
#: Greetings and sign-offs ("Bye for now.", "Ok, that's the plan for now."):
#: stored, so nothing is lost, but never worth a briefing line.
_SMALL_TALK = re.compile(
    r"^(hi|hello|hey|thanks|thank you|ok|okay|bye|goodbye|good night|talk later|see you|"
    r"cheers|that'?s (it|all))\b", re.IGNORECASE)
_SMALL_TALK_WORDS = 8


def fact_line(fact: Any) -> str:
    """``- user city = Braga (since 2026-10-03)``: one dense line per current fact."""
    since = getattr(fact, "valid_from", None)
    when = f" (since {since.date().isoformat()})" if since is not None else ""
    return f"- {fact.subject} {fact.attribute} = {fact.value}{when}"


def fact_key(fact: Any) -> tuple[str, str]:
    return (str(fact.subject).lower(), str(fact.attribute).lower())


def strip_role(text: str) -> str:
    """Content without a leading "User said:" style prefix."""
    return _ROLE_PREFIX.sub("", text or "", count=1)


def words(text: str) -> frozenset[str]:
    return frozenset(_WORD.findall(strip_role(text).lower()))


def is_echo(fragment: Any, task_words: frozenset[str]) -> bool:
    """True when the record just repeats the message being answered."""
    if not task_words:
        return False
    return words(getattr(fragment, "content", "") or "") == task_words


def trusted(fragment: Any) -> bool:
    """A record that may be shown as a note: not the model's words, not a bare question."""
    return not is_assistant(fragment) and not is_question_record(fragment)


def small_talk(fragment: Any) -> bool:
    """A short greeting or sign-off with no information in it."""
    text = strip_role(getattr(fragment, "content", "") or "").strip()
    return len(text.split()) <= _SMALL_TALK_WORDS and bool(_SMALL_TALK.match(text))


class Deduper:
    """Remembers kept lines; :meth:`fresh` is False for a (near) repeat."""

    def __init__(self) -> None:
        self._seen: list[frozenset[str]] = []

    def fresh(self, text: str) -> bool:
        key = words(text)
        if not key:
            return False
        for seen in self._seen:
            overlap = len(key & seen) / max(len(key | seen), 1)
            if overlap >= _NEAR_DUPLICATE:
                return False
        self._seen.append(key)
        return True


def current_facts(results: Iterable[Any]) -> list[Any]:
    """Current facts of several retrieval results, first occurrence of each (subject, attribute)."""
    seen: set[tuple[str, str]] = set()
    out: list[Any] = []
    for result in results:
        for fact in getattr(result, "facts", None) or []:
            if getattr(fact, "is_current", True) and fact_key(fact) not in seen:
                seen.add(fact_key(fact))
                out.append(fact)
    return out
