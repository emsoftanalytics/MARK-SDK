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

from mark.context.provenance import (
    UNTRUSTED_TAG,
    is_assistant,
    is_error_record,
    is_question_record,
    is_request_record,
)

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


def fact_line(fact: Any, previous: Any = None, *, since: bool = True) -> str:
    """``- user city = Braga (was Porto; since 2026-10-03)``: one dense line per current fact.

    *previous* is the value it replaced (small models need "was" to answer
    "where did I live before?" from the fact alone); ``since=False`` drops the date.
    """
    when = getattr(fact, "valid_from", None) if since else None
    parts = [f"was {escape_tags(str(previous))}"] if previous not in (None, "") and previous != fact.value else []
    parts += [f"since {when.date().isoformat()}"] if when is not None else []
    tail = f" ({'; '.join(parts)})" if parts else ""
    return f"- {escape_tags(str(fact.subject))} {escape_tags(str(fact.attribute))} = {escape_tags(str(fact.value))}{tail}"


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
    """A record that may be shown as a note: not the model's words, not a bare
    question or request, not a failed tool call."""
    return not (is_assistant(fragment) or is_question_record(fragment)
                or is_request_record(fragment) or is_error_record(fragment))


def small_talk(fragment: Any) -> bool:
    """A short greeting or sign-off with no information in it."""
    text = strip_role(getattr(fragment, "content", "") or "").strip()
    return len(text.split()) <= _SMALL_TALK_WORDS and bool(_SMALL_TALK.match(text))


def note_text(fragment: Any, line_chars: int, *, outdated: bool = False, prefix: str = "") -> str:
    """``- <content>`` for a briefing (without a "User said:" prefix).

    Instruction-like memories (``mark:untrusted``, set by the security layer)
    are labelled "(untrusted)" so the model reads them as data, outdated ones
    "(outdated)"; boundary-tag lookalikes in the text are escaped.
    """
    from mark.context.watch_support import _one_line

    label = "(untrusted) " if is_untrusted(fragment) else ""
    label += "(outdated) " if outdated else ""
    body = escape_tags(strip_role(getattr(fragment, "content", "") or ""))
    return f"- {label}{prefix}{_one_line(body, line_chars)}"


def is_untrusted(fragment: Any) -> bool:
    return UNTRUSTED_TAG in (getattr(fragment, "tags", None) or ())


#: Text that could pass for a MARK block boundary (``</mark_briefing>``,
#: ``[/MARK project memory]``): a stored memory must never close the block early.
_BOUNDARY = re.compile(r"<\s*/?\s*mark[\w\s:-]*>|\[\s*/?\s*mark\b[^\]]*\]", re.IGNORECASE)


def escape_tags(text: str) -> str:
    """*text* with MARK boundary-tag lookalikes defused (``<mark_briefing>`` -> ``‹mark_briefing›``)."""
    return _BOUNDARY.sub(lambda m: m.group(0).replace("<", "‹").replace(">", "›")
                         .replace("[", "⟦").replace("]", "⟧"), text or "")


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
