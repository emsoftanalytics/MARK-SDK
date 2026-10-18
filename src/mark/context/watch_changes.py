"""Change statements and replaced fact values for the context watch.

Split from ``watch_prefetch``: the parts of a prefetch that answer "what is
it *now*, and what was it before?" — the user's change statements ranked
against the question (:func:`_change_lane`) and the value each current fact
replaced (:func:`_previous_values`, rendered "Braga (was Porto)").
"""
from __future__ import annotations

import logging
import re
from typing import Any

from mark.context.provenance import USER_TAG
from mark.context.relevance import content_words
from mark.context.watch_lines import fact_key, small_talk, trusted

logger = logging.getLogger("mark.context.watch")


#: Questions about a value now, before or since it changed.
_STATE_QUESTION = re.compile(
    r"\b(now|current(ly)?|before|originally|original|old|previous(ly)?|used to|chang\w*|"
    r"still|any ?more|latest|new|moved|switched)\b", re.IGNORECASE)
#: Statements that replace an earlier value ("moved from Porto to Braga").
_CHANGE_STATEMENT = re.compile(
    r"\b(moved|changed|switched|no longer|actually|now|new|instead|stopped|started|"
    r"replaced|renamed|correction|rescheduled|not \w+ any ?more)\b", re.IGNORECASE)
_CHANGE_QUERY = "moved changed switched now no longer actually new instead stopped started replaced"
#: Question words -> words of the statements that change them ("where do I
#: live now?" shares no word with "I moved from Porto to Braga").
_TOPICS = (
    (frozenset({"live", "city", "home", "address", "where", "town", "country"}), frozenset({"mov", "live", "city"})),
    (frozenset({"job", "work", "role", "position", "employer", "career"}), frozenset({"job", "work", "manager", "role"})),
    (frozenset({"name", "call"}), frozenset({"actually", "renam", "call", "name"})),
    (frozenset({"phone", "number", "contact"}), frozenset({"number", "phone"})),
    (frozenset({"eat", "diet", "food", "meat"}), frozenset({"eat", "meat", "vegetarian", "vegan", "diet"})),
)


def _change_similarity(task_words: frozenset[str], text: str) -> int:
    """Shared content words, plus topic hints, between a question and a change statement."""
    words = content_words(text)
    score = 2 * len(task_words & words)
    for asks, says in _TOPICS:
        if task_words & asks and any(w.startswith(tuple(says)) for w in words):
            score += 1
    return score


def _change_lane(memory: Any, task: str, limit: int) -> tuple[list[Any], list[Any]]:
    """The user's change statements, for questions about current or earlier values.

    "Which city do I live in now?" shares no word with "I moved from Porto
    to Braga", so relevance search alone misses the one line that answers
    it (a10 replay, llama3.2:3b). For such questions the user's statements
    that replace an earlier value are fetched as their own small group:
    sign-offs ("Bye for now") are dropped *before* the limit, and the rest
    are ranked by how much they share with the question (a10 small-model
    run, root cause 3: one fixed query returned the same four lines for
    every question, small talk included). Statements that share nothing
    with the question are left out unless nothing else matches.

    Returns ``(shown, pool)``: *pool* holds every change statement found, also
    for other questions, so the briefing can tell which older lines they
    replaced ("(outdated)") even when it does not show them.
    """
    if limit <= 0:
        return [], []
    try:
        lanes = memory.retrieve_many_sync([task, _CHANGE_QUERY], tags=[USER_TAG])
    except Exception as exc:
        logger.warning("change-statement retrieval failed: %s", exc)
        return [], []
    seen: dict[str, Any] = {}
    for lane in lanes:
        for fragment in lane.fragments:
            if trusted(fragment) and not small_talk(fragment) and _CHANGE_STATEMENT.search(fragment.content):
                seen.setdefault(fragment.id, fragment)
    task_words = content_words(task)
    ranked = sorted(seen.values(), key=lambda f: -_change_similarity(task_words, f.content))
    if not _STATE_QUESTION.search(task):
        return [], ranked
    matching = [f for f in ranked if _change_similarity(task_words, f.content) > 0]
    return (matching or ranked)[:limit], ranked


def _previous_values(memory: Any, results: list[Any]) -> dict[tuple[str, str], Any]:
    """The value each current fact of *results* replaced ("Braga (was Porto)")."""
    out: dict[tuple[str, str], Any] = {}
    for result in results:
        for fact in getattr(result, "facts", None) or ():
            if not getattr(fact, "is_current", True) or fact_key(fact) in out:
                continue
            try:
                history = memory.facts.history(fact.subject, fact.attribute)
            except Exception:   # stores without fact history
                return out
            older = [h.value for h in history[:-1] if h.value != fact.value]
            out[fact_key(fact)] = older[-1] if older else None
    return out



#: First-person requests ("Suggest a dish for me", "my pets"): the user's own
#: facts are the context ("user diet = vegetarian"), whatever the words.
_FIRST_PERSON = re.compile(r"\b(i|me|my|mine|myself|i'm|i've)\b", re.IGNORECASE)


def user_profile(memory: Any, task: str, limit: int) -> list[Any]:
    """Current facts about the user for a first-person request (at most *limit*).

    a10 small-model run: "Suggest a dinner dish for me" shares no word with
    "I stopped eating meat", so llama3.2:1b suggested chicken; the fact
    ``user diet = vegetarian`` is six tokens and answers it.
    """
    if limit <= 0 or not _FIRST_PERSON.search(task or ""):
        return []
    subject = getattr(getattr(memory, "_pipeline", None), "_user_subject", None) or "user"
    try:
        facts = [f for f in memory.facts.current(subject) if getattr(f, "is_current", True)]
    except Exception:   # stores without facts
        return []
    facts.sort(key=lambda f: getattr(f, "valid_from", None) or 0, reverse=True)
    return facts[:limit]
