"""Negations and corrections: turn "no longer X" into a retraction of X.

A negated clause states that a value does *not* hold. MARK cannot store
"not 8080" as a value, but it can withdraw 8080 when that is the current
value. So a negated clause is rewritten in the positive form, matched by the
ordinary packs, and the resulting candidates are flagged ``retract=True``:

    "atlas-core no longer runs on port 8080" -> retract atlas-core.port = 8080
    "<user> does not live in Lagos anymore"  -> retract <user>.location = Lagos
    "<user> used to live in Madrid"          -> retract <user>.location = Madrid

Corrections ("Correction: X is actually Y, not Z") mark the assertion
``correction=True`` (the old value was wrong, not merely superseded) and a
trailing "not Z" clause retracts Z. Questions, conditionals and intents are
filtered before this module runs, so "will no longer" is never a retraction.
"""
from __future__ import annotations

import re

from mark.intelligence.fact_extraction.personal import conjugate
from mark.security.lazy_regex import lazy_compile

_ANYMORE = lazy_compile(r"\s*,?\s*\b(?:any\s?more|any\s+longer)\b", re.IGNORECASE)
_NO_LONGER = lazy_compile(r"\bno\s+longer\s+", re.IGNORECASE)
_USED_TO = lazy_compile(r"\bused\s+to\s+(?P<verb>[a-z]+)\b", re.IGNORECASE)
_AUX_NOT = lazy_compile(r"\b(?P<aux>is|are|was|were|has|have)\s+not\b|\b(?P<short>is|are|was|were|has|have)n['’]t\b",
                      re.IGNORECASE)
_DO_NOT = lazy_compile(r"\b(?:does\s+not|do\s+not|did\s+not|doesn['’]t|don['’]t|didn['’]t|never)\s+(?P<verb>[a-z]+)\b",
                     re.IGNORECASE)
_STILL_NEGATED = lazy_compile(r"\b(?:not|never|no\s+longer|neither|nor|cannot)\b|n['’]t\b", re.IGNORECASE)

_CORRECTION_LEAD = lazy_compile(
    r"^(?:correction|corrected|erratum|to\s+correct\s+(?:that|myself|this)|update\s+to\s+the\s+above|sorry|oops)"
    r"\s*[:,\-–—]\s*|^actually\s*,?\s*", re.IGNORECASE)
_ACTUALLY = lazy_compile(r"\b(?:actually|in\s+fact)\s+", re.IGNORECASE)
_NOT_VALUE = lazy_compile(r"^(?:and\s+|but\s+)?not\s+(?P<value>[^,;]+?)\s*$", re.IGNORECASE)
_RATHER = lazy_compile(r"\s+(?:rather\s+than|instead\s+of)\s+(?P<value>[^,;]+?)\s*$", re.IGNORECASE)


def positive_form(clause: str) -> str | None:
    """The clause with its negation removed, or None when it is not a simple negation.

    Only shapes with an unambiguous positive form are rewritten; anything
    still negated afterwards ("not ... nor ...") is left alone.
    """
    text = _ANYMORE.sub("", clause)
    rewritten = _NO_LONGER.sub("", text)
    rewritten = _USED_TO.sub(lambda m: conjugate(m.group("verb")), rewritten)
    rewritten = _AUX_NOT.sub(lambda m: (m.group("aux") or m.group("short")), rewritten)
    rewritten = _DO_NOT.sub(lambda m: conjugate(m.group("verb")), rewritten)
    if rewritten == clause or _STILL_NEGATED.search(rewritten):
        return None
    return re.sub(r"\s+", " ", rewritten).strip()


_STOPPED = lazy_compile(r"\b(?:has\s+|have\s+|had\s+)?(?:stopped|stops|quit|quits|gave\s+up|gaves\s+up|gives\s+up"
                        r"|given\s+up)\s+(?P<verb>drinking|using)\b", re.IGNORECASE)
# "<who> gave up coffee": giving up a drink (no verb) is no longer drinking it.
_GAVE_UP_DRINK = lazy_compile(
    r"\b(?:has\s+|have\s+|had\s+)?(?:quit|quits|gave\s+up|gaves\s+up|gives\s+up|given\s+up)\s+"
    r"(?P<drink>coffee|tea|green\s+tea|black\s+tea|alcohol|soda|fizzy\s+drinks|energy\s+drinks|beer|wine|caffeine"
    r"|cola|coke|milk|juice)\b(?=\s*$|\s*[,;.])", re.IGNORECASE)
# "... ; I switched to green tea": the new value of the slot the previous clause retracted.
_SWITCHED = lazy_compile(r"^(?:\S+\s+)?(?:has\s+|have\s+)?(?:switched|changed|moved\s+over)\s+(?:over\s+)?to\s+"
                         r"(?:drinking\s+|using\s+)?(?P<value>[^,;]+?)\s*(?:instead)?\s*$", re.IGNORECASE)


_CANT_STAND = lazy_compile(r"\b(?:can['’]t|cannot|can\s+not)\s+stand\b", re.IGNORECASE)


def rewrite_stopped(clause: str) -> str:
    """Idioms whose negation is not a retraction.

    "<who> stopped drinking beer" / "<who> gave up coffee" -> "<who> no longer
    drinks beer / coffee" (then read as a negation); "<who> can't stand X" ->
    "<who> hates X" (a preference, not a denial).
    """
    low = clause.lower()
    if "stand" not in low and "stop" not in low and "quit" not in low and "gave" not in low and "give" not in low:
        return clause  # the common case: no idiom, no regex work
    clause = _CANT_STAND.sub("hates", clause)
    clause = _GAVE_UP_DRINK.sub(lambda m: "no longer drinks " + m.group("drink"), clause)
    return _STOPPED.sub(lambda m: "no longer " + ("drinks" if m.group("verb").lower() == "drinking" else "uses"),
                        clause)


def switched_value(clause: str) -> str | None:
    """The Y of "<who> switched to Y" (only meaningful right after a retraction), or None."""
    match = _SWITCHED.match(clause.strip().rstrip(".!"))
    return match.group("value").strip() if match else None


def is_cessation(clause: str) -> bool:
    """True for "used to ..." (a past state that no longer holds, without a negation word)."""
    return bool(_USED_TO.search(clause))


def strip_correction(sentence: str) -> tuple[str, bool]:
    """Remove a correction marker; returns (sentence, is_correction).

    "Correction: X is actually Y" and "Actually, X is Y" both mark a
    correction; a bare "actually" inside the clause is dropped as well.
    """
    lead = _CORRECTION_LEAD.match(sentence)
    corrected = lead is not None
    if lead:
        sentence = sentence[lead.end():]
    if _ACTUALLY.search(sentence):
        corrected = True
        sentence = _ACTUALLY.sub("", sentence, count=1)
    return sentence, corrected


def denied_value(clause: str) -> str | None:
    """The Z of a trailing ", not Z" clause (the value a correction rules out)."""
    match = _NOT_VALUE.match(clause.strip().rstrip(".!"))
    return match.group("value") if match else None


def split_rather(clause: str) -> tuple[str, str | None]:
    """"X uses Y rather than Z" -> ("X uses Y", "Z")."""
    match = _RATHER.search(clause.rstrip(".!"))
    if match is None:
        return clause, None
    return clause[:match.start()], match.group("value")
