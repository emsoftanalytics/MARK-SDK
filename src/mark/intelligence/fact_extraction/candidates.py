"""Small candidate builders shared by :class:`~.patterns.PatternFactExtractor`.

  - :func:`triple` builds a candidate outside the regex packs (records,
    steps, labels, lists), cleaned and validated;
  - :func:`retraction` withdraws a value in the slot of another candidate;
  - :func:`previous_value` turns the "from A" of "from A to B" into a
    ``previous`` candidate (history before B, never current);
  - :func:`for_user` replaces the first-person placeholder with the user
    subject, and is the one place that decides what a leftover placeholder
    in a value means ("I like my tea without sugar" -> "tea without sugar",
    "I love my dog Rex" -> ``user.dog = Rex``). The placeholder never leaks.
"""
from __future__ import annotations

import re
from dataclasses import replace
from datetime import datetime

from mark.intelligence.fact_extraction import personal
from mark.intelligence.fact_extraction.base import FactCandidate, clean_text, is_redacted, valid_candidate
from mark.intelligence.fact_extraction.normalise import peel_value
from mark.intelligence.fact_extraction.vocab import RELATION_ALT
from mark.security.lazy_regex import lazy_compile

_OWNED = lazy_compile(rf"^{personal.USER_TOKEN}['’]s\s+(?P<rest>.+)$")
_RELATION_NAME = lazy_compile(rf"^(?P<relation>{RELATION_ALT})\s+(?P<name>(?-i:[A-Z])[\w'’-]*(?:\s+(?-i:[A-Z])[\w'’-]*)?)$",
                            re.IGNORECASE)


def triple(subject: str, attribute: str, value: str, when: datetime | None, rule: str,
           *, multi: bool = False) -> FactCandidate | None:
    """A candidate built outside the regex packs, cleaned and validated (None if unusable)."""
    subject, value = clean_text(subject), personal.restore(clean_text(value))
    if is_redacted(subject) or is_redacted(value):
        return None
    candidate = FactCandidate(subject=subject, attribute=clean_text(attribute).lower(), value=value,
                              valid_from=when, confidence=0.7, rule=rule, multi=multi)
    return candidate if valid_candidate(candidate) else None


def keep_attribute_case(candidate: FactCandidate | None, attribute: str) -> FactCandidate | None:
    """Config keys keep their spelling ("MAX_UPLOAD_MB"); fact keys fold case anyway."""
    return None if candidate is None else replace(candidate, attribute=attribute)


def retraction(candidate: FactCandidate | None, value: str) -> FactCandidate | None:
    """A retraction of *value* in the slot of *candidate*."""
    if candidate is None or not value:
        return None
    return replace(candidate, value=value, retract=True, correction=False, previous=False)


def previous_value(candidate: FactCandidate | None, old: str | None) -> FactCandidate | None:
    """The "A" of "<S> changed from A to B" as a ``previous`` candidate in B's slot."""
    if candidate is None or not old or candidate.retract:
        return None
    value = peel_value(clean_text(old))[0]
    value = re.sub(r"^(?:the|a|an)\s+", "", value, flags=re.IGNORECASE)
    if (not value or is_redacted(value) or len(value.split()) > 6
            or value.lower() == candidate.value.lower() or personal.USER_TOKEN in value):
        return None
    if candidate.rule == "status" or candidate.rule.startswith("status"):
        value = value.lower()
    return replace(candidate, value=value, previous=True, change=False)


def for_user(candidate: FactCandidate, user_subject: str) -> FactCandidate | None:
    """Replace the first-person placeholder with *user_subject* (None when it cannot be placed)."""
    subject, value = candidate.subject, candidate.value
    if personal.USER_TOKEN in subject and subject != personal.USER_TOKEN:
        return None  # "<user>'s favourite" is a mis-split, not a subject
    owned = _OWNED.match(value)
    if owned is not None and candidate.rule == "preference":
        rest = owned.group("rest")
        relation = _RELATION_NAME.match(rest)
        if relation is not None:
            # "I love my dog Rex": the useful fact is who Rex is.
            candidate = replace(candidate, attribute=relation.group("relation").lower(),
                                value=relation.group("name"), multi=False, rule="personal_relation")
            value = candidate.value
        elif len(rest.split()) >= 2:
            value = rest  # "I like my tea without sugar" -> likes "tea without sugar"
        else:
            return None  # "I love my job": which job is not in the text
    if personal.USER_TOKEN in value and value != personal.USER_TOKEN:
        return None
    if candidate.rule == "personal_relation" and value[-2:] in ("'s", "’s"):
        value = value[:-2]  # "my friend Joao's phone ...": the friend is Joao, not "Joao's"
    if personal.is_user(subject):
        subject = user_subject
    if value == personal.USER_TOKEN:
        value = user_subject
    if (subject, value) == (candidate.subject, candidate.value):
        return candidate
    return replace(candidate, subject=subject, value=value)
