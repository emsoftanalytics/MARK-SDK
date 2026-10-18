"""Lists of values in one sentence: "My emails are A and B", "I speak X, Y and Z".

A plural personal noun ("emails", "phone numbers", "languages") or the verb
"speaks" announces several values of one attribute at once. Each item
becomes its own *multi-valued* candidate (recorded with ``facts.add``), so
the values sit side by side instead of replacing one another.

All-or-nothing on purpose: when one item does not look like a value of that
attribute (an email without "@", a language that is not a capitalised
word), the sentence is left to the ordinary packs. "My kids are 7 and 10"
is not a list this module knows, so it records nothing.
"""
from __future__ import annotations

import re

from mark.intelligence.fact_extraction.personal import USER_TOKEN
from mark.security.lazy_regex import lazy_compile

_EMAIL = lazy_compile(r"^[\w.+%-]+@[\w-]+(?:\.[\w-]+)+$")
_PHONE = lazy_compile(r"^\+?[\d\s().-]{7,}$")
_CAPITALISED = lazy_compile(r"^(?-i:[A-Z])[\w'’-]*(?:\s+(?-i:[A-Z])[\w'’-]*)?$")
_WORDS = lazy_compile(r"^[\w'’.+#-]+(?:\s+[\w'’.+#-]+){0,2}$")

#: Plural noun -> (attribute, item check).
_PLURALS: dict[str, tuple[str, re.Pattern[str]]] = {
    "emails": ("email", _EMAIL), "email addresses": ("email", _EMAIL),
    "phones": ("phone", _PHONE), "phone numbers": ("phone", _PHONE), "mobile numbers": ("phone", _PHONE),
    "numbers": ("phone", _PHONE), "languages": ("language", _CAPITALISED),
    "nationalities": ("nationality", _CAPITALISED), "citizenships": ("nationality", _CAPITALISED),
    "hobbies": ("hobby", _WORDS), "allergies": ("allergy", _WORDS),
}
_PLURAL_ALT = "|".join(re.escape(p).replace(r"\ ", r"\s+") for p in sorted(_PLURALS, key=len, reverse=True))
_OWNER = rf"(?P<owner>{USER_TOKEN}|(?-i:[A-Z])[\w.'’-]*(?:\s+(?-i:[A-Z])[\w.'’-]*){{0,2}})"
_PLURAL_LIST = lazy_compile(rf"^{_OWNER}['’]s\s+(?P<noun>{_PLURAL_ALT})\s+(?:are|include)\s+(?P<items>.+?)[.!]?\s*$",
                          re.IGNORECASE)
_SPEAKS = lazy_compile(rf"^{_OWNER}\s+(?:also\s+|now\s+)?(?:speaks|is\s+fluent\s+in)\s+(?:fluent\s+)?(?P<items>.+?)[.!]?\s*$")
_REDACTED_ITEM = lazy_compile(r"^\[[A-Z]+(?::[a-z0-9_]+)?\]$")
_SPLIT = lazy_compile(r"\s*,\s*(?:and\s+|&\s+)?|\s+(?:and|&)\s+")


def list_triples(sentence: str) -> list[tuple[str, str, str]] | None:
    """(owner, attribute, item) for every item of a list sentence, or None when it is not one."""
    if "'s " not in sentence and "’s " not in sentence and "speaks" not in sentence and "fluent" not in sentence:
        return None  # cheap exit: neither list shape can match
    plural = _PLURAL_LIST.match(sentence)
    if plural is not None:
        key = re.sub(r"\s+", " ", plural.group("noun").lower())
        attribute, check = _PLURALS[key]
        return _items(plural.group("owner"), attribute, plural.group("items"), check)
    speaks = _SPEAKS.match(sentence)
    if speaks is not None:
        return _items(speaks.group("owner"), "language", speaks.group("items"), _CAPITALISED)
    return None


def _items(owner: str, attribute: str, text: str, check: re.Pattern[str]) -> list[tuple[str, str, str]] | None:
    items = [item.strip().strip("\"'`") for item in _SPLIT.split(text.strip()) if item.strip()]
    kept = [item for item in items if not _REDACTED_ITEM.match(item)]  # "[EMAIL]" is unknown, not wrong
    if not kept or len(items) > 8 or not all(check.match(item) for item in kept):
        return None
    return [(owner, attribute, item) for item in kept]
