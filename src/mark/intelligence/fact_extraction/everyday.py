"""Whole-sentence shapes from everyday talk that clause splitting would break apart.

Run on one sentence after first-person rewriting, before clauses are split:

  - a changed contact detail with the new value in a second clause:
    "Kemi changed her number, the new one is +234 803 555 0100" ->
    ``Kemi.phone = +234 ...`` (a plain set, so it supersedes the old number);
    also "... changed her email to bo@new.io";
  - the last step of a procedure: "Step 7 is the last step of the deploy
    procedure" -> ``deploy procedure.step count = 7``; without a named
    procedure the one whose steps were listed just before is used, and with
    neither nothing is recorded.

Each value must look like what it claims to be (a phone has 7+ digits, an
email an "@"), otherwise the sentence is left to the ordinary packs.
"""
from __future__ import annotations

import re

from mark.intelligence.fact_extraction.numbers import NUMBER_WORD, parse_number
from mark.intelligence.fact_extraction.personal import USER_TOKEN
from mark.security.lazy_regex import lazy_compile

Triple = tuple[str, str, str]
_WHO = rf"(?P<who>{USER_TOKEN}|(?-i:[A-Z])[\w.'’-]*(?:\s+(?-i:[A-Z])[\w.'’-]*){{0,2}})"
_POSSESSIVE = rf"(?:his|her|their|{USER_TOKEN}['’]s)"
_DETAIL = (r"(?P<noun>(?:phone\s+|mobile\s+|cell\s+)?number|phone|mobile|email(?:\s+address)?|e-mail"
           r"|(?:home\s+)?address)")
_CONTACT_CHANGE = lazy_compile(
    rf"^{_WHO}\s+(?:has\s+|just\s+|recently\s+)*(?:changed|updated|got)\s+{_POSSESSIVE}\s+(?:new\s+)?{_DETAIL}"
    r"(?:\s*[,;:—-]+\s*(?:and\s+)?(?:the\s+new\s+(?:one|number|email|address)|it)\s+is(?:\s+now)?\s+"
    r"|\s+to\s+)(?P<value>.+?)\s*[.!]?\s*$", re.IGNORECASE)
_PHONE = lazy_compile(r"^\+?[\d\s().-]{7,}$")
_EMAIL = lazy_compile(r"^[\w.+%-]+@[\w-]+(?:\.[\w-]+)+$")
_LAST_STEP = lazy_compile(
    rf"^(?:step\s+(?P<n>{NUMBER_WORD})|the\s+(?P<ord>{NUMBER_WORD})\s+step)"
    r"(?:\s+(?:of|in)\s+(?:the\s+|our\s+)?(?P<p1>[^,]+?))?\s+is\s+the\s+(?:last|final)(?:\s+(?:one|step))?"
    r"(?:\s+(?:of|in)\s+(?:the\s+|our\s+)?(?P<p2>[^,]+?))?\s*[.!]?\s*$", re.IGNORECASE)


def sentence_triples(sentence: str, procedures: list[str]) -> list[Triple] | None:
    """Triples for one of the shapes above, or None when the sentence is not one."""
    low = sentence.lower()
    if "step" in low and ("last" in low or "final" in low):
        return _last_step(sentence, procedures)
    if any(word in low for word in ("changed", "updated", "got")):
        return _contact_change(sentence)
    return None


def _contact_change(sentence: str) -> list[Triple] | None:
    match = _CONTACT_CHANGE.match(sentence)
    if match is None:
        return None
    noun, value = match.group("noun").lower(), match.group("value").strip()
    if "mail" in noun:
        attribute, ok = "email", bool(_EMAIL.match(value))
    elif "address" in noun:
        attribute, ok = "address", 0 < len(value.split()) <= 10
    else:
        attribute, ok = "phone", bool(_PHONE.match(value)) and sum(c.isdigit() for c in value) >= 7
    return [(match.group("who"), attribute, value)] if ok else None


def _last_step(sentence: str, procedures: list[str]) -> list[Triple] | None:
    match = _LAST_STEP.match(sentence.strip())
    if match is None:
        return None
    number = parse_number(match.group("n") or match.group("ord") or "")
    procedure = (match.group("p1") or match.group("p2") or (procedures[-1] if procedures else "")).strip()
    if not number or not procedure:
        return None  # "Step 5 is the last one." with no procedure in sight says nothing usable
    return [(procedure, "step count", str(number))]
