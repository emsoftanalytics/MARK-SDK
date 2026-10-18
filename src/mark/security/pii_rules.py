# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 MARK Contributors
"""Opt-in PII rules: emails, phone numbers, IBANs and US SSNs.

These are *identifiers*, not credentials: agents legitimately need to
remember "Dr. Bello's clinic phone" or where to send an invoice, so the
default write policy keeps them and ``RedactionPolicy(pii=True)`` removes
them. Placeholders keep the legacy names (``[EMAIL]``, ``[PHONE]``,
``[SSN]``) that fact extraction already recognises, plus ``[IBAN]``.
Matches always include the country code / parentheses so nothing like
"+234 [PHONE]" or "([PHONE]" is left behind.
"""
from __future__ import annotations

import re

from mark.security.checksums import iban_valid
from mark.security.rules import Rule
from mark.security.lazy_regex import lazy_compile

EMAIL_PATTERN = lazy_compile(r"(?<![\w.+%-])[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")

# E.164-style international numbers with spaces, dots, dashes and an optional
# trunk prefix in parentheses: +234 803 555 0192, +44 (0)20 7946 0958, +49-30-901820.
_INTL_PHONE = lazy_compile(r"(?<![\w+])\+\d{1,3}(?:[ .\-]?\(\d{1,4}\))?(?:[ .\-]?\d{1,5}){1,6}(?!\w)")
# North-American numbers with separators, including the opening parenthesis.
_NANP_PHONE = lazy_compile(r"(?<![\w(+.])(?:\(\d{3}\)[ .\-]?|\d{3}[.\-])\d{3}[.\-]\d{4}(?![\w\-])")
# National-format numbers with a trunk "0" (0803-555-0177, 020 7946 0958,
# 06 12 34 56 78, (020) 7946 0959): 3+ separated groups and 9-11 digits, so
# versions (0.10.12), dates (01-02-2026) and times never qualify.
_NATIONAL_PHONE = lazy_compile(
    r"(?<![\w+(.\-/:])(?:\(0[1-9]\d{0,3}\)[ .\-]?\d{2,4}(?:[ .\-]\d{2,4}){1,3}|0[1-9]\d{0,3}(?:[ .\-]\d{2,4}){2,4})"
    r"(?![\w\-/:]|\.\d)")
_IBAN = lazy_compile(r"\b[A-Z]{2}\d{2}(?:[ ]?[A-Z0-9]{4}){2,7}(?:[ ]?[A-Z0-9]{1,3})?\b")
_SSN = lazy_compile(r"\b(?!000|666|9\d\d)\d{3}-(?!00)\d{2}-(?!0000)\d{4}\b")


def _phone_ok(match: "re.Match[str]") -> bool:
    digits = sum(ch.isdigit() for ch in match.group(0))
    return 7 <= digits <= 15  # E.164 maximum is 15 digits


def _national_phone_ok(match: "re.Match[str]") -> bool:
    text = match.group(0)
    digits = sum(ch.isdigit() for ch in text)
    # One separator style throughout: "06 12 34 56 78" is a phone, "01.02.2026 10" is not.
    separators = set(re.findall(r"[ .\-]", text.replace(") ", ")")))
    if not (9 <= digits <= 11 and len(separators) <= 1):
        return False
    # Hex dumps ("7f 45 4c 46 02 01 01 00 00 00") are byte pairs, not phone groups.
    before = match.string[max(0, match.start() - 4):match.start()]
    after = match.string[match.end():match.end() + 4]
    return not (re.search(r"\b[0-9A-Fa-f]{2} $", before) or re.match(r" [0-9A-Fa-f]{2}\b", after))


def _iban_ok(match: "re.Match[str]") -> bool | int:
    """Accept a valid IBAN, trimming trailing words the greedy match took.

    ``DE89 3704 0044 0532 0130 00 EUR`` matches with " EUR" attached; drop
    trailing space-separated groups until the checksum passes.
    """
    text = match.group(0)
    while text:
        if iban_valid(text):
            return True if text == match.group(0) else len(text)
        if " " not in text:
            return False
        text = text.rsplit(" ", 1)[0]
    return False


EMAIL_RULE = Rule("email", EMAIL_PATTERN, placeholder="[EMAIL]", hint="@")
PHONE_RULES: list[Rule] = [
    Rule("phone", _INTL_PHONE, placeholder="[PHONE]", validate=_phone_ok, hint="+"),
    Rule("phone", _NANP_PHONE, placeholder="[PHONE]", validate=_phone_ok, prefilter=lazy_compile(r"\d{3}[.\-]\d{4}")),
    Rule("phone", _NATIONAL_PHONE, placeholder="[PHONE]", validate=_national_phone_ok,
         prefilter=lazy_compile(r"(?<!\d)\(?0\d")),
]
OTHER_PII_RULES: list[Rule] = [
    Rule("iban", _IBAN, placeholder="[IBAN]", validate=_iban_ok, prefilter=lazy_compile(r"[A-Z]{2}\d{2}")),
    Rule("ssn", _SSN, placeholder="[SSN]", prefilter=lazy_compile(r"\d{3}-\d{2}-\d{4}")),
]

__all__ = ["EMAIL_PATTERN", "EMAIL_RULE", "OTHER_PII_RULES", "PHONE_RULES"]
