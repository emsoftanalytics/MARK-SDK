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

EMAIL_PATTERN = re.compile(r"(?<![\w.+%-])[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")

# E.164-style international numbers with spaces, dots, dashes and an optional
# trunk prefix in parentheses: +234 803 555 0192, +44 (0)20 7946 0958, +49-30-901820.
_INTL_PHONE = re.compile(r"(?<![\w+])\+\d{1,3}(?:[ .\-]?\(\d{1,4}\))?(?:[ .\-]?\d{1,5}){1,6}(?!\w)")
# North-American numbers with separators, including the opening parenthesis.
_NANP_PHONE = re.compile(r"(?<![\w(+.])(?:\(\d{3}\)[ .\-]?|\d{3}[.\-])\d{3}[.\-]\d{4}(?![\w\-])")
_IBAN = re.compile(r"\b[A-Z]{2}\d{2}(?:[ ]?[A-Z0-9]{4}){2,7}(?:[ ]?[A-Z0-9]{1,3})?\b")
_SSN = re.compile(r"\b(?!000|666|9\d\d)\d{3}-(?!00)\d{2}-(?!0000)\d{4}\b")


def _phone_ok(match: "re.Match[str]") -> bool:
    digits = sum(ch.isdigit() for ch in match.group(0))
    return 7 <= digits <= 15  # E.164 maximum is 15 digits


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
    Rule("phone", _NANP_PHONE, placeholder="[PHONE]", validate=_phone_ok, prefilter=re.compile(r"\d{3}[.\-]\d{4}")),
]
OTHER_PII_RULES: list[Rule] = [
    Rule("iban", _IBAN, placeholder="[IBAN]", validate=_iban_ok, prefilter=re.compile(r"[A-Z]{2}\d{2}")),
    Rule("ssn", _SSN, placeholder="[SSN]", prefilter=re.compile(r"\d{3}-\d{2}-\d{4}")),
]

__all__ = ["EMAIL_PATTERN", "EMAIL_RULE", "OTHER_PII_RULES", "PHONE_RULES"]
