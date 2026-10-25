# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 Somtochukwu Owen Ikeanyi
"""Checksum validators that keep number-shaped redaction rules precise.

A 16-digit order number or a 22-character customer reference looks exactly
like a card number or an IBAN to a regex. Requiring the real checksum
(Luhn for cards, ISO 13616 mod-97 for IBANs) removes ~90% / ~99% of those
false positives at negligible cost.
"""
from __future__ import annotations

import re
from mark.security.lazy_regex import lazy_compile

_NON_DIGIT = lazy_compile(r"\D")
_NON_ALNUM = lazy_compile(r"[^A-Za-z0-9]")

# ISO 13616 IBAN lengths per country (the common ones; unknown countries
# fall back to the generic 15–34 range so new registrations still match).
_IBAN_LENGTHS = {
    "AD": 24, "AE": 23, "AT": 20, "BE": 16, "BG": 22, "BH": 22, "BR": 29, "CH": 21,
    "CY": 28, "CZ": 24, "DE": 22, "DK": 18, "EE": 20, "ES": 24, "FI": 18, "FR": 27,
    "GB": 22, "GR": 27, "HR": 21, "HU": 28, "IE": 22, "IL": 23, "IS": 26, "IT": 27,
    "KW": 30, "KZ": 20, "LT": 20, "LU": 20, "LV": 21, "MT": 31, "NL": 18, "NO": 15,
    "PL": 28, "PT": 25, "QA": 29, "RO": 24, "SA": 24, "SE": 24, "SI": 19, "SK": 24,
    "TR": 26, "UA": 29,
}


def luhn_valid(text: str) -> bool:
    """Return True when the digits in *text* form a Luhn-valid 13–19 digit PAN.

    The issuer prefix must be 2–6 (Mastercard 2-series, Amex/Diners 3,
    Visa 4, Mastercard 5, Discover 6): numbers starting with 0, 1, 7, 8 or 9
    are timestamps, order ids or phone numbers far more often than cards.
    """
    digits = _NON_DIGIT.sub("", text)
    if not 13 <= len(digits) <= 19 or digits[0] not in "23456":
        return False
    if len(set(digits)) == 1:  # 4444…: Luhn-valid but never a real PAN
        return False
    total = 0
    for index, char in enumerate(reversed(digits)):
        value = int(char)
        if index % 2 == 1:
            value *= 2
            if value > 9:
                value -= 9
        total += value
    return total % 10 == 0


def iban_valid(text: str) -> bool:
    """Return True when *text* (spaces allowed) is an ISO 13616 valid IBAN."""
    iban = _NON_ALNUM.sub("", text).upper()
    if len(iban) < 15 or len(iban) > 34 or not iban[:2].isalpha() or not iban[2:4].isdigit():
        return False
    expected = _IBAN_LENGTHS.get(iban[:2])
    if expected is not None and len(iban) != expected:
        return False
    rearranged = iban[4:] + iban[:4]
    # Letters become two-digit numbers (A=10 … Z=35) before the mod-97 test.
    numeric = "".join(str(int(ch, 36)) for ch in rearranged)
    return int(numeric) % 97 == 1


__all__ = ["iban_valid", "luhn_valid"]
