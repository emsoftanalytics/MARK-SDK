# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 MARK Contributors
"""Default redaction patterns: recall on planted secrets, no partial leaks,
labelled placeholders, checksum-validated cards/IBANs, and no false positives
on code, hashes, ids, dates and other benign memory text.
"""
from __future__ import annotations

import pytest

from mark.security import RedactionPolicy, redact

from _security_corpus import BENIGN, PLANTED, PLANTED_PII


@pytest.mark.parametrize(("text", "leak", "kind"), PLANTED, ids=[p[2] + str(i) for i, p in enumerate(PLANTED)])
def test_default_policy_redacts_planted_secret(text: str, leak: str, kind: str) -> None:
    result = redact(text)
    assert leak not in result.text
    assert f"[SECRET:{kind}]" in result.text
    assert kind in result.kinds


@pytest.mark.parametrize("text", BENIGN)
def test_default_policy_leaves_benign_text_untouched(text: str) -> None:
    result = redact(text)
    assert result.text == text and not result.kinds


@pytest.mark.parametrize(("text", "leak"), PLANTED_PII)
def test_pii_is_opt_in_and_redacted_whole(text: str, leak: str) -> None:
    assert redact(text).text == text  # kept by default: agents need contacts
    out = redact(text, RedactionPolicy(pii=True)).text
    assert leak not in out
    # No partial leaks such as "+234 [PHONE]" or "([PHONE]".
    assert "+234" not in out and "(415" not in out and "+44" not in out


def test_password_value_redacted_whole_not_partially() -> None:
    out = redact("DB password=Hunter2!xyz&3 now").text
    assert out == "DB password=[SECRET:password] now"


def test_placeholders_are_deterministic_and_idempotent() -> None:
    text = PLANTED[3][0] + " and " + PLANTED[0][0]
    once = redact(text).text
    assert redact(text).text == once
    assert redact(once).text == once


def test_luhn_and_iban_checksums_gate_matches() -> None:
    assert "[SECRET:card_number]" not in redact("card 4111 1111 1111 1112").text
    assert "[IBAN]" not in redact("GB82 WEST 1234 5698 7654 33", RedactionPolicy(pii=True)).text


def test_pin_requires_explicit_context() -> None:
    assert redact("door code 4821 and 1234 apples").text == "door code 4821 and 1234 apples"
    assert redact("PIN: 4821").text == "PIN: [SECRET:pin]"


def test_url_credentials_keep_user_and_host() -> None:
    out = redact("postgres://admin:hunter2@db.internal/app").text
    assert out == "postgres://admin:[SECRET:password]@db.internal/app"


def test_custom_patterns_and_disabled_secrets() -> None:
    import re

    policy = RedactionPolicy(extra_patterns=[("employee_badge", re.compile(r"\bBADGE-\d{6}\b"))])
    assert redact("badge BADGE-123456", policy).text == "badge [SECRET:employee_badge]"
    off = RedactionPolicy(secrets=False)
    assert redact(PLANTED[0][0], off).text == PLANTED[0][0]


def test_emails_can_be_enabled_without_other_pii() -> None:
    out = redact("mail ops@kestrel.io or +234 803 555 0192", RedactionPolicy(emails=True)).text
    assert "[EMAIL]" in out and "+234 803 555 0192" in out


def test_legacy_helpers_still_use_shared_patterns() -> None:
    from mark.security import redact_secrets

    out = redact_secrets("The staging Stripe key is sk_" + "test_51HxYzAbCdEfGhIjKlMnOp, mail a@b.io")
    assert "51HxYz" not in out and "[EMAIL]" in out
