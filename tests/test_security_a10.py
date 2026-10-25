# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 Somtochukwu Owen Ikeanyi
"""Redaction and injection misses found by the 0.2.0a10 breadth re-test.

Recall on the new planted phrasings (prose passwords, "pw:"/"pass:", long
keys after "API key"/"token"/"secret", seed phrases, national-format phones
under pii=True, non-English and indirect injections) and zero false
positives on the near-miss corpora next to them.
"""
from __future__ import annotations

import pytest

from mark.security import RedactionPolicy, redact, screen_injection

from _security_corpus import (BENIGN, BENIGN_A10, BENIGN_PROMPT_TALK, BENIGN_PROMPT_TALK_A10, INJECTIONS_A10,
                              PLANTED_A10, PLANTED_PII_A10)


@pytest.mark.parametrize(("text", "leak", "kind"), PLANTED_A10, ids=[p[2] + str(i) for i, p in enumerate(PLANTED_A10)])
def test_new_secret_phrasings_are_redacted(text: str, leak: str, kind: str) -> None:
    result = redact(text)
    assert leak not in result.text
    assert kind in result.kinds


@pytest.mark.parametrize(("text", "leak"), PLANTED_PII_A10)
def test_national_phone_numbers_are_redacted_under_pii(text: str, leak: str) -> None:
    assert redact(text).text == text
    assert leak not in redact(text, RedactionPolicy(pii=True)).text


@pytest.mark.parametrize("text", BENIGN_A10 + BENIGN)
def test_near_misses_stay_untouched(text: str) -> None:
    assert redact(text).text == text


@pytest.mark.parametrize("text", BENIGN_A10)
def test_near_misses_stay_untouched_under_pii(text: str) -> None:
    assert redact(text, RedactionPolicy(pii=True)).text == text


@pytest.mark.parametrize(("text", "reason"), INJECTIONS_A10)
def test_multilingual_and_indirect_injections_are_flagged(text: str, reason: str) -> None:
    report = screen_injection(text)
    assert report.flagged and reason in report.reasons


@pytest.mark.parametrize("text", BENIGN_PROMPT_TALK_A10 + BENIGN_PROMPT_TALK)
def test_benign_prompt_talk_is_not_flagged(text: str) -> None:
    assert not screen_injection(text).flagged
