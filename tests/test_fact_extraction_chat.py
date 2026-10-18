# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 MARK Contributors
"""Extraction on chat-stored text and everyday personal phrasing (small-model re-test).

MarkChat and the middleware store turns with a provenance prefix ("User
said: ..."). User text is read as if the prefix were not there; assistant,
agent and tool text is never a source of facts (a model's guess is not the
user's statement).
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from mark.intelligence.fact_extraction import PatternFactExtractor

NOW = datetime(2026, 10, 18, 12, 0, tzinfo=timezone.utc)
X = PatternFactExtractor()


def _triples(text: str) -> set[tuple[str, str, str]]:
    return {(c.subject, c.attribute, c.value) for c in X.extract(text, occurred_at=NOW)}


@pytest.mark.parametrize("text", ["User said: I live in Lagos.", "User: I live in Lagos.", "Human: I live in Lagos"])
def test_user_turns_are_read_without_their_prefix(text: str) -> None:
    assert _triples(text) == {("user", "location", "Lagos")}


@pytest.mark.parametrize("text", [
    "Assistant replied: You live in Lagos and Dana owns atlas-core.",
    "Assistant said: atlas-core runs on port 8080.",
    "Agent reasoning: atlas-core runs on port 8080.",
    "Agent completed task. Response: atlas-core runs on port 8080.",
    "Tool [search] returned: atlas-core runs on port 8080.",
    "Assistant: Noted, you prefer tea over coffee.",
    "AI: ledger-sync runs on port 7000.",
])
def test_assistant_agent_and_tool_text_is_never_extracted(text: str) -> None:
    assert X.extract_changes(text, occurred_at=NOW) == []


def test_a_transcript_keeps_only_user_lines() -> None:
    text = ("User said: I moved to Porto.\nAssistant replied: Great, so you live in Porto now and like tea.\n"
            "It is a nice city.\nUser said: My dog is called Bruno.")
    assert _triples(text) == {("user", "location", "Porto"), ("user", "dog", "Bruno")}


@pytest.mark.parametrize(("text", "expected"), [
    ("I started a new job at Stripe.", ("user", "employer", "Stripe")),
    ("I just started working at Globex.", ("user", "employer", "Globex")),
    ("Femi started a new job at Paystack.", ("Femi", "employer", "Paystack")),
    ("I started a new job as a product manager.", ("user", "occupation", "product manager")),
    ("Kemi changed her number, the new one is +234 803 555 0100.", ("Kemi", "phone", "+234 803 555 0100")),
    ("I changed my number; the new one is 0803 555 0111.", ("user", "phone", "0803 555 0111")),
    ("Bo changed his email, the new one is bo@new.io.", ("Bo", "email", "bo@new.io")),
    ("The dentist appointment was moved to 14 November at 10:00.",
     ("dentist appointment", "date", "14 November at 10:00")),
    ("My appointment was rescheduled to Friday.", ("appointment", "date", "Friday")),
    ("The kickoff meeting has been postponed to 2026-11-03.", ("kickoff meeting", "date", "2026-11-03")),
    ("Step 7 is the last step of the deploy procedure.", ("deploy procedure", "step count", "7")),
    ("My cat is called Miso.", ("user", "cat", "Miso")),
])
def test_everyday_personal_phrasing(text: str, expected: tuple[str, str, str]) -> None:
    assert expected in _triples(text)


def test_last_step_uses_the_procedure_listed_before() -> None:
    text = "Deploy procedure:\n1. Freeze merges\n2. Tag the release\n3. Build images\nStep 3 is the last one."
    assert ("Deploy procedure", "step count", "3") in _triples(text)


@pytest.mark.parametrize("text", [
    "Step 5 is the last one.",             # which procedure? not in the text
    "I started a new job.",                # where? not in the text
    "Kemi changed her number.",            # the new number is not given
    "The appointment was moved to the other room.",
    "The meeting moved to Thursday.",      # active "moved" stays a non-fact (meetings move rooms too)
])
def test_everyday_phrasing_without_the_value_records_nothing(text: str) -> None:
    assert _triples(text) == set()
