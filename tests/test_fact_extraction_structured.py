# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 MARK Contributors
"""Structured content, retractions/corrections and list steps in PatternFactExtractor.

key: value blocks, k=v pairs, JSON objects and YAML maps take their subject
from a ``name``/``id``/``service`` field or a preceding header. Negated
statements ("no longer runs on port 8080") are retractions, reported only by
``extract_changes``; corrections carry ``correction=True``. Number words and
ordinals are normalised for step lists.
"""
from __future__ import annotations

import pytest

from mark.intelligence.fact_extraction import PatternFactExtractor
from mark.intelligence.fact_extraction.numbers import normalise_step_refs, parse_number

X = PatternFactExtractor()


def triples(text: str) -> list[tuple[str, str, str]]:
    return [(c.subject, c.attribute, c.value) for c in X.extract(text)]


@pytest.mark.parametrize(("text", "expected"), [
    ('{"service": "billing-api", "port": 9090}', [("billing-api", "port", "9090")]),
    ("billing-api:\n  port: 9091", [("billing-api", "port", "9091")]),
    ("port=8080 for atlas-core", [("atlas-core", "port", "8080")]),
    ("id: OPS-7\nstatus: blocked\nassignee: Lee Park", [("OPS-7", "status", "blocked"), ("OPS-7", "assignee", "Lee Park")]),
    ("# ledger-sync\n- owner: Team Otter\n- test_command: make test",
     [("ledger-sync", "owner", "Team Otter"), ("ledger-sync", "test command", "make test")]),
    ('Config for orders-api: {"region": "eu-west-1", "replicas": 3}',
     [("orders-api", "region", "eu-west-1"), ("orders-api", "replicas", "3")]),
])
def test_structured_records(text: str, expected: list[tuple[str, str, str]]) -> None:
    assert triples(text) == expected


@pytest.mark.parametrize("text", [
    "port: 8080",                                  # no subject anywhere
    "Note: call the vendor\nSummary: all good",    # prose labels, not fields
    '{"port": 8080}',                              # JSON without a subject
    "Usage: mark recall QUERY",
])
def test_structured_without_subject_yields_nothing(text: str) -> None:
    assert triples(text) == []


def test_negations_are_retractions_only_in_extract_changes() -> None:
    assert X.extract("atlas-core no longer runs on port 8080.") == []
    (change,) = X.extract_changes("atlas-core no longer runs on port 8080.")
    assert (change.subject, change.attribute, change.value, change.retract) == ("atlas-core", "port", "8080", True)
    (moved,) = X.extract_changes("I don't live in Lagos anymore.")
    assert (moved.subject, moved.attribute, moved.value, moved.retract) == ("user", "location", "Lagos", True)
    assert X.extract_changes("Is atlas-core no longer on port 8080?") == []


def test_corrections_are_flagged_and_name_the_wrong_value() -> None:
    changes = X.extract_changes("Correction: atlas-core actually runs on port 9443, not 9444.")
    assert [(c.attribute, c.value, c.correction, c.retract) for c in changes] == [
        ("port", "9443", True, False), ("port", "9444", False, True)]
    (fixed,) = X.extract("Correction: my favourite city is actually Lisbon")
    assert (fixed.subject, fixed.attribute, fixed.value, fixed.correction) == ("user", "favourite city", "Lisbon", True)


def test_numbered_steps_become_step_facts() -> None:
    found = triples("Deploy procedure: 1. Freeze merges. 2. Tag the release. 3. Build images.")
    assert found == [("Deploy procedure", "step 1", "Freeze merges"), ("Deploy procedure", "step 2", "Tag the release"),
                     ("Deploy procedure", "step 3", "Build images")]
    assert triples("Restore runbook:\nStep one: stop writes\nStep two: copy the snapshot") == [
        ("Restore runbook", "step 1", "stop writes"), ("Restore runbook", "step 2", "copy the snapshot")]
    assert triples("The fifth step of the release checklist is to notify support.") == [
        ("release checklist", "step 5", "notify support")]


def test_number_words_and_ordinals() -> None:
    assert parse_number("four") == 4 and parse_number("fourth") == 4 and parse_number("twenty-one") == 21
    assert parse_number("4th") == 4 and parse_number("banana") is None
    assert normalise_step_refs("what is step four of the deploy?") == "what is step 4 of the deploy?"
    assert normalise_step_refs("the fourth step") == "the step 4"


def test_markdown_tables_are_records() -> None:
    assert triples("| service | port | owner |\n|---|---|---|\n| ledger | 7000 | Dana |") == [
        ("ledger", "port", "7000"), ("ledger", "owner", "Dana")]
    assert triples("Monthly budget:\n| item | amount |\n|---|---|\n| rent | 900 EUR |") == [
        ("Monthly budget", "rent", "900 EUR")]
    assert triples("| item | amount |\n|---|---|\n| rent | 900 EUR |") == []  # untitled key/value table
