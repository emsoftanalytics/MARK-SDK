# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 Somtochukwu Owen Ikeanyi
"""PatternFactExtractor coverage: first-person facts, personal details, developer phrasing.

First-person statements ("I moved to Porto", "My sister is Ines") become
facts about a configurable user subject; pronouns and possessives are never
subjects; titles stay part of a name ("Dr. Bello"). Developer notes ("test
command is ...", "switched from A to B", "HAL-1 ... status: resolved") and
prices/measurements keep their units.
"""
from __future__ import annotations

import pytest

from mark.intelligence.fact_extraction import PatternFactExtractor

X = PatternFactExtractor()


def triples(text: str, extractor: PatternFactExtractor = X) -> list[tuple[str, str, str]]:
    return [(c.subject, c.attribute, c.value) for c in extractor.extract(text)]


@pytest.mark.parametrize(("text", "expected"), [
    ("I moved to Porto.", ("user", "location", "Porto")),
    ("I live in Lagos and work at Seefar.", ("user", "location", "Lagos")),
    ("I live in Lagos and work at Seefar.", ("user", "employer", "Seefar")),
    ("My city is Porto now.", ("user", "city", "Porto")),
    ("My name is Somto.", ("user", "name", "Somto")),
    ("My sister is Ines.", ("user", "sister", "Ines")),
    ("I'm vegetarian.", ("user", "diet", "vegetarian")),
    ("I no longer eat meat.", ("user", "diet", "no meat")),
    ("My birthday is 14 March.", ("user", "birthday", "14 March")),
    ("I am 182 cm tall.", ("user", "height", "182 cm")),
    ("I prefer dark mode.", ("user", "prefers", "dark mode")),
    ("I love jazz.", ("user", "loves", "jazz")),
    ("I'm a software engineer.", ("user", "occupation", "software engineer")),
    ("Priya manages me.", ("user", "manager", "Priya")),
])
def test_first_person_statements_become_user_facts(text: str, expected: tuple[str, str, str]) -> None:
    assert expected in triples(text)


def test_user_subject_is_configurable() -> None:
    assert triples("I moved to Porto.", PatternFactExtractor(user_subject="Somto")) == [
        ("Somto", "location", "Porto")]


@pytest.mark.parametrize("text", [
    "My plan is to ship on Friday.",
    "I think it broke.",
    "I am tired today.",
    "I'd like to move to Berlin one day.",
    "Do I still live in Porto?",
    "He moved to Leeds.",
])
def test_vague_first_person_and_unresolved_pronouns_yield_nothing(text: str) -> None:
    assert triples(text) == []


def test_pronouns_and_possessives_are_never_subjects() -> None:
    found = triples("Kemi's email is kemi@a.io; her other email is kemi@b.io.")
    assert found == [("Kemi", "email", "kemi@a.io")]
    assert all(s.lower() not in {"my", "her other", "i'm"} for s, _, _ in triples("My email is a@b.io"))


def test_titles_stay_part_of_the_name() -> None:
    assert triples("Dr. Bello's phone is +234 803 555 0192.") == [("Dr. Bello", "phone", "+234 803 555 0192")]


def test_trivially_safe_pronoun_resolution() -> None:
    found = triples("Grace was born on 14 March 1990. She moved to Leeds.")
    assert found == [("Grace", "birthday", "14 March 1990"), ("Grace", "location", "Leeds")]
    # Two people named before the pronoun: ambiguous, so it is skipped.
    assert ("Ana", "location", "Leeds") not in triples("Ana met Bo Kim. She moved to Leeds.")


@pytest.mark.parametrize(("text", "expected"), [
    ("Priya manages Tomasz.", ("Tomasz", "manager", "Priya")),
    ("Grace was born on 14 March 1990.", ("Grace", "birthday", "14 March 1990")),
    ("User prefers dark roast coffee.", ("user", "prefers", "dark roast coffee")),
    ("The Growth plan costs $79 per month.", ("Growth plan", "price", "$79 per month")),
    ("The kitchen tiles cost 42 euros per square metre.", ("kitchen tiles", "price", "42 euros per square metre")),
    ("The renovation budget is capped at 38,000 euros.", ("renovation", "budget", "38,000 euros")),
    ("The crate weighs 12.5 kg.", ("crate", "weight", "12.5 kg")),
    ("Mei Lin works as a pharmacist.", ("Mei Lin", "occupation", "pharmacist")),
    ("atlas-core uses port 9999.", ("atlas-core", "port", "9999")),
])
def test_people_prices_and_measurements(text: str, expected: tuple[str, str, str]) -> None:
    assert expected in triples(text)


@pytest.mark.parametrize("text", [
    "The scheduler manages retries.",
    "The meeting moved to Thursday.",
    "A Maria mudou-se para o Porto.",
    "Der Dienst läuft auf Port 8080.",
])
def test_vague_or_foreign_phrasing_is_ignored(text: str) -> None:
    assert triples(text) == []


@pytest.mark.parametrize(("text", "expected"), [
    ("[2026-10-01] tasky storage backend is JSON.", ("tasky", "storage backend", "JSON")),
    ("[2026-10-05] tasky storage backend moved from JSON to SQLite.", ("tasky", "storage backend", "SQLite")),
    ("[2026-10-05] tasky test command is pytest -q.", ("tasky", "test command", "pytest -q")),
    ("[2026-10-09] tasky test command changed from pytest -q to pytest -x -q.",
     ("tasky", "test command", "pytest -x -q")),
    ("[2026-10-09] HAL-1: plugin hook ordering bug. status: resolved", ("HAL-1", "status", "resolved")),
    ("[2026-10-09] tasky config file is ~/.config/tasky/config.toml", ("tasky", "config file", "~/.config/tasky/config.toml")),
    ("checkout-web switched from npm to pnpm.", ("checkout-web", "package manager", "pnpm")),
    ("We migrated tasky from MySQL to SQLite.", ("tasky", "datastore", "SQLite")),
    ("TODO: HAL-9 add retries", ("HAL-9", "status", "todo")),
    ("DONE: HAL-9", ("HAL-9", "status", "done")),
])
def test_developer_phrasing(text: str, expected: tuple[str, str, str]) -> None:
    assert expected in triples(text)


def test_line_labels_tag_candidates_with_their_kind() -> None:
    (candidate,) = X.extract("DECISION: tasky storage backend is SQLite.")
    assert candidate.tags == ("decision",)
    assert X.extract("BUG: tasky crashes on empty input") == []


def test_llm_replies_map_first_person_subjects_to_the_user() -> None:
    from mark.intelligence.fact_extraction.llm import parse_fact_reply

    reply = '{"facts": [{"subject": "I", "attribute": "city", "value": "Porto"}, {"subject": "she", "attribute": "x", "value": "y"}]}'
    assert [(c.subject, c.attribute, c.value) for c in parse_fact_reply(reply, user_subject="me")] == [
        ("me", "city", "Porto")]


@pytest.mark.parametrize(("text", "expected"), [
    ("I'm vegan - no dairy either.", ("user", "diet", "vegan")),
    ("My brother Femi is a pilot.", ("user", "brother", "Femi")),
    ("My brother Femi is a pilot.", ("Femi", "occupation", "pilot")),
    ("I moved from Lisbon to Porto.", ("user", "location", "Porto")),
    ("I am 182 cm tall and weigh 78 kg.", ("user", "weight", "78 kg")),
    ("My dentist is Dr. Silva at the Boavista clinic.", ("user", "dentist", "Dr. Silva")),
    ("My monthly rent is 1,350 EUR, paid on the 1st.", ("user", "monthly rent", "1,350 EUR")),
])
def test_trailing_clauses_appositions_and_titles(text: str, expected: tuple[str, str, str]) -> None:
    assert expected in triples(text)
