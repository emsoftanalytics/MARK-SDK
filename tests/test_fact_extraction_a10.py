# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 MARK Contributors
"""Extraction fixes from the 0.2.0a10 re-tests (dev M-d, breadth #7, agent N3).

Shapes (documented in the fact_extraction package):
  - preferences: ``user.likes = tea without sugar``, ``user.prefers = tea over coffee``
    (the verb is the attribute, the thing is the value, comparisons kept; multi-valued);
  - "I love my dog Rex" -> ``user.dog = Rex``; the placeholder never leaks;
  - "I used to work at X" -> a past value (``historical=True``), "from A to B"
    -> A as the ``previous`` value before B; lists -> one multi-valued candidate each.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from mark.intelligence.fact_extraction import PatternFactExtractor

NOW = datetime(2026, 10, 18, 12, 0, tzinfo=timezone.utc)
X = PatternFactExtractor()


def _triples(text: str, **kw) -> set[tuple[str, str, str]]:
    return {(c.subject, c.attribute, c.value) for c in X.extract(text, occurred_at=NOW, **kw)}


def _changes(text: str) -> list:
    return X.extract_changes(text, occurred_at=NOW)


@pytest.mark.parametrize(("text", "expected"), [
    ("I like my coffee black.", ("user", "likes", "coffee black")),
    ("I prefer trains over planes.", ("user", "prefers", "trains over planes")),
    ("I prefer tea to coffee.", ("user", "prefers", "tea to coffee")),
    ("I really enjoy board games.", ("user", "likes", "board games")),
    ("I can't stand loud music.", ("user", "hates", "loud music")),
    ("Femi loves jazz.", ("Femi", "loves", "jazz")),
    ("I love my cat Miso.", ("user", "cat", "Miso")),
])
def test_preferences_use_the_verb_as_attribute(text: str, expected: tuple[str, str, str]) -> None:
    found = _triples(text)
    assert expected in found
    assert not any("MARKUSER" in part for triple in found for part in triple)


@pytest.mark.parametrize("text", ["I love my job.", "I love you.", "I like it when it rains.", "I love my dog."])
def test_vague_preferences_record_nothing(text: str) -> None:
    assert _triples(text) == set()


def test_preferences_are_multi_valued() -> None:
    assert all(c.multi for c in _changes("I like jazz and I like tea."))


def test_drinks_and_stopping() -> None:
    assert ("user", "drinks", "green tea") in _triples("I drink green tea every morning.")
    gone = [c for c in _changes("I no longer drink soda.") if c.retract]
    assert [(c.subject, c.attribute, c.value) for c in gone] == [("user", "drinks", "soda")]
    stopped = [c for c in _changes("Tunde stopped drinking beer.") if c.retract]
    assert [(c.subject, c.attribute, c.value) for c in stopped] == [("Tunde", "drinks", "beer")]


def test_used_to_is_a_past_value() -> None:
    [past] = _changes("I used to work at Shell.")
    assert (past.subject, past.attribute, past.value) == ("user", "employer", "Shell")
    assert past.retract and past.historical


def test_from_a_to_b_records_a_as_the_previous_value() -> None:
    found = [(c.subject, c.attribute, c.value, c.previous) for c in _changes("I switched from Vim to Neovim.")]
    assert found == [("user", "editor", "Vim", True), ("user", "editor", "Neovim", False)]
    texts = [(c.subject, c.attribute, c.value, c.previous)
             for c in _changes("The release codename changed from Bison to Heron.")]
    assert texts == [("release", "codename", "Bison", True), ("release", "codename", "Heron", False)]
    assert _triples("The meeting moved from Monday to Tuesday.") == set()
    assert all(not c.previous for c in X.extract("svc-a moved from port 1 to port 2"))


@pytest.mark.parametrize(("text", "expected"), [
    ("The test command is `make test`.", ("project", "test command", "make test")),
    ("The format command is `ruff format .`", ("project", "format command", "ruff format .")),
    ("kestrel has 4 replicas.", ("kestrel", "replicas", "4")),
    ("I have three kids.", ("user", "kids", "3")),
    ("ledger-api uses pnpm.", ("ledger-api", "package manager", "pnpm")),
    ("I use Arch Linux.", ("user", "os", "Arch Linux")),
    ("Status of HAL-9: closed", ("HAL-9", "status", "closed")),
    ("TODO HAL-12: rotate certs", ("HAL-12", "status", "todo")),
    ("kestrel config path: /etc/kestrel.toml", ("kestrel", "config file", "/etc/kestrel.toml")),
])
def test_new_phrasings(text: str, expected: tuple[str, str, str]) -> None:
    assert expected in _triples(text)


@pytest.mark.parametrize("text", [
    "The team has 3 days left.", "It has 4 replicas.", "The dashboard uses a lot of memory.",
    "atlas-core: 8080", "kestrel uses it for caching.",
])
def test_new_phrasings_negatives(text: str) -> None:
    assert _triples(text) == set()


def test_config_blocks_and_port_lists() -> None:
    found = _triples("MAX_UPLOAD_MB: 250\nSESSION_TIMEOUT_MIN=45")
    assert found == {("config", "MAX_UPLOAD_MB", "250"), ("config", "SESSION_TIMEOUT_MIN", "45")}
    ports = _triples("Service ports:\n- kestrel-api: 7100\n- kestrel-web: 7101")
    assert ports == {("kestrel-api", "port", "7100"), ("kestrel-web", "port", "7101")}
    assert _triples("TODO: fix it\nNOTE: later") == set()


def test_lists_become_multi_valued_candidates() -> None:
    found = [(c.subject, c.attribute, c.value, c.multi) for c in _changes("I speak Igbo, Twi and Dutch.")]
    assert found == [("user", "language", v, True) for v in ("Igbo", "Twi", "Dutch")]
    emails = {(c.attribute, c.value) for c in _changes("Ngozi's emails are n@a.io and n@b.io.")}
    assert emails == {("email", "n@a.io"), ("email", "n@b.io")}
    assert _triples("My kids are 7 and 10.") == set()


def test_future_dates_schedule_the_value() -> None:
    [fact] = X.extract("Starting 2027-05-01, Ines leads the data team.", occurred_at=NOW)
    assert (fact.subject, fact.attribute, fact.value) == ("data team", "lead", "Ines")
    assert fact.valid_from == datetime(2027, 5, 1, tzinfo=timezone.utc)
    [moved] = X.extract("From 2027-02-01 the platform team moves to Nairobi.", occurred_at=NOW)
    assert (moved.attribute, moved.value, moved.valid_from.date().isoformat()) == ("location", "Nairobi", "2027-02-01")
    assert X.extract("The platform team moves to Nairobi.", occurred_at=NOW) == []


def test_month_names_date_past_moves() -> None:
    [sept] = X.extract("I moved to Faro in August.", occurred_at=NOW)
    assert sept.valid_from == datetime(2026, 8, 1, tzinfo=timezone.utc)
    [dec] = X.extract("I moved to Oslo in December.", occurred_at=NOW)
    assert dec.valid_from == datetime(2025, 12, 1, tzinfo=timezone.utc)


def test_known_subject_keeps_the_rest_as_attribute() -> None:
    text = "kestrel is a deploy bot. The kestrel plugin entry-point group is kestrel.plugins."
    assert ("kestrel", "plugin entry-point group", "kestrel.plugins") in _triples(text)
    assert ("kestrel", "default database", "k.db") in _triples("kestrel uses SQLite. kestrel default database: k.db")
    # People are never split ("Mei Lin" is one name even when "Mei" is known).
    assert ("Mei Lin", "location", "Oslo") in _triples("Mei lives in Bergen. Mei Lin lives in Oslo.")


def test_possessive_after_a_relation_names_the_person() -> None:
    from mark.intelligence.fact_extraction import PatternFactExtractor

    facts = {(c.subject, c.attribute, c.value)
             for c in PatternFactExtractor().extract("My friend Joao's phone number is 912 345 678.")}
    assert ("user", "friend", "Joao") in facts and ("Joao", "phone", "912 345 678") in facts
    assert not any(value.endswith("'s") for _, _, value in facts)


@pytest.mark.parametrize("sentence, expected", [
    ("I changed jobs: I'm now a nurse manager at Hospital de Braga.",
     {("user", "occupation", "nurse manager"), ("user", "employer", "Hospital de Braga")}),
    ("I adopted a cat named Pixel.", {("user", "cat", "Pixel")}),
    ("I got a dog named after my grandfather.", set()),
    ("The film is called Dune.", set()),
    ("She is now a director at the bank.", set()),
])
def test_new_pet_and_new_job_sentences(sentence: str, expected: set) -> None:
    from mark.intelligence.fact_extraction import PatternFactExtractor

    found = {(c.subject, c.attribute, c.value) for c in PatternFactExtractor().extract(sentence)}
    assert found == expected
