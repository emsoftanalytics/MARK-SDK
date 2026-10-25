# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 Somtochukwu Owen Ikeanyi
"""PatternFactExtractor on present-tense descriptions and multi-clause lines.

"X runs on port 8080" or "dana owns X" state a current value just as
"X moved to port 8080" does. These tests cover the generic verb families
(placement, location, ownership, reporting lines, datastores, assignment),
subject carry-over between clauses, and the precision guards that keep
vague phrasing ("uses Slack for chat", "runs in production") out.
"""
from __future__ import annotations

import pytest

from mark.intelligence.fact_extraction import PatternFactExtractor

X = PatternFactExtractor()


def triples(text: str) -> list[tuple[str, str, str]]:
    return [(c.subject, c.attribute, c.value) for c in X.extract(text)]


@pytest.mark.parametrize(("text", "expected"), [
    ("atlas-core runs on port 6430", ("atlas-core", "port", "6430")),
    ("atlas-core listens on port 6430.", ("atlas-core", "port", "6430")),
    ("The ingest-svc is listening on 9001", ("ingest-svc", "port", "9001")),
    ("atlas-core runs in eu-west-1", ("atlas-core", "region", "eu-west-1")),
    ("billing is deployed in region us-central1", ("billing", "region", "us-central1")),
    ("search-api is hosted in the Frankfurt datacenter", ("search-api", "datacenter", "Frankfurt")),
    ("dana now owns atlas-core", ("atlas-core", "owner", "dana")),
    ("Team Kestrel maintains the docs-site service.", ("docs-site", "maintainer", "Team Kestrel")),
    ("atlas-core is now owned by dana", ("atlas-core", "owner", "dana")),
    ("Priya Nair is the owner of ledger-v2", ("ledger-v2", "owner", "Priya Nair")),
    ("atlas-core uses Postgres", ("atlas-core", "datastore", "Postgres")),
    ("orders-db now runs on MySQL 8.0, per the migration plan", ("orders-db", "datastore", "MySQL 8.0")),
    ("the cart service uses Valkey as its cache", ("cart service", "cache", "Valkey")),
    ("Mei Lin works from Berlin", ("Mei Lin", "location", "Berlin")),
    ("Mei Lin is based in Berlin.", ("Mei Lin", "location", "Berlin")),
    ("Tomás Ruiz lives in São Paulo", ("Tomás Ruiz", "location", "São Paulo")),
    ("Ana Costa reports to Dana Iqbal", ("Ana Costa", "manager", "Dana Iqbal")),
    ("OPS-17 is assigned to Bo Kim", ("OPS-17", "assignee", "Bo Kim")),
    ("Bo Kim is assigned to OPS-17", ("OPS-17", "assignee", "Bo Kim")),
    ("Kenji Sato leads the Mobile squad", ("Mobile squad", "lead", "Kenji Sato")),
    ("Our billing-api is now listening on port 8443", ("billing-api", "port", "8443")),
])
def test_present_tense_descriptions(text: str, expected: tuple[str, str, str]) -> None:
    assert expected in triples(text)


def test_first_mention_line_with_several_clauses() -> None:
    text = "[2026-01-02] INVENTORY: service atlas-core is owned by Dana, listens on port 6430, runs in eu-west-1, uses PostgreSQL 15."
    assert triples(text) == [("atlas-core", "owner", "Dana"), ("atlas-core", "port", "6430"),
                             ("atlas-core", "region", "eu-west-1"), ("atlas-core", "datastore", "PostgreSQL 15")]


def test_subject_carries_over_and_clauses_and_participles() -> None:
    assert triples("Mei Lin is based in Lisbon and reports to Ravi Shah.") == [
        ("Mei Lin", "location", "Lisbon"), ("Mei Lin", "manager", "Ravi Shah")]
    assert ("Ola Berg", "location", "Oslo") in triples("HR: Ola Berg joined the Data team, working from Oslo.")
    assert ("INC-42", "assignee", "Lee Park") in triples(
        "Ticket INC-42 opened (high): login fails, assigned to Lee Park.")


def test_carry_over_stops_at_a_new_subject_and_at_sentence_end() -> None:
    assert triples("api-a runs on port 1, api-b runs on port 2") == [("api-a", "port", "1"), ("api-b", "port", "2")]
    assert triples("api-a runs on port 1. Listens on port 2.") == [("api-a", "port", "1")]


@pytest.mark.parametrize("text", [
    "the team uses Slack for chat",
    "atlas-core runs in production",
    "atlas-core runs on Kubernetes",
    "Mei Lin works in Sales",
    "Everyone owns quality",
    "This leads to timeouts",
    "atlas-core uses Oracle Cloud",
    "Who owns atlas-core?",
    "Does atlas-core run on port 6430?",
    "atlas-core no longer listens on port 6430",
    "atlas-core doesn't use Postgres",
    "dana will own atlas-core next quarter",
    "If atlas-core runs in eu-west-1 we pay more",
    "[REDACTED] owns atlas-core",
])
def test_vague_negated_and_hypothetical_phrasing_is_ignored(text: str) -> None:
    assert triples(text) == []


def test_negated_first_clause_still_carries_its_subject() -> None:
    assert triples("atlas-core does not use Redis, uses Postgres") == [("atlas-core", "datastore", "Postgres")]


def test_bare_article_is_never_a_subject() -> None:
    assert triples("The office is located in Dublin") == [("office", "location", "Dublin")]
