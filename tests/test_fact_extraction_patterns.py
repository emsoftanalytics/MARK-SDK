# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 Somtochukwu Owen Ikeanyi
"""PatternFactExtractor: built-in packs on varied phrasings, and the guards
against questions, negations, hypotheticals and plans.

The sentences are written to vary verbs, punctuation, casing, labels and
extra clauses; none is copied from a benchmark.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from mark.intelligence.fact_extraction import FactCandidate, FactExtractor, PatternFactExtractor, RulePack

X = PatternFactExtractor()


def triples(text: str, extractor: PatternFactExtractor = X) -> list[tuple[str, str, str]]:
    return [(c.subject, c.attribute, c.value) for c in extractor.extract(text)]


@pytest.mark.parametrize(("text", "expected"), [
    # changes: attribute from the value, from the "from" part, or between verb and "to"
    ("[2026-01-04] CHANGE: orders-api moved from port 8080 to port 9090.", ("orders-api", "port", "9090")),
    ("search-svc was moved to port 7000 on Tuesday", ("search-svc", "port", "7000")),
    ("The gateway changed its region from us-east-1 to eu-central-1 after the outage.",
     ("gateway", "region", "eu-central-1")),
    ("ledger switched its database to MariaDB because of licensing", ("ledger", "database", "MariaDB")),
    ("Reports-Worker has migrated to the zone b-2.", ("Reports-Worker", "zone", "b-2")),
    ("UPDATE: Kai Moreno transferred to the Billing department, effective immediately.",
     ("Kai Moreno", "department", "Billing")),
    ("Lena Fischer relocated to Porto.", ("Lena Fischer", "location", "Porto")),
    ("cache-tier upgraded to v3.2.1 last night", ("cache-tier", "version", "v3.2.1")),
    ("Meanwhile, frontend renamed to web-shell", ("frontend", "name", "web-shell")),
    ("svc-alpha port changed from 1 to 2", ("svc-alpha", "port", "2")),
    # ownership
    ("[2026-02-01] OWNERSHIP: payments-core is now owned by Ravi Shah.", ("payments-core", "owner", "Ravi Shah")),
    ("Service notify-hub is owned by Team Kestrel, listens on port 5050", ("notify-hub", "owner", "Team Kestrel")),
    ("Ownership of the build farm was transferred to Ines Duarte.", ("build farm", "owner", "Ines Duarte")),
    ("docs-site is maintained by Omar.", ("docs-site", "maintainer", "Omar")),
    # status (values lower-cased; common spellings folded)
    ("TICKET OPS-881 status changed to In-Progress.", ("OPS-881", "status", "in progress")),
    ("The status of INC-12 was updated from open to Resolved on 2026-05-02.", ("INC-12", "status", "resolved")),
    ("bug BUG-7 was marked as done", ("BUG-7", "status", "done")),
    ("Release-9's priority went to high", ("Release-9", "priority", "high")),
    # membership / assignment
    ("HR: Mia Chen joined the Data Platform team, working remotely.", ("Mia Chen", "team", "Data Platform")),
    ("Issue ABC-42 reassigned to Tom Novak.", ("ABC-42", "assignee", "Tom Novak")),
    ("Jo Park was assigned to the Security group", ("Jo Park", "group", "Security")),
    # key/value
    ("atlas port: 4444", ("atlas", "port", "4444")),
    ("mailer.owner = Dana", ("mailer", "owner", "Dana")),
    ("Payroll's owner is Dana Iqbal", ("Payroll", "owner", "Dana Iqbal")),
    ("billing region = eu-west-2;", ("billing", "region", "eu-west-2")),
])
def test_builtin_packs_extract_varied_phrasings(text: str, expected: tuple[str, str, str]) -> None:
    assert expected in triples(text)


@pytest.mark.parametrize("text", [
    # questions
    "Did orders-api move to port 9090?",
    "Has payments-core been moved to port 9?",
    "Is the gateway now owned by Ravi?",
    "orders-api moved to port 9090?",
    # negations
    "orders-api did not move to port 9090.",
    "orders-api never moved to port 9090",
    "payments-core will not be owned by Ravi Shah.",
    "payments-core is no longer owned by Ravi Shah.",
    "OPS-1 status wasn't changed to closed",
    "Kai Moreno didn't transfer to the Billing department.",
    # hypotheticals and plans
    "If orders-api moves to port 9090, update the firewall.",
    "If orders-api moved to port 9090 we would need a new rule.",
    "Assuming payments-core is owned by Ravi, he approves.",
    "orders-api could be moved to port 9090.",
    "orders-api will move to port 9090 next week.",
    "orders-api is scheduled to be moved to port 9090.",
    "We plan to move orders-api to port 9090.",
    # no subject / no attribute
    "It moved to port 9090.",
    "We switched to Slack.",
    "The meeting moved to Thursday because of the holiday and more people",
])
def test_no_facts_from_questions_negations_or_hypotheticals(text: str) -> None:
    assert triples(text) == []


def test_negation_in_one_clause_does_not_block_another() -> None:
    text = "orders-api did not move to port 1; it was moved to port 2. search-svc moved to port 3."
    assert triples(text) == [("search-svc", "port", "3")]


def test_multiple_sentences_and_labels_in_one_text() -> None:
    text = ("[2026-03-01] DEPLOY: web moved to port 81. Afterwards web switched its region to eu-north-1.\n"
            "[2026-03-02] NOTE: ticket T-9 status changed to blocked")
    assert triples(text) == [("web", "port", "81"), ("web", "region", "eu-north-1"), ("T-9", "status", "blocked")]


def test_valid_from_from_inline_dates_and_later_line_stamps() -> None:
    first = datetime(2026, 3, 1, tzinfo=timezone.utc)
    text = ("[2026-03-01] web moved to port 81.\n"
            "[2026-03-04] web moved to port 82.\n"
            "web status changed to degraded on 2026-03-06.")
    found = X.extract(text, occurred_at=first)
    assert [(c.value, c.valid_from.date().isoformat()) for c in found] == [
        ("81", "2026-03-01"), ("82", "2026-03-04"), ("degraded", "2026-03-06"),
    ]
    # Without an occurred_at the first line is left for the caller to date.
    assert X.extract("[2026-03-01] web moved to port 81.")[0].valid_from is None


def test_candidates_are_ordered_and_deduplicated() -> None:
    found = X.extract("db1 moved to port 1. db1 moved to port 2. db1 moved to port 2.")
    assert [c.value for c in found] == ["1", "2"]
    assert all(isinstance(c, FactCandidate) and c.rule == "changes" and 0 < c.confidence <= 1 for c in found)


def test_user_packs_and_pack_selection_by_name() -> None:
    titles = RulePack("titles", patterns=(r"^{S}\s+was\s+promoted\s+to\s+{V}",), attribute="title",
                      value_case="lower", confidence=0.9)
    custom = PatternFactExtractor(packs=[titles, "ownership"])
    assert triples("Ana Ruiz was promoted to Staff Engineer.", custom) == [("Ana Ruiz", "title", "staff engineer")]
    assert triples("x-api is owned by Ana.", custom) == [("x-api", "owner", "Ana")]
    assert triples("x-api moved to port 5.", custom) == []  # "changes" pack not selected
    assert isinstance(custom, FactExtractor)
    with pytest.raises(KeyError):
        PatternFactExtractor(packs=["no-such-pack"])


def test_value_map_and_custom_attribute_from_verb() -> None:
    pack = RulePack("moods", patterns=(r"^{S}\s+(?P<verb>felt|seemed)\s+{V}",),
                    verb_attributes={"felt": "mood", "seemed": "appearance"}, value_map={"meh": "neutral"})
    extractor = PatternFactExtractor(packs=[pack])
    assert triples("Rex felt meh. Rex seemed tired", extractor) == [("Rex", "mood", "neutral"),
                                                                    ("Rex", "appearance", "tired")]


def test_long_or_empty_values_are_rejected() -> None:
    clause = "orders-api moved to " + " ".join(f"word{i}" for i in range(15))
    assert triples(clause) == []
    assert triples("") == [] and triples("   \n  ") == []
