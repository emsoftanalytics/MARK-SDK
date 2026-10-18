# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 MARK Contributors
"""G4 ranking: out-of-vocabulary exact hits, unnamed supersession and conflicts,
dates named in questions, gap noise, and anchors for small-model chat text."""
from __future__ import annotations

import json
import math
import zlib
from datetime import datetime, timedelta, timezone

import pytest

from mark.embeddings import EmbeddingProvider, HashEmbeddingProvider
from mark.intelligence.query_anchors import name_mentioned, query_anchors
from mark.memory.runtime import MarkRuntime
from mark.types import GapReport, GapSeverity

DISTRACTORS = [
    "The cafeteria serves soup on Mondays.", "Backups run every night at two.",
    "The printer on the second floor jams.", "Our landlord reviewed the release notes.",
    "The design system documented the logo refresh.", "Sofia benchmarked the onboarding flow.",
]


@pytest.fixture()
def mem():
    runtime = MarkRuntime.local(embedder=HashEmbeddingProvider())
    yield runtime.memory("g4")
    runtime.shutdown()


class BlurEmbedder(EmbeddingProvider):
    """A 'semantic' embedder that barely sees emoji, CJK or ids: every text shares one
    strong common direction plus weak character-trigram noise."""

    semantic = True

    def embed(self, text: str) -> list[float]:
        vector = [1.0] + [0.0] * 31
        lowered = text.lower()
        for i in range(max(0, len(lowered) - 2)):
            gram = lowered[i:i + 3]
            if gram.isascii() and gram.strip():
                vector[1 + zlib.crc32(gram.encode()) % 31] += 0.15
        norm = math.sqrt(sum(v * v for v in vector))
        return [v / norm for v in vector]

    @property
    def dim(self) -> int:
        return 32


# ── 5. Exact lexical hit vs vector under semantic embedders ─────────────────

@pytest.mark.parametrize("query, needle", [
    ("🦉", "Team mascot is the 🦉 owl named Hooty, chosen by the whole office after a long vote."),
    ("浦东", "我们的上海仓库下个月将搬到浦东新区。 Warehouse move planned by the logistics team in spring."),
    ("❌", "Deploy status: ✅ staging, ❌ prod, reported by the release manager after the nightly run."),
    ("QX-7781", "Build agent QX-7781 has a failing disk and was drained by the infrastructure team."),
])
def test_sole_exact_lexical_hit_wins_under_a_semantic_embedder(query: str, needle: str) -> None:
    runtime = MarkRuntime.local(embedder=BlurEmbedder())
    try:
        mem = runtime.memory("oov")
        for i in range(12):
            mem.remember(f"Note {i}.")
        mem.remember(needle)
        assert mem.recall(query).fragments[0].content == needle
    finally:
        runtime.shutdown()


# ── 6. Unnamed supersession and free-text conflicts ──────────────────────────

def _fill(mem) -> None:
    for text in DISTRACTORS:
        mem.remember(text)


def test_moved_to_without_from_demotes_the_old_city(mem) -> None:
    _fill(mem)
    mem.remember("I live in Lagos.", occurred_at="2025-01-01")
    mem.remember("I moved to Porto last month.", occurred_at="2026-09-20")
    result = mem.recall("Where do I live?")
    assert "Porto" in result.fragments[0].content
    assert "outdated" in result.as_context()
    assert "Lagos" in mem.recall("Where did I live before?").fragments[0].content


def test_now_live_in_supersedes_the_older_statement(mem) -> None:
    _fill(mem)
    mem.remember("I live in Porto and work as a nurse.", occurred_at="2026-01-01")
    mem.remember("These days I now live in Braga.", occurred_at="2026-08-01")
    assert "Braga" in mem.recall("Which city do I live in?").fragments[0].content


def test_changed_number_and_moved_appointment(mem) -> None:
    _fill(mem)
    mem.remember("My friend Joao's phone number is +351 912 345 678.", occurred_at="2026-10-01")
    mem.remember("My locker number at the gym is 4471.", occurred_at="2026-10-02")
    mem.remember("Joao changed his number, the new one is +351 934 000 111.", occurred_at="2026-10-05")
    mem.remember("Biscuit has a vet appointment on 14 October at 10:30 with Dr. Lemos.", occurred_at="2026-10-01")
    mem.remember("Biscuit's vet appointment moved to 21 October, same time.", occurred_at="2026-10-04")
    assert "934" in mem.recall("What's Joao's phone number?").fragments[0].content
    assert "912" in mem.recall("What was Joao's old phone number?").fragments[0].content
    assert "21 October" in mem.recall("When is Biscuit's vet appointment?").fragments[0].content
    assert "4471" in mem.recall("What's my gym locker number?").fragments[0].content


def test_conflicting_statements_put_the_newer_first_and_are_flagged(mem) -> None:
    _fill(mem)
    mem.remember("Grace's birthday is 14 March.", occurred_at="2026-01-01")
    mem.remember("Grace's birthday is 21 March.", occurred_at="2026-02-01")
    result = mem.recall("When is Grace's birthday?")
    assert "21 March" in result.fragments[0].content
    assert "conflicts with" in result.as_context()
    flagged = [item for item in json.loads(result.as_context(format="json"))["results"] if item.get("conflicts_with")]
    assert len(flagged) == 2


# ── 7. Dates named in questions ──────────────────────────────────────────────

def test_month_in_question_is_a_soft_window(mem) -> None:
    year = datetime.now(timezone.utc).year
    _fill(mem)
    mem.remember("The printer approved the vendor contract in January.")
    mem.remember(f"Dentist appointment moved to {year}-11-20.")
    mem.remember("What happens at the retro is decided by the team.")
    assert "Dentist" in mem.recall("What happens in November?").fragments[0].content


def test_yesterday_in_question_prefers_yesterdays_memories(mem) -> None:
    yesterday = (datetime.now(timezone.utc) - timedelta(days=1)).replace(hour=12).isoformat()
    _fill(mem)
    mem.remember("We had the team lunch and did the dishes.")
    mem.remember("We had the security audit kickoff.", occurred_at=yesterday)
    assert "audit" in mem.recall("What did we do yesterday?").fragments[0].content


def test_since_filter_leaving_nothing_relevant_returns_nothing(mem) -> None:
    mem.remember("Old note: renewed the domain.", occurred_at="2025-03-01")
    _fill(mem)
    result = mem.recall("domain renewal", since="2026-01-01")
    assert result.fragments == []


# ── 8. Gap noise ─────────────────────────────────────────────────────────────

def test_missing_anchor_is_reported_as_written_with_one_action(mem) -> None:
    _fill(mem)
    gap = mem.recall("What did the Quuxinator-9000 cost?").gap_report
    assert gap.severity in (GapSeverity.HIGH, GapSeverity.CRITICAL)
    assert "Quuxinator-9000" in gap.reason and gap.missing == ["Quuxinator-9000"]
    assert gap.action_hint == gap.suggested_action == "web_search"
    assert GapReport().action_hint == "answer"
    assert GapReport(severity=GapSeverity.HIGH, suggested_action="web_search").action_hint == "web_search"


def test_no_false_gap_for_a_full_match(mem) -> None:
    _fill(mem)
    mem.remember("My monthly rent is 1,350 EUR, paid on the 1st.")
    mem.remember("My manager is Priya Raman, who reports to the CTO, David Okafor.")
    assert not mem.recall("How much rent do I pay?").gap_detected
    assert not mem.recall("Who does Priya report to?").gap_detected


def test_boss_question_reaches_the_manager_fact(mem) -> None:
    mem.facts.set("user", "manager", "Priya Raman")
    mem.facts.set("user", "city", "Porto")
    facts = mem.recall("Who is my boss?").facts
    assert [f.attribute for f in facts] == ["manager"]


# ── 9. Small-model chat text ─────────────────────────────────────────────────

@pytest.mark.parametrize("text", [
    "My dog is called Biscuit, he's a 4-year-old beagle.",
    "Step 3: bake at 180 C for 35 minutes in a paper-lined tin.",
    "We booked a 3-day trip with a well-known guide.",
])
def test_hyphenated_adjectives_are_not_anchors(text: str) -> None:
    assert not set(query_anchors(text)) & {"4 year old", "paper lined", "3 day", "well known"}


def test_identifiers_are_still_anchors() -> None:
    assert query_anchors("Is atlas-core in eu-west-1 since HAL-1234?") == ["atlas core", "eu west 1", "hal 1234"]


@pytest.mark.parametrize("content, name, expected", [
    ("I work as a nurse at Sao Joao hospital.", "joao", False),
    ("My friend Joao's phone number is +351 912.", "joao", True),
    ("I work as a nurse at Sao Joao hospital.", "sao joao", True),
    ("My friend Joao's phone number is +351 912.", "sao joao", False),
    ("joao changed his number", "joao", True),
])
def test_whole_name_matching(content: str, name: str, expected: bool) -> None:
    assert name_mentioned(content, name) is expected
