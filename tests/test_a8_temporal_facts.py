# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 MARK Contributors
"""v0.2.0a8: query understanding (temporal intent, anchors, stemming), temporal
ordering, and anchor-aware gap detection.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from mark.intelligence.retrieval import (
    TemporalIntent,
    _stem,
    detect_temporal_intent,
    query_anchors,
)
from mark.types import GapSeverity

from _helpers_a8_temporal import runtime


def _port_history(mem) -> None:
    mem.store_sync("[2026-03-01] INVENTORY: atlas-core is owned by Liam, listens on port 6430, uses MongoDB 7.")
    mem.store_sync("[2026-03-30] CHANGE: atlas-core moved from port 9630 to port 4444.")
    mem.store_sync("[2026-03-17] CHANGE: atlas-core moved from port 6430 to port 9239.")
    mem.store_sync("[2026-03-31] DATABASE: atlas-core switched its datastore to PostgreSQL 15.")
    mem.store_sync("[2026-03-29] CHANGE: atlas-gate moved from port 7001 to port 7002.")


# ── query understanding ───────────────────────────────────────────────────────

@pytest.mark.parametrize("query,intent", [
    ("Which port does atlas-core listen on now?", TemporalIntent.CURRENT),
    ("Who currently owns mosaic-gate?", TemporalIntent.CURRENT),
    ("What is the latest release?", TemporalIntent.CURRENT),
    ("What was the original port of quartz-gateway?", TemporalIntent.ORIGINAL),
    ("Which region did it run in at first?", TemporalIntent.ORIGINAL),
    ("Which port does atlas-core listen on?", TemporalIntent.NONE),
])
def test_temporal_intent(query: str, intent: TemporalIntent) -> None:
    assert detect_temporal_intent(query) == intent


def test_query_anchors() -> None:
    assert query_anchors("Which port does atlas-core listen on now?") == ["atlas core"]
    assert query_anchors("What is the status of ticket HAL-0042?") == ["hal 0042"]
    assert query_anchors("Which team is Marcus Webb on?") == ["marcus webb"]
    assert query_anchors("Who owns the service in eu-west-1?") == ["eu west 1"]
    assert query_anchors("How is authentication implemented?") == []


@pytest.mark.parametrize("left,right", [
    ("owned", "owns"), ("owner", "own"), ("listens", "listen"), ("ports", "port"), ("gates", "gate"),
    ("boxes", "box"), ("policies", "policy"), ("running", "run"), ("database", "databases"),
    ("filter", "filtered"), ("order", "ordered"), ("register", "registered"), ("rate", "rating"),
    ("cache", "caching"), ("speed", "speeding"),
])
def test_stemmer_merges_word_forms(left: str, right: str) -> None:
    assert _stem(left) == _stem(right)


@pytest.mark.parametrize("left,right", [("titan", "titanium"), ("status", "state"), ("class", "clas")])
def test_stemmer_keeps_distinct_words_apart(left: str, right: str) -> None:
    assert _stem(left) != _stem(right)


# ── temporal ordering ─────────────────────────────────────────────────────────

def test_current_query_puts_latest_value_first(runtime) -> None:
    mem = runtime.memory("t")
    _port_history(mem)

    result = mem.retrieve_sync("Which port does atlas-core listen on now?")

    assert result.temporal_intent == "current"
    assert "port 4444" in result.fragments[0].content
    # A newer record about a different attribute (datastore) must not jump ahead,
    # and a different service (atlas-gate) is not part of the subject band.
    assert "DATABASE" not in result.fragments[0].content
    assert result.scores == sorted(result.scores, reverse=True)


def test_original_query_puts_first_value_first(runtime) -> None:
    mem = runtime.memory("t")
    _port_history(mem)

    result = mem.retrieve_sync("What was the original port of atlas-core?")

    assert result.temporal_intent == "original"
    assert "port 6430" in result.fragments[0].content and "INVENTORY" in result.fragments[0].content


def test_dates_inside_text_are_not_event_times(runtime) -> None:
    mem = runtime.memory("t")
    mem.store_sync("[2026-03-05] PLAN: the drill for orion-core is scheduled for 2026-04-30.")
    mem.store_sync("[2026-03-20] PLAN: the drill for orion-core was moved to 2026-04-10.")

    result = mem.retrieve_sync("When is the drill for orion-core scheduled now?")

    assert "2026-04-10" in result.fragments[0].content


def test_two_clocks_are_never_compared(runtime) -> None:
    """Dated records are ordered by declared time; an undated record leads only
    when it was stored after every dated record (a live update after a log import)."""
    mem = runtime.memory("t")
    mem.store_sync("Note: mosaic-gate owner is Kavya.")  # undated, stored before the log
    mem.store_sync("[2026-03-01] OWNERSHIP: mosaic-gate is now owned by Elif.")
    mem.store_sync("[2026-03-16] OWNERSHIP: mosaic-gate is now owned by Wanjiru.")

    ranked = [f.content for f in mem.retrieve_sync("Who currently owns mosaic-gate?").fragments]
    assert "Wanjiru" in ranked[0] and "Kavya" in ranked[2]

    mem.store_sync("Live update: mosaic-gate owner is now Zoe.")  # undated, stored last
    assert "Zoe" in mem.retrieve_sync("Who currently owns mosaic-gate?").fragments[0].content


def test_occurred_at_is_the_declared_event_time(runtime) -> None:
    mem = runtime.memory("t")
    mem.store_sync("vega-core runs in eu-west-1", occurred_at="2026-01-01")
    mem.store_sync("vega-core runs in us-east-1", occurred_at=datetime(2026, 2, 1, tzinfo=timezone.utc))

    assert "us-east-1" in mem.retrieve_sync("Which region does vega-core run in now?").fragments[0].content
    assert "eu-west-1" in mem.retrieve_sync("Original region of vega-core?").fragments[0].content
    with pytest.raises(ValueError):
        mem.store_sync("bad", occurred_at="last tuesday")


# ── anchor-aware gaps ─────────────────────────────────────────────────────────

def test_unknown_names_and_ids_report_a_gap(runtime) -> None:
    mem = runtime.memory("t")
    mem.store_sync("[2026-03-03] INVENTORY: billing-gate listens on port 9066.")
    mem.store_sync("[2026-03-04] HR: Samir Khan transferred to the Messaging team.")
    mem.store_sync("[2026-03-05] TICKET HAL-8557 status changed to in progress.")

    for query, missing in [
        ("Which port does the titan-billing service listen on?", "titan billing"),
        ("Which team is Marcus Webb on?", "marcus webb"),
        ("What is the status of ticket HAL-0042?", "hal 0042"),
    ]:
        result = mem.retrieve_sync(query)
        assert result.gap_report.severity == GapSeverity.HIGH, query
        assert result.missing_anchors == [missing]


def test_known_anchor_outside_top_k_is_not_a_gap(runtime) -> None:
    mem = runtime.memory("t")
    for i in range(40):
        mem.store_sync(f"Routine note {i} about the payments service and its port.")
    mem.store_sync("The sable-core service listens on port 7777 and handles payments.")

    result = mem.retrieve_sync("payments service port sable-core", policy="fast")

    assert result.missing_anchors == []


def test_titan_never_matches_titanium(runtime) -> None:
    mem = runtime.memory("t")
    mem.store_sync("titanium-core listens on port 5000.")

    result = mem.retrieve_sync("Which port does titan-core listen on?")

    assert result.missing_anchors == ["titan core"]


@pytest.mark.parametrize(("query", "anchors"), [
    ("restart checkout-api on 8080; zephyr-9 too", ["checkout api", "zephyr 9"]),
    ("What changed in 2025 for HAL-4433?", ["hal 4433"]),
    ("Scale to 12 replicas by 2026-03-01 at 10:30", []),
    ("Pin web2 to 1.5 cores", ["web2"]),
    ("Is 10.0.0.12 still the gateway?", ["10 0 0 12"]),  # IP addresses stay
])
def test_bare_numbers_are_not_anchors(query: str, anchors: list[str]) -> None:
    assert query_anchors(query) == anchors
