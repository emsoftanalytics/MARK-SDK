"""a10 query understanding: code identifiers and names as anchors, temporal intent words (B4, C5)."""
from __future__ import annotations

import pytest

from mark.intelligence.query_understanding import (
    TemporalIntent,
    anchor_present,
    detect_temporal_intent,
    gap_anchors,
    is_blank_query,
    query_anchors,
)


@pytest.mark.parametrize("query, anchors", [
    ("What is TaskQueue used for?", ["taskqueue"]),
    ("What does PaymentsService do?", ["paymentsservice"]),
    ("Where is load_config defined?", ["load config"]),
    ("SESSION_TIMEOUT_MIN", ["session timeout min"]),
    ("debounceSearch", ["debouncesearch"]),
    ("What does TodoStore._migrate do?", ["todostore migrate"]),
    ("How is authentication implemented?", []),
    ("Which team is Marcus Webb on?", ["marcus webb"]),
])
def test_code_identifiers_are_anchors(query: str, anchors: list[str]) -> None:
    assert query_anchors(query) == anchors


@pytest.mark.parametrize("query, anchors", [
    ("How is Redis configured?", ["redis"]),
    ("Who does Priya report to?", ["priya"]),
    ("Who sits above Tomasz's boss?", ["tomasz"]),
    ("Redis is down, what now?", []),                         # sentence-initial capital
    ("What did we talk about last Tuesday?", []),             # calendar words are not names
    ("When is my Chinese class?", []),
    ("Which AWS region hosts it?", []),                       # all-caps words are not names
    ("Is it ok? Then check Kafka.", ["kafka"]),
])
def test_gap_anchors_add_single_proper_nouns(query: str, anchors: list[str]) -> None:
    assert gap_anchors(query) == anchors


def test_anchor_present_accepts_identifier_parts_in_one_memory() -> None:
    assert anchor_present("todostore init calls self migrate", "todostore migrate")
    assert not anchor_present("todostore init", "todostore migrate")


@pytest.mark.parametrize("query, intent", [
    ("What was the last bug?", TemporalIntent.CURRENT),
    ("Which deploy happened most recently?", TemporalIntent.CURRENT),
    ("What is the newest release?", TemporalIntent.CURRENT),
    ("Is 10.0.0.12 still the gateway?", TemporalIntent.CURRENT),
    ("What did we talk about last Tuesday?", TemporalIntent.NONE),
    ("What happened last week?", TemporalIntent.NONE),
    ("What was the test command originally?", TemporalIntent.ORIGINAL),
    ("What did we use previously?", TemporalIntent.ORIGINAL),
    ("Where did I live before?", TemporalIntent.ORIGINAL),
    ("What did I use to drink? I used to drink coffee", TemporalIntent.ORIGINAL),
    ("Which tool is used to deploy?", TemporalIntent.NONE),
    ("What is the user's last name?", TemporalIntent.NONE),
    ("How long does the battery last?", TemporalIntent.NONE),
    ("Traceback (most recent call last)", TemporalIntent.CURRENT),
])
def test_temporal_intent(query: str, intent: TemporalIntent) -> None:
    assert detect_temporal_intent(query) == intent


@pytest.mark.parametrize("query, blank", [("", True), ("   ", True), ("?!...", True), ("ok", False), ("上海", False)])
def test_blank_queries(query: str, blank: bool) -> None:
    assert is_blank_query(query) is blank
