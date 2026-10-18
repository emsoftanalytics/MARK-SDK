# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 MARK Contributors
"""G4 ranking and token economy: intent-aware retirement, displayed score = order,
relevance floor, hard token cap, lean rendering, facts that match, document crowding."""
from __future__ import annotations

import json
import math

import pytest

from mark.embeddings import HashEmbeddingProvider
from mark.intelligence.fact_relevance import select_facts
from mark.intelligence.query_intent import retirement_applies
from mark.memory.runtime import MarkRuntime


def _tokens(text: str) -> int:
    return math.ceil(len(text) / 4)


@pytest.fixture()
def mem():
    runtime = MarkRuntime.local(embedder=HashEmbeddingProvider())
    yield runtime.memory("g4")
    runtime.shutdown()


# ── 1. Intent-aware retirement ───────────────────────────────────────────────

@pytest.mark.parametrize("query, expected", [
    ("Which TODOs are still open?", True),
    ("What is the current default code length?", True),
    ("How are short codes generated?", True),
    ("tomllib TypeError binary mode", False),
    ("Why did old databases fail after adding expiry?", False),
    ("How did we fix the collision bug?", False),
    ("What error did tomllib raise and how was it fixed?", False),
    ("How were codes generated originally?", False),
    ("sqlite3.OperationalError: table urls has no column named expires_at", False),
])
def test_retirement_applies_only_to_current_state_questions(query: str, expected: bool) -> None:
    assert retirement_applies(query) is expected


def _shorty(mem) -> dict[str, str]:
    ids = {
        "bug": mem.remember("BUG: shorty/config.py::load_config raised TypeError: File must be opened in binary "
                            "mode because tomllib.load needs a binary file.", kind="bug"),
        "fix": mem.remember("FIX: shorty/config.py::load_config opens the TOML file with 'rb'.", kind="fix"),
        "dec": mem.remember("DECISION: shorty config is TOML read with stdlib tomllib from ~/.config/shorty.",
                            kind="decision"),
        "todo_done": mem.remember("TODO: add a stats command showing hits per code", kind="todo"),
        "todo_open": mem.remember("TODO: publish shorty to PyPI.", kind="todo"),
    }
    for i in range(6):
        mem.remember(f"shorty note {i}: the binary mode flag is documented in the README.")
    mem.resolve(ids["bug"], note="open with rb")
    mem.resolve(ids["todo_done"], note="stats command")
    return ids


def test_error_text_query_ranks_the_resolved_bug_by_relevance(mem) -> None:
    ids = _shorty(mem)
    result = mem.recall("tomllib TypeError binary mode")
    assert result.fragments[0].id == ids["bug"]
    why = mem.recall("Why did load_config raise TypeError?")
    assert ids["bug"] in [f.id for f in why.fragments[:2]]


def test_current_state_question_still_demotes_resolved_items(mem) -> None:
    ids = _shorty(mem)
    order = [f.id for f in mem.recall("Which TODOs are still open?").fragments]
    assert order.index(ids["todo_open"]) < order.index(ids["todo_done"]) if ids["todo_done"] in order else True


# ── 2. Displayed score = sort order ──────────────────────────────────────────

def test_displayed_scores_never_increase_down_the_list(mem) -> None:
    _shorty(mem)
    for query in ("tomllib TypeError binary mode", "Which TODOs are still open?", "shorty config file",
                  "What does the README document?"):
        result = mem.recall(query)
        shown = result.relevance
        assert all(a >= b for a, b in zip(shown, shown[1:])), (query, shown)
        payload = json.loads(result.as_context(format="json"))
        scores = [item["score"] for item in payload["results"]]
        assert scores == sorted(scores, reverse=True)


# ── 3. Token economy ─────────────────────────────────────────────────────────

def test_zero_relevance_results_are_not_padded(mem) -> None:
    mem.remember("The gate code for the warehouse is 5521.")
    for i in range(12):
        mem.remember(f"Lunch menu {i}: soup, bread and a salad of the day.")
    result = mem.recall("What is the warehouse gate code?")
    assert result.fragments[0].content.startswith("The gate code")
    assert len(result.fragments) <= 3
    assert all(r > 0 for r in result.relevance)


@pytest.mark.parametrize("budget", [60, 100, 400])
def test_max_tokens_is_a_hard_cap(mem, budget: int) -> None:
    for i in range(10):
        mem.remember(f"Deploy note {i}: " + "the deploy pipeline builds images and pushes them " * 20)
    result = mem.recall("deploy pipeline images")
    for fmt in ("xml", "numbered", "json"):
        assert _tokens(result.as_context(max_tokens=budget, format=fmt)) <= budget, fmt


def test_default_rendering_has_no_state_or_excerpt_labels(mem) -> None:
    mem.remember("Deploy runbook: " + "check the dashboards before you deploy. " * 80 + "Rollback with make undo.")
    result = mem.recall("How do I roll back a deploy?")
    xml, text = result.as_context(), result.as_text()
    for rendered in (xml, text):
        assert "unverified" not in rendered and "excerpt" not in rendered
    entry = json.loads(result.as_context(format="json"))["results"][0]
    assert entry["state"] == "unverified" and entry["excerpt_of_chars"] > 1500


def test_facts_attach_only_when_the_question_matches_them() -> None:
    rows = [{"attribute": "code_length", "value": "7"}, {"attribute": "config_path", "value": "~/.config"},
            {"attribute": "db_env_var", "value": "SHORTY_DB"}, {"attribute": "schema_version", "value": "1"}]
    assert select_facts("shorty: Why did expired links still resolve?", "shorty", rows) == []
    assert select_facts("Which TODOs of shorty are still open?", "shorty", rows) == []
    assert [r["attribute"] for r in select_facts("What is the shorty code length?", "shorty", rows)] == ["code_length"]
    assert len(select_facts("Tell me about shorty", "shorty", rows)) == 4
    assert len(select_facts("What is shorty?", "shorty", rows)) == 3


# ── 4. Documents ─────────────────────────────────────────────────────────────

def _renovation_doc() -> str:
    filler = ("The team discussed the sequencing of the renovation work. Every change to the renovation plan "
              "must be confirmed by email. Storage was a recurring theme in the plan. ")
    rooms = []
    for room in ("Kitchen", "Bathroom", "Living room", "Bedroom", "Hallway", "Terrace", "Office", "Laundry"):
        body = f"In the {room.lower()}, " + filler * 6
        if room == "Living room":
            body += "The main contractor for the whole renovation is Bruno Almeida, phone 912 345 678. "
        rooms.append(f"{room}\n\n{body}\n")
    return "Apartment renovation plan\n\n" + "\n".join(rooms)


def test_document_chunks_do_not_crowd_out_short_memories(mem) -> None:
    short = mem.remember("Renovation reminder: the plan deposit of 2,000 euros is due on Friday.")
    record = mem.store_document(_renovation_doc(), title="Renovation plan")
    assert len(record.chunk_ids) >= 6
    result = mem.recall("renovation plan deposit")
    ids = [f.id for f in result.fragments]
    assert short in ids[:5]
    per_doc = sum(1 for fid in ids if fid in set(record.chunk_ids))
    assert per_doc <= 2


def test_answer_chunk_ranks_first_despite_heading_prefix(mem) -> None:
    record = mem.store_document(_renovation_doc(), title="Renovation plan")
    result = mem.recall("Who is the renovation contractor?")
    top = [f for f in result.fragments[:2] if f.id in set(record.chunk_ids)]
    assert any("Bruno Almeida" in f.content for f in top)


# ── G2 follow-ups: open-only scope, kinds ───────────────────────────────────

def test_exclude_tags_filters_before_top_k() -> None:
    from mark.intelligence.retrieval_policy import SessionFilter

    runtime = MarkRuntime.local(embedder=HashEmbeddingProvider())
    try:
        mem = runtime.memory("g4")
        done = [mem.remember(f"TODO: migrate service {i} to the new queue", kind="todo") for i in range(12)]
        for fid in done:
            mem.resolve(fid, note="done")
        still_open = mem.remember("TODO: migrate the billing service to the new queue", kind="todo")
        sf = SessionFilter(exclude_tags=["mark:resolved"])
        result = runtime.pipeline.retrieve("migrate service to the new queue", "g4", session_filter=sf)
        assert [f.id for f in result.fragments] == [still_open]
    finally:
        runtime.shutdown()


def test_fragment_kind_and_json_kind(mem) -> None:
    fid = mem.remember("TODO: publish shorty to PyPI.", kind="todo")
    fragment = mem.get(fid)
    assert fragment.kind == "todo"
    assert mem.get(mem.remember("plain note about shorty")).kind is None
    result = mem.recall("publish shorty")
    assert json.loads(result.as_context(format="json"))["results"][0]["kind"] == "todo"
    assert result.to_dict()["results"][0]["kind"] == "todo"
