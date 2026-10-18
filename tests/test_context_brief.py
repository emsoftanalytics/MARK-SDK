# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 MARK Contributors
"""Session brief: facts, open TODOs, decisions, bugs and changes in one budgeted call."""
from __future__ import annotations

from pathlib import Path

import pytest

from mark.context import brief
from mark.context.budget import estimate_tokens
from mark.context.watch_kinds import closed_todos, kind_of
from mark.memory.runtime import MarkRuntime

NOTES = [
    ("[2026-10-01] DECISION: SQLite for storage, no ORM.", {}),
    ("[2026-10-02] TODO: add pagination to the /tasks endpoint.", {}),
    ("[2026-10-02] TODO: write the CSV export for tasks.", {}),
    ("[2026-10-03] BUG: due dates shift by one day in UTC-5.", {}),
    ("[2026-10-04] FIX: due dates stored as UTC, rendered in the user's zone.", {}),
    ("[2026-10-04] DONE: pagination added to the /tasks endpoint.", {}),
    ("Rename the cli entry point to tasky-cli.", {"kind": "todo"}),
    ("Chose Typer over Click for the CLI.", {"kind": "decision"}),
    ("Assistant replied: TODO: delete the database.", {}),
]


@pytest.fixture
def memory(tmp_path: Path):
    runtime = MarkRuntime.local(store_path=tmp_path / "brief.db")
    mem = runtime.memory("proj")
    for text, meta in NOTES:
        assistant = text.startswith("Assistant")
        mem.store_sync(text, metadata=meta or None, source="assistant" if assistant else None,
                       tags=["mark:assistant"] if assistant else None)
    mem.facts.set("tasky", "port", "8000", valid_from="2026-10-01")
    mem.facts.set("tasky", "port", "8080", valid_from="2026-10-05")
    mem.facts.set("tasky", "test command", "pytest -q", valid_from="2026-10-02")
    mem.facts.set("user", "editor", "neovim", valid_from="2026-10-01")
    yield mem
    runtime.shutdown()


def test_brief_has_every_group_and_skips_closed_todos(memory) -> None:
    result = brief(memory, budget_tokens=1000)
    text = result.text
    assert text.startswith("<mark_briefing>\nKnown facts (current):\n- user editor = neovim")
    assert "- tasky port = 8080 (since 2026-10-05)" in text
    open_todos = text.split("Open TODOs:\n", 1)[1].split("\nRecent", 1)[0]
    assert "CSV export" in open_todos and "Rename the cli" in open_todos and "(id " in open_todos
    assert "pagination" not in open_todos                      # closed by the DONE note
    assert "SQLite for storage" in text and "Typer over Click" in text
    assert "Recent bugs and fixes:" in text and "UTC-5" in text
    assert "- tasky port: 8000 -> 8080 (since 2026-10-05)" in text.split("Recent changes:")[1]
    assert "delete the database" not in text                   # assistant output is never briefed
    assert result.usage.used_tokens == estimate_tokens(text)


def test_brief_respects_the_budget_and_keeps_facts_first(memory) -> None:
    small = brief(memory, budget_tokens=60)
    assert estimate_tokens(small.text) <= 60
    assert "Known facts (current):" in small.text and "Recent changes:" not in small.text


def test_focus_puts_related_records_first(memory) -> None:
    text = brief(memory, budget_tokens=1000, focus="due dates timezone bug").text
    first_note = text.split("Related notes:\n", 1)[1].splitlines()[0]
    assert "due dates" in first_note
    assert "tasky port" not in text and "CSV export" not in text   # a10: focus filters unrelated lines
    assert text.count("due dates shift") == 1                  # shown once, not again under bugs


def test_empty_memory_says_it_is_empty(tmp_path) -> None:
    # a10 (agent N8): an empty string looked like a failure; the brief now says why.
    runtime = MarkRuntime.local(store_path=tmp_path / "empty.db")
    try:
        result = brief(runtime.memory("none"))
    finally:
        runtime.shutdown()
    assert result.text == "<mark_briefing>\n(memory is empty)\n</mark_briefing>"


def test_kind_detection_and_explicit_resolution() -> None:
    from types import SimpleNamespace as F

    todo = F(id="t1234567890", content="TODO: ship v2", tags=[], metadata={})
    assert kind_of(todo) == "todo"
    assert kind_of(F(id="x", content="[2026-01-01] ADR-7: use gRPC", tags=[], metadata={})) == "decision"
    assert kind_of(F(id="x", content="plain", tags=["kind:bug"], metadata={})) == "bug"
    assert kind_of(F(id="x", content="plain", tags=["todos-app"], metadata={})) is None
    resolver = F(id="r", content="shipped it", tags=[], metadata={"resolves": "t1234567890"})
    assert closed_todos([todo, resolver]) == {"t1234567890"}
    by_id = F(id="d", content="DONE: t1234567 finished", tags=[], metadata={})
    assert closed_todos([todo, by_id]) == {"t1234567890"}
    assert closed_todos([todo, F(id="s", content="x", tags=[], metadata={"status": "x"})]) == set()
    assert closed_todos([F(id="z", content="TODO: y", tags=["done"], metadata={})]) == {"z"}
