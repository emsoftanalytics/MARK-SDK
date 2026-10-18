"""Session brief at small budgets: open TODOs and bugs are reserved, never crowded out.

At ``--budget 100`` the brief used to fill with facts and drop every TODO;
at a budget too small for the briefing tags it returned nothing.
"""
from __future__ import annotations

import pytest

from mark import Mark
from mark.context.session_brief import brief

TODOS = ["TODO: add pagination to the /tasks endpoint and document the cursor format",
         "TODO: rotate the staging database credentials before Friday",
         "TODO: write the migration guide for the 2.0 release"]


@pytest.fixture
def memory(tmp_path, monkeypatch):
    monkeypatch.setenv("MARK_EMBEDDER", "hash")
    with Mark.local(tmp_path) as mark:
        m = mark.runtime.memory("x")
        for i in range(12):
            m.facts.set(f"service{i}", "port", str(8000 + i))
        for text in TODOS:
            m.store_sync(text)
        m.store_sync("BUG: due dates shift by one day in UTC-5 when saved from the mobile app")
        yield m


def test_small_budget_keeps_open_todos_and_bugs(memory) -> None:
    text = brief(memory, budget_tokens=100).text
    assert "Open TODOs:" in text
    assert sum("TODO:" in line for line in text.splitlines()) >= 2
    assert "BUG:" in text
    assert len(text) <= 100 * 4


def test_large_budget_still_shows_facts_and_todos(memory) -> None:
    text = brief(memory, budget_tokens=400).text
    assert "Known facts (current):" in text and text.count("TODO:") == 3


def test_tiny_budget_still_says_something(memory) -> None:
    text = brief(memory, budget_tokens=10).text
    assert text and "TODO" in text and len(text) <= 40   # the top TODO, cut to fit
    assert brief(memory, budget_tokens=40).text.count("TODO:") >= 1
