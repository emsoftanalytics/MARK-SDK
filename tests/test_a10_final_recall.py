# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 Somtochukwu Owen Ikeanyi
"""Final a10 recall fixes: exact error text, long floats, score display and ties."""
from __future__ import annotations

import pytest

from mark.embeddings import HashEmbeddingProvider
from mark.intelligence.query_understanding import _content_terms, _coverage
from mark.memory.runtime import MarkRuntime


@pytest.fixture()
def mem():
    runtime = MarkRuntime.local(embedder=HashEmbeddingProvider())
    yield runtime.memory("final")
    runtime.shutdown()


def _spendlog(mem) -> dict[str, str]:
    ids = {
        "float_bug": mem.remember(
            "BUG: tests/test_db.py::test_totals failed with AssertionError: assert {'food': 0.30000000000000004} "
            "== {'food': 0.3} because amounts were stored as REAL floats.", kind="bug"),
        "main_bug": mem.remember(
            "BUG: running python -m spendlog failed with 'No module named spendlog.__main__; spendlog is a "
            "package and cannot be directly executed'.", kind="bug"),
        "main_fix": mem.remember("FIX: added spendlog/__main__.py which calls spendlog.cli.main so python -m "
                                 "spendlog works.", kind="fix"),
        "toml_bug": mem.remember(
            "BUG: spendlog/config.py::load failed with TypeError: File must be opened in binary mode, e.g. use "
            "open('foo.toml', 'rb') because tomllib.load was given a text-mode file.", kind="bug"),
        "toml_fix": mem.remember("FIX: spendlog/config.py::load opens the TOML file with mode 'rb' as tomllib "
                                 "requires.", kind="fix"),
        "toml_dec": mem.remember("DECISION: spendlog configuration is a TOML file read with tomllib; the file "
                                 "is opened once.", kind="decision"),
        "version": mem.remember("Run the spendlog tests with python -m pytest -W error (since 0.3.0)."),
        "move": mem.remember("spendlog default database moved to the XDG data dir in version 0.3.0."),
        "cli": mem.remember("NOTE: the CLI entry point is spendlog/cli.py::main and the console script is "
                            "named spendlog.", kind="note"),
    }
    for key in ("float_bug", "main_bug", "toml_bug"):
        mem.resolve(ids[key], note="fixed")
    return ids


# ── 1. A query that IS error text finds the memory quoting it ────────────────

@pytest.mark.parametrize("query, key", [
    ("No module named spendlog.__main__", "main_bug"),
    ("File must be opened in binary mode", "toml_bug"),
    ("is a package and cannot be directly executed", "main_bug"),
    ("0.30000000000000004", "float_bug"),
    ("AssertionError 0.30000000000000004", "float_bug"),
])
def test_exact_error_text_ranks_the_resolved_bug_first(mem, query: str, key: str) -> None:
    ids = _spendlog(mem)
    result = mem.recall(query)
    assert result.fragments and result.fragments[0].id == ids[key]


def test_long_float_is_one_term_not_a_version() -> None:
    terms = _content_terms("0.30000000000000004")
    assert terms == ["0.30000000000000004"]
    assert _coverage(terms, "since version 0.3.0") == 0.0
    assert _coverage(terms, "assert {'food': 0.30000000000000004} == 1") == 1.0
    # Versions and short decimals keep matching component-wise.
    assert _coverage(_content_terms("version 0.3.0"), "since version 0.3.0") == 1.0
    assert _coverage(_content_terms("2.4 bar"), "tyres at 2.4 bar") == 1.0


def test_long_float_does_not_select_the_version_fact(mem) -> None:
    mem.facts.set("spendlog", "version", "0.3.0")
    _spendlog(mem)
    result = mem.recall("0.30000000000000004")
    assert not any("0.3.0" in fact.as_text() for fact in result.facts)


# ── 7. Gap noise ──────────────────────────────────────────────────────────────

def test_adjacent_identifiers_present_separately_are_not_a_gap(mem) -> None:
    mem.remember("BUG: load_exporters crashed with AttributeError: 'EntryPoints' object has no attribute 'get'.",
                 kind="bug")
    mem.remember("FIX: load_exporters uses entry_points(group='spendlog.exporters') instead of .get.", kind="fix")
    result = mem.recall("EntryPoints AttributeError fix", kind="fix")
    assert result.missing_anchors == []
    assert "Not found" not in (result.gap_report.reason or "")


def test_gap_calibration_rules() -> None:
    from types import SimpleNamespace

    from mark.intelligence.gap_calibration import fact_answers, vector_outlier

    boss = SimpleNamespace(attribute="boss", value="Priya Raman", is_current=True)
    assert fact_answers("Who is my boss?", [boss])
    assert not fact_answers("Who is my dentist?", [boss])
    assert vector_outlier("a", {"a": 0.396, "b": 0.154, "c": 0.149})
    assert not vector_outlier("a", {"a": 0.169, "b": 0.157})  # near-tie
    assert not vector_outlier("b", {"a": 0.396, "b": 0.154})  # not the shown top
    assert not vector_outlier("a", {"a": 0.2, "b": 0.01})  # too weak to be a match


def test_open_todos_question_lists_open_todos_first(mem) -> None:
    ids = _spendlog(mem)
    done = mem.remember("TODO: add a JSON exporter.", kind="todo")
    open_a = mem.remember("TODO: add a spendlog import command for bank CSV statements.", kind="todo")
    open_b = mem.remember("TODO: show budget usage percentages in spendlog report.", kind="todo")
    mem.resolve(done, note="json exporter added")
    result = mem.recall("What TODOs are still open?")
    assert {f.id for f in result.fragments[:2]} == {open_a, open_b}
    assert ids["toml_fix"] not in [f.id for f in result.fragments[:2]]


@pytest.mark.parametrize("query, kind, open_only", [
    ("What TODOs are still open?", "todo", True), ("open bugs", "bug", True),
    ("What tasks are left?", "todo", True), ("What decisions did we make?", "decision", False),
])
def test_asked_kind(query: str, kind: str, open_only: bool) -> None:
    from mark.intelligence.kind_intent import asked_kind

    assert asked_kind(query) == (kind, open_only)


@pytest.mark.parametrize("query", ["What is the todo app?", "How do I open the file?", "Why did the open call fail?"])
def test_not_a_kind_listing(query: str) -> None:
    from mark.intelligence.kind_intent import asked_kind

    assert asked_kind(query) is None
