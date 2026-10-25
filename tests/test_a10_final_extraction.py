# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 Somtochukwu Owen Ikeanyi
"""Final a10 extraction: subject splitting, db path key, retraction + switch, quit/gave up, tool slots."""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from mark.intelligence.fact_extraction import PatternFactExtractor

NOW = datetime(2026, 10, 18, 12, tzinfo=timezone.utc)


def _facts(text: str) -> set[tuple[str, str, str, str]]:
    out = set()
    for c in PatternFactExtractor().extract_changes(text, occurred_at=NOW):
        flag = "previous" if c.previous else ("past" if c.historical else ("retract" if c.retract else ""))
        out.add((c.subject, c.attribute, c.value, flag))
    return out


@pytest.mark.parametrize("text, expected", [
    ("spendlog default currency: EUR", {("spendlog", "default currency", "EUR", "")}),
    ("NOTE: the CLI entry point is spendlog/cli.py::main.", {("project", "cli entry point", "spendlog/cli.py::main", "")}),
    ("spendlog default database path is ~/.spendlog.db, overridable with SPENDLOG_DB.",
     {("spendlog", "db path", "~/.spendlog.db", "")}),
    ("spendlog default database moved from ~/.spendlog.db to ~/.local/share/spendlog/spendlog.db.",
     {("spendlog", "default database", "~/.spendlog.db", "previous"),
      ("spendlog", "default database", "~/.local/share/spendlog/spendlog.db", "")}),
    ("billtrack database path: /var/lib/billtrack/data.db", {("billtrack", "db path", "/var/lib/billtrack/data.db", "")}),
    ("The default timeout is 30 seconds.", {("project", "default timeout", "30 seconds", "")}),
    ("I no longer drink coffee; I switched to green tea.",
     {("user", "drinks", "coffee", "retract"), ("user", "drinks", "green tea", "")}),
    ("I stopped drinking coffee.", {("user", "drinks", "coffee", "retract")}),
    ("I quit drinking soda.", {("user", "drinks", "soda", "retract")}),
    ("I gave up coffee.", {("user", "drinks", "coffee", "retract")}),
    ("I used to work at Acme.", {("user", "employer", "Acme", "past")}),
    ("I work at Acme.", {("user", "employer", "Acme", "")}),
    ("My editor is Vim.", {("user", "editor", "Vim", "")}),
    ("My shell is zsh.", {("user", "shell", "zsh", "")}),
    ("I now report to Dana Whitfield.", {("user", "manager", "Dana Whitfield", "")}),
    ("Update: standup was moved to 10:15.", {("standup", "date", "10:15", "")}),
])
def test_new_phrasings(text: str, expected: set) -> None:
    assert _facts(text) == expected


@pytest.mark.parametrize("text", [
    "My editor at the newspaper is Joan.", "My editor is Joan.", "Maybe my editor is the problem.",
    "The meeting moved to room 4.", "atlas-core: 8080", "TODO: migrate spendlog to Python 3.13.",
    "If I quit coffee I would sleep better.", "I quit.", "We switched to Linear for tickets.",
    "The project uses a database at /tmp/x.db for tests and a cache.",
])
def test_precision_guards(text: str) -> None:
    assert _facts(text) == set()
