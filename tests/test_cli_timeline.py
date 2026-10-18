# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 MARK Contributors
"""The `mark timeline` command and `recall --chronological`."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from mark import cli


def _run(capsys, *argv: str) -> str:
    assert cli.main(list(argv)) == 0
    return capsys.readouterr().out


def _seed(tmp_path: Path, capsys) -> str:
    db = str(tmp_path / "t.db")
    log = tmp_path / "log.txt"
    # Stored newest first on purpose: output order must follow declared dates.
    log.write_text(
        "[2026-04-07] Dana Kim moved from Payments to the Search team.\n"
        "[2026-04-07] Dana Kim moved from Search to the Billing team.\n"
        "[2026-02-01] Dana Kim joined the Payments team.\n"
        "[2026-03-01] The titanium build server was upgraded.\n"
    )
    _run(capsys, "--db", db, "--no-server", "ingest", str(log))
    return db


def test_timeline_lists_subject_events_oldest_first(tmp_path: Path, capsys) -> None:
    db = _seed(tmp_path, capsys)
    header, *lines = _run(capsys, "--db", db, "--no-server", "timeline", "Dana Kim").splitlines()
    assert header == "Timeline of Dana Kim: 3 events" and len(lines) == 3
    assert "joined the Payments team" in lines[0]
    # a10: same-day events keep their order without a "same day k/m" marker.
    assert "same day" not in "\n".join(lines)
    assert "to the Search team" in lines[1] and "to the Billing team" in lines[2]


def test_timeline_json_limit_and_window(tmp_path: Path, capsys) -> None:
    db = _seed(tmp_path, capsys)
    data = json.loads(_run(capsys, "--db", db, "--no-server", "timeline", "Dana Kim", "--limit", "1", "--json"))
    assert data["count"] == 1 and data["truncated"] == 2
    assert "Billing" in data["latest"]["text"]
    windowed = json.loads(_run(capsys, "--db", db, "--no-server", "timeline", "Dana Kim",
                               "--until", "2026-03-31", "--json"))
    assert [e["when"][:10] for e in windowed["events"]] == ["2026-02-01"]
    assert "joined the Payments team" in windowed["events"][0]["text"]


def test_timeline_unknown_subject(tmp_path: Path, capsys) -> None:
    db = _seed(tmp_path, capsys)
    # "titanium" must not count as a mention of "Titan".
    assert "Titan: 0 events" in _run(capsys, "--db", db, "--no-server", "timeline", "Titan")


def test_recall_chronological_keeps_results_and_sorts_by_date(tmp_path: Path, capsys) -> None:
    db = _seed(tmp_path, capsys)
    base = ["--db", db, "--no-server", "recall", "Dana Kim team", "-k", "3", "--json"]
    by_relevance = json.loads(_run(capsys, *base))
    by_time = json.loads(_run(capsys, *base, "--chronological"))
    ids = lambda r: [m["id"] for m in r["memories"]]  # noqa: E731
    assert sorted(ids(by_relevance)) == sorted(ids(by_time))
    contents = [m["content"] for m in by_time["memories"]]
    assert contents[0].startswith("[2026-02-01]")
    assert contents.index(next(c for c in contents if "Search" in c)) < \
        contents.index(next(c for c in contents if "Billing" in c))


def test_recall_many_chronological_payload(tmp_path: Path) -> None:
    db = tmp_path / "e.db"
    result = cli.execute(db, "recall-many", {"agent": "a", "queries": ["x"], "order": "chronological"},
                         use_server=False)
    assert result["results"][0]["memories"] == []
    with pytest.raises(ValueError):
        cli.execute(db, "recall", {"agent": "a", "query": "x", "order": "random"}, use_server=False)
