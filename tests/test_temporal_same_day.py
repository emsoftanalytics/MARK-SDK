# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 Somtochukwu Owen Ikeanyi
"""Same-day events keep their logged order (found by the run-2 retention experiment)."""
from __future__ import annotations

from pathlib import Path

from mark import ContextWatch
from mark.memory.runtime import MarkRuntime


def test_same_day_changes_follow_storage_order(tmp_path: Path) -> None:
    runtime = MarkRuntime.local(store_path=tmp_path / "d.db")
    mem = runtime.memory("t")
    mem.store_many_sync([
        "[2026-04-02] CHANGE: willow-gateway moved from port 4100 to port 5200.",
        "[2026-04-02] CHANGE: willow-gateway moved from port 5200 to port 5810.",
    ])
    now = mem.retrieve_sync("Which port does willow-gateway listen on now?")
    first = mem.retrieve_sync("What was the original port of willow-gateway?")
    runtime.shutdown()

    assert "to port 5810" in now.fragments[0].content
    assert "to port 5200" in first.fragments[0].content


def test_watch_briefing_orders_same_day_events(tmp_path: Path) -> None:
    runtime = MarkRuntime.local(store_path=tmp_path / "w.db")
    mem = runtime.memory("t")
    mem.store_many_sync([
        "[2026-03-15] TICKET HAL-4861 status changed to in progress.",
        "[2026-03-15] TICKET HAL-4861 status changed to resolved.",
        "[2026-03-15] TICKET HAL-4861 status changed to closed.",
    ])
    with ContextWatch(mem, background=False) as watch:
        watch.observe("What is happening with HAL-4861?", role="user")
        lines = watch.briefing().text.splitlines()
    runtime.shutdown()

    assert "closed" in lines[2]


def test_recall_output_shows_same_day_order(tmp_path: Path) -> None:
    """Run-2 MH10: two same-day transfers must not leave the reader guessing."""
    from mark._cli.engine import _result_json

    runtime = MarkRuntime.local(store_path=tmp_path / "o.db")
    mem = runtime.memory("t")
    mem.store_many_sync([
        "[2026-04-07] HR: Tomasz Nowak transferred to the Messaging team.",
        "[2026-04-07] HR: Tomasz Nowak transferred to the Platform team.",
        "[2026-03-08] HR: Tomasz Nowak transferred to the Data team.",
    ])
    result = mem.retrieve_sync("Tomasz Nowak transferred team")
    runtime.shutdown()

    labels = {m["content"].split("the ")[-1]: m["same_day"] for m in _result_json(result, 10)["memories"]}
    assert labels == {"Messaging team.": "1/2", "Platform team.": "2/2", "Data team.": None}
    assert "same day 2/2] [2026-04-07] HR: Tomasz Nowak transferred to the Platform team." in result.as_context()
