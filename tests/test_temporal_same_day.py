# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 MARK Contributors
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
