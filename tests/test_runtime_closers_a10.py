# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 MARK Contributors
"""runtime.register_closer: background readers stop before the store closes (integ re-test #1)."""
from __future__ import annotations

from pathlib import Path

from mark import Mark
from mark.memory.runtime import MarkRuntime


def test_closers_run_first_newest_first_and_once(tmp_path: Path) -> None:
    runtime = MarkRuntime.local(store_path=tmp_path / "c.db")
    calls: list[str] = []

    def reader() -> None:  # the store must still be open while closers run
        calls.append(f"reader:{runtime.store.count('a')}")

    runtime.register_closer(reader)
    runtime.register_closer(lambda: calls.append("second"))
    dropped = runtime.register_closer(lambda: calls.append("dropped"))
    dropped()
    runtime.register_closer(lambda: 1 / 0)          # a failing closer is logged, not fatal
    runtime.shutdown()
    runtime.shutdown()
    assert calls == ["second", "reader:0"]
    late: list[int] = []
    runtime.register_closer(lambda: late.append(1))  # after shutdown: runs at once
    assert late == [1]


def test_mark_close_runs_the_closers_of_its_runtime(tmp_path: Path) -> None:
    calls: list[str] = []
    with Mark.local(tmp_path) as mark:
        mark.memory._runtime.register_closer(lambda: calls.append("watch"))
    assert calls == ["watch"]
