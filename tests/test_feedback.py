# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 MARK Contributors
from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from mark import Mark
from mark.adapters.backend import LocalMarkBackend
from mark.middlewares.base import MiddlewareStack


def test_feedback_persists_and_is_bounded(tmp_path: Path) -> None:
    with Mark.local(tmp_path) as mark:
        record = mark.memory.block("fb").write("Feedback target.", importance=0.5)

        assert mark.memory.feedback(record.id, 1.0) is True
        assert mark.memory.feedback(record.id, 5.0) is True  # clamped to 1.0
        loaded = mark.runtime.store.get(record.id)
        assert loaded.importance == pytest.approx(0.6)
        assert loaded.metadata["feedback_count"] == 2
        assert loaded.metadata["feedback_sum"] == pytest.approx(2.0)

        mark.memory.feedback(record.id, -1.0)
        assert mark.runtime.store.get(record.id).importance == pytest.approx(0.55)
        assert mark.memory.feedback("missing-id", 1.0) is False


def test_backend_feedback_reaches_memory(tmp_path: Path) -> None:
    with Mark.local(tmp_path) as mark:
        record = mark.memory.block("fb").write("Backend feedback target.", importance=0.5)
        backend = LocalMarkBackend(mark)

        result = asyncio.run(backend.feedback(fragment_ids=[record.id], outcome="completed"))

        assert result.ok
        assert mark.runtime.store.get(record.id).importance > 0.5


def test_from_bound_does_not_rebind() -> None:
    class Spy:
        binds = 0

        def bind(self, runtime) -> None:
            Spy.binds += 1

    item = Spy()
    original = MiddlewareStack([item])
    original.bind(object())
    copy = MiddlewareStack.from_bound(original.list(), object())

    assert Spy.binds == 1
    assert copy.list() == [item] and copy is not original
