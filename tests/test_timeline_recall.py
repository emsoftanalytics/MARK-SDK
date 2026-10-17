# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 MARK Contributors
"""Timelines beyond the basics: chronological recall, CLI JSON helper,
encrypted stores (no FTS5), middleware routing and a performance smoke test.
"""
from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

import pytest

from mark.memory.runtime import MarkRuntime
from mark.memory.timeline import timeline_json
from mark.middlewares.base import BaseMiddleware

from _helpers_a8_temporal import runtime  # noqa: F401  (pytest fixture)

_TRANSFERS = [
    "[2026-04-07] HR: Tomasz Nowak transferred to the Messaging team.",
    "[2026-03-08] HR: Tomasz Nowak transferred to the Data team.",
    "[2026-04-07] HR: Tomasz Nowak transferred to the Platform team.",
    "[2026-01-15] HR: Tomasz Nowak transferred to the Billing team.",
]


# ── chronological recall ─────────────────────────────────────────────────────

def test_chronological_recall_reorders_the_same_results(runtime) -> None:
    mem = runtime.memory("t")
    mem.store_many_sync(_TRANSFERS)
    relevance = mem.retrieve_sync("Tomasz Nowak transferred team")
    chrono = mem.retrieve_sync("Tomasz Nowak transferred team", order="chronological")

    assert {f.id for f in chrono.fragments} == {f.id for f in relevance.fragments}
    assert [f.content.split("the ")[-1] for f in chrono.fragments] == [
        "Billing team.", "Data team.", "Messaging team.", "Platform team.",
    ]
    # Each score still belongs to its own fragment (scores drift between
    # recalls through reinforcement, so permute one result in place).
    from mark.memory.timeline_order import chronological_result

    resorted = chronological_result(relevance)
    pairs = dict(zip((f.id for f in relevance.fragments), zip(relevance.scores, relevance.relevance)))
    assert list(zip(resorted.scores, resorted.relevance)) == [pairs[f.id] for f in resorted.fragments]


def test_default_recall_order_is_unchanged(runtime) -> None:
    mem = runtime.memory("t")
    mem.store_many_sync(_TRANSFERS)
    default = mem.retrieve_sync("Tomasz Nowak transferred team")
    explicit = mem.retrieve_sync("Tomasz Nowak transferred team", order="relevance")
    assert [f.id for f in default.fragments] == [f.id for f in explicit.fragments]
    assert default.scores == sorted(default.scores, reverse=True)


def test_chronological_recall_keeps_two_clock_rule(runtime) -> None:
    mem = runtime.memory("t")
    mem.store_sync("Tomasz Nowak team note without a date.")
    time.sleep(0.01)
    mem.store_many_sync(_TRANSFERS[:2])
    time.sleep(0.01)
    mem.store_sync("Tomasz Nowak team change just now, to Security.")
    chrono = mem.retrieve_sync("Tomasz Nowak team", order="chronological")
    contents = [f.content for f in chrono.fragments]
    assert contents[:2] == [_TRANSFERS[1], _TRANSFERS[0]]
    assert contents[2:] == ["Tomasz Nowak team note without a date.",
                            "Tomasz Nowak team change just now, to Security."]


def test_invalid_order_is_rejected_and_async_accepts_order(runtime) -> None:
    mem = runtime.memory("t")
    mem.store_many_sync(_TRANSFERS)
    with pytest.raises(ValueError):
        mem.retrieve_sync("Tomasz Nowak", order="newest")
    result = asyncio.run(mem.retrieve("Tomasz Nowak transferred team", order="chronological"))
    assert result.fragments[0].content == _TRANSFERS[3]


# ── CLI helper ───────────────────────────────────────────────────────────────

def test_timeline_json_is_serialisable_and_complete(runtime) -> None:
    mem = runtime.memory("t")
    mem.store_many_sync(_TRANSFERS)
    payload = timeline_json(mem, "Tomasz Nowak", "team", limit=3)
    json.dumps(payload)  # must not raise

    assert payload["count"] == 3 and payload["truncated"] == 1
    assert payload["latest"]["text"].endswith("Platform team.")
    assert payload["latest"]["same_day"] == "2/2" and payload["latest"]["dated"] is True
    assert [e["kind"] for e in payload["events"]] == ["memory"] * 3
    assert payload["text"].splitlines()[0] == "Timeline of Tomasz Nowak · team: 3 events (+1 earlier)"


# ── encrypted store: no FTS5, full-scan fallback ─────────────────────────────

def test_encrypted_store_without_fts_still_builds_timelines(tmp_path: Path) -> None:
    pytest.importorskip("cryptography")
    from mark.security import FernetEncryptionProvider

    rt = MarkRuntime.local(store_path=tmp_path / "enc.db",
                           encryption=FernetEncryptionProvider(passphrase="pw"))
    try:
        mem = rt.memory("t")
        assert not rt.store.fts_available
        mem.store_many_sync(_TRANSFERS + ["Titanium order shipped.", "Titan kicked off."])
        tl = mem.timeline("Tomasz Nowak")
        assert len(tl) == 4 and tl.latest().text.endswith("Platform team.")
        assert [e.text for e in mem.timeline("titan")] == ["Titan kicked off."]
    finally:
        rt.shutdown()


# ── middleware routing ───────────────────────────────────────────────────────

class _Recorder(BaseMiddleware):
    name = "recorder"

    def __init__(self) -> None:
        self.operations: list[str] = []

    def before(self, context) -> None:
        self.operations.append(context.operation)


def test_timeline_runs_through_the_middleware_stack(tmp_path: Path) -> None:
    recorder = _Recorder()
    rt = MarkRuntime.local(store_path=tmp_path / "m.db", middleware=[recorder])
    try:
        mem = rt.memory("t")
        mem.store_sync(_TRANSFERS[0])
        assert len(mem.timeline("Tomasz Nowak")) == 1
        assert "timeline" in recorder.operations
    finally:
        rt.shutdown()


# ── performance smoke test ───────────────────────────────────────────────────

def test_long_history_is_fast(runtime) -> None:
    """A modest version of the 5,000-in-50,000 benchmark (see the report)."""
    mem = runtime.memory("t")
    texts = [
        f"[2026-{1 + i % 12:02d}-{1 + i % 28:02d}] HR: Tomasz Nowak transferred to team {i}."
        if i % 10 == 0 else f"[2026-01-01] Note {i}: Anna Kowalska reviewed HAL-{i} titanium."
        for i in range(5000)
    ]
    for start in range(0, len(texts), 1000):
        mem.store_many_sync(texts[start:start + 1000])
    started = time.perf_counter()
    tl = mem.timeline("Tomasz Nowak", "team")
    elapsed = time.perf_counter() - started
    assert len(tl) == 500
    assert elapsed < 1.0, f"timeline took {elapsed:.3f}s"
