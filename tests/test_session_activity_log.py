# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 MARK Contributors
from __future__ import annotations

from pathlib import Path

import pytest

from mark import LocalMemoryStore, SessionActivityLog
from mark.memory.runtime import MarkRuntime


# ---------------------------------------------------------------------------
# Store-level: log_event / list_events
# ---------------------------------------------------------------------------

def test_log_event_returns_id() -> None:
    store    = LocalMemoryStore()
    event_id = store.log_event("agent-1", "observe")
    assert isinstance(event_id, str) and len(event_id) > 0


def test_list_events_returns_logged_event() -> None:
    store = LocalMemoryStore()
    store.log_event("agent-1", "observe", query="Elena scene")
    events = store.list_events("agent-1")
    assert len(events) == 1
    assert events[0]["event_type"] == "observe"
    assert events[0]["query"] == "Elena scene"


def test_list_events_filters_by_agent() -> None:
    store = LocalMemoryStore()
    store.log_event("agent-A", "retrieve")
    store.log_event("agent-B", "observe")
    assert len(store.list_events("agent-A")) == 1
    assert len(store.list_events("agent-B")) == 1


def test_list_events_filters_by_session() -> None:
    store = LocalMemoryStore()
    store.log_event("agent", "retrieve", session_id="ep-01")
    store.log_event("agent", "observe",  session_id="ep-02")
    ep1 = store.list_events("agent", session_id="ep-01")
    assert len(ep1) == 1
    assert ep1[0]["session_id"] == "ep-01"


def test_list_events_filters_by_event_type() -> None:
    store = LocalMemoryStore()
    store.log_event("agent", "retrieve")
    store.log_event("agent", "observe")
    store.log_event("agent", "retrieve")
    retrieves = store.list_events("agent", event_type="retrieve")
    assert len(retrieves) == 2
    assert all(e["event_type"] == "retrieve" for e in retrieves)


def test_list_events_respects_limit() -> None:
    store = LocalMemoryStore()
    for _ in range(10):
        store.log_event("agent", "observe")
    events = store.list_events("agent", limit=3)
    assert len(events) == 3


def test_list_events_newest_first() -> None:
    store = LocalMemoryStore()
    store.log_event("agent", "event-1")
    store.log_event("agent", "event-2")
    events = store.list_events("agent")
    assert events[0]["event_type"] == "event-2"   # most recent first


def test_log_event_stores_metadata() -> None:
    store = LocalMemoryStore()
    store.log_event("agent", "observe", metadata={"text_len": 80, "tags": ["character:elena"]})
    e = store.list_events("agent")[0]
    assert e["metadata"]["text_len"] == 80
    assert "character:elena" in e["metadata"]["tags"]


def test_log_event_stores_fragment_id() -> None:
    store = LocalMemoryStore()
    store.log_event("agent", "write", fragment_id="frag-123")
    e = store.list_events("agent")[0]
    assert e["fragment_id"] == "frag-123"


def test_list_events_empty_when_none_logged() -> None:
    store = LocalMemoryStore()
    assert store.list_events("agent") == []


# ---------------------------------------------------------------------------
# SessionActivityLog helper
# ---------------------------------------------------------------------------

def test_session_log_log_method() -> None:
    store = LocalMemoryStore()
    log   = SessionActivityLog(store, "agent-X", "ep-01")
    eid   = log.log("retrieve", query="What is Elena's scarf color?")
    events = store.list_events("agent-X", session_id="ep-01")
    assert len(events) == 1
    assert events[0]["id"] == eid


def test_session_log_list_method() -> None:
    store = LocalMemoryStore()
    log   = SessionActivityLog(store, "agent-X", "ep-01")
    log.log("observe")
    log.log("retrieve")
    assert len(log.list()) == 2


def test_session_log_list_filtered_by_event_type() -> None:
    store = LocalMemoryStore()
    log   = SessionActivityLog(store, "agent", "ep-01")
    log.log("observe")
    log.log("retrieve")
    log.log("observe")
    assert len(log.list(event_type="observe")) == 2


def test_session_log_summary() -> None:
    store = LocalMemoryStore()
    log   = SessionActivityLog(store, "agent", "ep-01")
    log.log("observe")
    log.log("observe")
    log.log("retrieve")
    s = log.summary()
    assert s["total"] == 3
    assert s["by_type"]["observe"]  == 2
    assert s["by_type"]["retrieve"] == 1


def test_session_log_summary_no_session() -> None:
    store = LocalMemoryStore()
    log   = SessionActivityLog(store, "agent", None)
    log.log("observe")
    s = log.summary()
    assert s["session_id"] is None
    assert s["total"] == 1


# ---------------------------------------------------------------------------
# MarkRuntime.session_log integration
# ---------------------------------------------------------------------------

def test_runtime_session_log_returns_activity_log(tmp_path: Path) -> None:
    runtime = MarkRuntime.local(store_path=tmp_path / "mem.db")
    log     = runtime.session_log("agent", "ep-01")
    assert isinstance(log, SessionActivityLog)
    runtime.shutdown()


def test_runtime_session_log_scoped_to_agent(tmp_path: Path) -> None:
    runtime = MarkRuntime.local(store_path=tmp_path / "mem.db")
    log_a   = runtime.session_log("agent-A")
    log_b   = runtime.session_log("agent-B")

    log_a.log("observe")
    assert len(log_b.list()) == 0
    assert len(log_a.list()) == 1
    runtime.shutdown()


def test_runtime_session_log_no_session_id(tmp_path: Path) -> None:
    runtime = MarkRuntime.local(store_path=tmp_path / "mem.db")
    log     = runtime.session_log("agent")
    log.log("observe", metadata={"note": "cross-session"})
    events  = runtime.store.list_events("agent")
    assert len(events) == 1
    assert events[0]["session_id"] is None
    runtime.shutdown()


def test_session_events_table_created_automatically(tmp_path: Path) -> None:
    runtime = MarkRuntime.local(store_path=tmp_path / "mem.db")
    # If the table doesn't exist, list_events would raise
    events = runtime.store.list_events("any-agent")
    assert events == []
    runtime.shutdown()


def test_session_events_persist_to_disk(tmp_path: Path) -> None:
    db = tmp_path / "mem.db"
    r1 = MarkRuntime.local(store_path=db)
    r1.store.log_event("agent", "observe")
    r1.shutdown()

    r2 = MarkRuntime.local(store_path=db)
    assert len(r2.store.list_events("agent")) == 1
    r2.shutdown()
