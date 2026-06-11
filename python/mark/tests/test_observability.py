# SPDX-License-Identifier: MIT
# Copyright (c) 2025 MARK Contributors
from __future__ import annotations

import json
from pathlib import Path

import pytest

from mark import (
    EventReplayer,
    LocalTracer,
    RuntimeEvent,
    RuntimeEventLog,
    SessionTrace,
)
from mark.embeddings import HashEmbeddingProvider
from mark.memory.runtime import MarkRuntime


# ---------------------------------------------------------------------------
# RuntimeEvent
# ---------------------------------------------------------------------------

def test_runtime_event_is_json_serializable() -> None:
    event = RuntimeEvent(type="memory.retrieve", payload={"query": "Elena"})
    data  = {"type": event.type, "payload": event.payload,
             "created_at": event.created_at.isoformat()}
    assert json.dumps(data)   # must not raise


def test_runtime_event_has_timestamp() -> None:
    event = RuntimeEvent(type="memory.retrieve")
    assert event.created_at is not None


def test_runtime_event_payload_defaults_to_empty() -> None:
    event = RuntimeEvent(type="agent.run.start")
    assert event.payload == {}


def test_runtime_event_log_append_and_list() -> None:
    log = RuntimeEventLog()
    log.append("memory.retrieve", query="test")
    log.append("agent.run.finish", agent_id="a")
    events = log.list()
    assert len(events) == 2
    assert events[0].type == "memory.retrieve"
    assert events[1].type == "agent.run.finish"


# ---------------------------------------------------------------------------
# LocalTracer — in-memory
# ---------------------------------------------------------------------------

def test_local_tracer_emit_returns_event() -> None:
    tracer = LocalTracer()
    event  = tracer.emit("memory.retrieve", query="scarf")
    assert isinstance(event, RuntimeEvent)
    assert event.type == "memory.retrieve"
    assert event.payload["query"] == "scarf"


def test_local_tracer_events_returns_all_emitted() -> None:
    tracer = LocalTracer()
    tracer.emit("agent.run.start",  agent_id="a")
    tracer.emit("agent.run.finish", agent_id="a")
    assert len(tracer.events()) == 2


def test_local_tracer_clear_resets_events() -> None:
    tracer = LocalTracer()
    tracer.emit("memory.retrieve")
    tracer.clear()
    assert tracer.events() == []


def test_local_tracer_preserves_all_event_types() -> None:
    tracer = LocalTracer()
    types  = [
        "memory.block.created", "memory.record.written", "memory.retrieve",
        "agent.run.start", "agent.run.finish", "skill.run",
        "sync.start", "sync.finish",
        "memory.observe", "memory.promote", "memory.prune",
        "graph.activate", "context.compress", "gap.detected",
        "governance.decision", "bus.publish", "bus.reconcile", "benchmark.run",
    ]
    for t in types:
        tracer.emit(t)
    assert len(tracer.events()) == len(types)


# ---------------------------------------------------------------------------
# LocalTracer — JSONL file
# ---------------------------------------------------------------------------

def test_local_tracer_writes_jsonl_file(tmp_path: Path) -> None:
    path   = tmp_path / "events.jsonl"
    tracer = LocalTracer(path=path)
    tracer.emit("memory.retrieve", query="Elena")
    tracer.emit("agent.run.start")

    lines = path.read_text().strip().split("\n")
    assert len(lines) == 2
    assert json.loads(lines[0])["type"] == "memory.retrieve"


def test_local_tracer_jsonl_line_has_required_keys(tmp_path: Path) -> None:
    path   = tmp_path / "ev.jsonl"
    tracer = LocalTracer(path=path)
    tracer.emit("skill.run", skill="web_search")

    data = json.loads(path.read_text().strip())
    assert "type" in data
    assert "payload" in data
    assert "created_at" in data


def test_local_tracer_appends_across_instances(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    LocalTracer(path=path).emit("event.one")
    LocalTracer(path=path).emit("event.two")
    lines = [l for l in path.read_text().strip().split("\n") if l]
    assert len(lines) == 2


def test_local_tracer_creates_parent_dirs(tmp_path: Path) -> None:
    path   = tmp_path / "nested" / "dir" / "events.jsonl"
    tracer = LocalTracer(path=path)
    tracer.emit("test.event")
    assert path.exists()


# ---------------------------------------------------------------------------
# SessionTrace
# ---------------------------------------------------------------------------

def test_session_trace_filters_by_session_id() -> None:
    tracer = LocalTracer()
    tracer.emit("memory.retrieve", session_id="ep-01", query="a")
    tracer.emit("memory.retrieve", session_id="ep-02", query="b")

    st = SessionTrace(tracer, "ep-01")
    assert len(st.events()) == 1
    assert st.events()[0].payload["query"] == "a"


def test_session_trace_emit_auto_tags_session_id() -> None:
    tracer = LocalTracer()
    st     = SessionTrace(tracer, "ep-03")
    st.emit("memory.observe", content="Elena enters.")
    events = st.events()
    assert len(events) == 1
    assert events[0].payload["session_id"] == "ep-03"


def test_session_trace_filter_by_event_type() -> None:
    tracer = LocalTracer()
    st     = SessionTrace(tracer, "ep-01")
    st.emit("memory.retrieve", query="q1")
    st.emit("memory.observe",  content="c1")
    st.emit("memory.retrieve", query="q2")
    retrieves = st.events(event_type="memory.retrieve")
    assert len(retrieves) == 2
    observes  = st.events(event_type="memory.observe")
    assert len(observes) == 1


def test_session_trace_summary() -> None:
    tracer = LocalTracer()
    st     = SessionTrace(tracer, "ep-01")
    st.emit("memory.retrieve")
    st.emit("memory.retrieve")
    st.emit("memory.observe")
    s = st.summary()
    assert s["total"] == 3
    assert s["by_type"]["memory.retrieve"] == 2
    assert s["session_id"] == "ep-01"


def test_session_trace_empty_when_no_matching_events() -> None:
    tracer = LocalTracer()
    tracer.emit("memory.retrieve", session_id="other")
    st = SessionTrace(tracer, "ep-99")
    assert st.events() == []


# ---------------------------------------------------------------------------
# EventReplayer
# ---------------------------------------------------------------------------

def test_event_replayer_load_empty_when_file_missing(tmp_path: Path) -> None:
    r = EventReplayer(tmp_path / "nonexistent.jsonl")
    assert r.load() == []


def test_event_replayer_load_events(tmp_path: Path) -> None:
    path   = tmp_path / "events.jsonl"
    tracer = LocalTracer(path=path)
    tracer.emit("memory.retrieve", query="a")
    tracer.emit("skill.run",       skill="web")

    events = EventReplayer(path).load()
    assert len(events) == 2
    assert events[0].type == "memory.retrieve"
    assert events[1].type == "skill.run"


def test_event_replayer_preserves_payload(tmp_path: Path) -> None:
    path   = tmp_path / "events.jsonl"
    LocalTracer(path=path).emit("memory.retrieve", query="Elena", count=3)
    events = EventReplayer(path).load()
    assert events[0].payload["query"] == "Elena"
    assert events[0].payload["count"] == 3


def test_event_replayer_replay_calls_handler(tmp_path: Path) -> None:
    path   = tmp_path / "events.jsonl"
    tracer = LocalTracer(path=path)
    tracer.emit("agent.run.start")
    tracer.emit("agent.run.finish")

    seen: list[str] = []
    count = EventReplayer(path).replay(lambda e: seen.append(e.type))
    assert count == 2
    assert seen == ["agent.run.start", "agent.run.finish"]


def test_event_replayer_replay_preserves_order(tmp_path: Path) -> None:
    path   = tmp_path / "events.jsonl"
    tracer = LocalTracer(path=path)
    for i in range(5):
        tracer.emit("step", n=i)

    events = EventReplayer(path).load()
    assert [e.payload["n"] for e in events] == list(range(5))


def test_event_replayer_skips_invalid_lines(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    path.write_text('{"type":"ok","payload":{},"created_at":"2025-01-01T00:00:00+00:00"}\nnot json\n')
    events = EventReplayer(path).load()
    assert len(events) == 1
    assert events[0].type == "ok"


# ---------------------------------------------------------------------------
# Runtime integration
# ---------------------------------------------------------------------------

def test_runtime_exposes_local_tracer(tmp_path: Path) -> None:
    tracer = LocalTracer()
    runtime = MarkRuntime.local(
        store_path=tmp_path / "m.db",
        embedder=HashEmbeddingProvider(dim=32),
        tracer=tracer,
    )

    assert runtime.tracer() is tracer
    runtime.tracer().emit("agent.run.start", session_id="s1")
    assert len(runtime.session_trace("s1").events()) == 1
    runtime.shutdown()


def test_runtime_maintenance_emits_trace_events(tmp_path: Path) -> None:
    runtime = MarkRuntime.local(
        store_path=tmp_path / "m.db",
        embedder=HashEmbeddingProvider(dim=32),
    )
    runtime.working_memory("trace-agent").store(
        "Traceable memory.",
        ttl_seconds=9999,
        importance=0.9,
    )

    runtime.run_cycle("trace-agent")
    types = [event.type for event in runtime.tracer().events()]

    assert "memory.consolidate.start" in types
    assert "memory.consolidate.finish" in types
    assert "memory.prune" in types
    runtime.shutdown()
