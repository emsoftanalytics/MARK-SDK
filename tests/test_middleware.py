from __future__ import annotations

from pathlib import Path
from typing import Any

from mark import (
    BaseMiddleware,
    CompressionMiddleware,
    GovernanceMiddleware,
    Mark,
    MediaContinuityMiddleware,
    MiddlewareContext,
    ObserveMiddleware,
    ObservabilityMiddleware,
    PublisherTrust,
    RecallMiddleware,
    SandboxMiddleware,
    SimpleWindowCompressor,
    SyncMiddleware,
    TrustBusMiddleware,
)


class RecorderMiddleware(BaseMiddleware):
    name = "recorder"

    def __init__(self) -> None:
        self.events: list[tuple[str, str, dict[str, Any]]] = []

    def before(self, context: MiddlewareContext) -> None:
        self.events.append(("before", context.operation, dict(context.payload)))

    def after(self, context: MiddlewareContext) -> None:
        self.events.append(("after", context.operation, dict(context.payload)))


def test_middleware_imports_from_public_package() -> None:
    from mark.middlewares import MiddlewareStack

    assert MiddlewareStack is not None
    assert RecallMiddleware(compress=True).name == "recall"
    assert GovernanceMiddleware().name == "governance"
    assert TrustBusMiddleware().name == "trust_bus"
    assert MediaContinuityMiddleware().name == "media_continuity"
    assert SandboxMiddleware().name == "sandbox"


def test_runtime_registers_and_runs_middleware_in_order(tmp_path: Path) -> None:
    recorder = RecorderMiddleware()

    with Mark.local(tmp_path, middleware=[RecallMiddleware(compress=True), recorder]) as mark:
        mem = mark.runtime.memory("coder")
        mem.store_sync("FastAPI handles API routes.")
        result = mem.retrieve_sync("Which framework?")

    assert result.fragments
    retrieve_before = [
        payload for phase, op, payload in recorder.events
        if phase == "before" and op == "retrieve"
    ][0]
    assert retrieve_before["compress"] is True


def test_observe_middleware_adds_source_tags_and_metadata(tmp_path: Path) -> None:
    with Mark.local(
        tmp_path,
        middleware=[
            ObserveMiddleware(
                source="middleware",
                tags=["workflow"],
                metadata={"kind": "decision"},
            )
        ],
    ) as mark:
        mem = mark.runtime.memory("planner")
        observed = mem.observe("Use FastAPI dependency injection.")
        stored = mark.runtime.store.get(observed.fragment_id)

    assert stored is not None
    assert stored.source == "middleware"
    assert "workflow" in stored.tags
    assert stored.metadata["kind"] == "decision"


def test_compression_middleware_configures_retrieval(tmp_path: Path) -> None:
    with Mark.local(
        tmp_path,
        middleware=[CompressionMiddleware(SimpleWindowCompressor(max_chars=12))],
    ) as mark:
        mem = mark.runtime.memory("writer")
        mem.store_sync(
            "Elena wears the red scarf through every warehouse scene.",
            importance=0.9,
        )
        result = mem.retrieve_sync("What does Elena wear?")

    assert result.was_compressed is True
    assert result.fragments[0].content.endswith("…")
    assert len(result.fragments[0].content) <= 15


def test_observability_middleware_emits_runtime_events(tmp_path: Path) -> None:
    with Mark.local(tmp_path, middleware=[ObservabilityMiddleware()]) as mark:
        mem = mark.runtime.memory("observer")
        mem.store_sync("The release workflow uses uv publish.")

        event_types = [event.type for event in mark.runtime.tracer().events()]

    assert "middleware.store.start" in event_types
    assert "middleware.store.finish" in event_types


def test_sync_middleware_prepares_delta_after_observe(tmp_path: Path) -> None:
    sync = SyncMiddleware()

    with Mark.local(tmp_path, middleware=[sync]) as mark:
        mark.runtime.memory("sync-agent").observe(
            "API key: sk-secret should be redacted before sync.",
            source="test",
        )

    assert sync.last_delta is not None
    assert sync.last_stats.synced >= 1
    contents = [fragment["content"] for fragment in sync.last_delta.fragments]
    assert not any("sk-secret" in content for content in contents)


def test_governance_middleware_quarantines_rejected_store(tmp_path: Path) -> None:
    with Mark.local(tmp_path, middleware=[GovernanceMiddleware()]) as mark:
        mem = mark.runtime.memory("governed")
        fragment_id = mem.store_sync("Traceback (most recent call last)")
        stored = mark.runtime.store.get(fragment_id)

    assert stored is not None
    assert stored.state == "quarantined"
    assert "mark:governance" in stored.tags
    assert stored.metadata["mark_governance"]["passed"] is False


def test_trust_bus_middleware_publishes_observations(tmp_path: Path) -> None:
    trust_bus = TrustBusMiddleware(trust=PublisherTrust.VERIFIED_AGENT, tags=["handoff"])

    with Mark.local(tmp_path, middleware=[trust_bus]) as mark:
        mark.runtime.memory("planner").observe("Planner chose FastAPI for the API.")
        messages = mark.runtime.trust_bus().all_messages()

    assert trust_bus.last_message is not None
    assert messages
    assert messages[-1].trust == PublisherTrust.VERIFIED_AGENT
    assert "handoff" in messages[-1].tags


def test_media_continuity_middleware_defaults_observe_and_retrieve(tmp_path: Path) -> None:
    recorder = RecorderMiddleware()

    with Mark.local(
        tmp_path,
        middleware=[MediaContinuityMiddleware(tags=["character:elena"]), recorder],
    ) as mark:
        mem = mark.runtime.memory("video-agent")
        observed = mem.observe(
            "Elena enters the North Warehouse wearing the red scarf.",
            session_id="season-01/scene-01",
        )
        stored = mark.runtime.store.get(observed.fragment_id)
        mem.retrieve_sync("What must stay consistent for Elena?")

    assert stored is not None
    assert "continuity" in stored.tags
    assert "character:elena" in stored.tags
    assert stored.metadata["mark_continuity"]["memory_type"] == "scene"
    retrieve_before = [
        payload for phase, op, payload in recorder.events
        if phase == "before" and op == "retrieve"
    ][0]
    assert retrieve_before["expand"] is True


def test_sandbox_middleware_blocks_local_execution_by_default(tmp_path: Path) -> None:
    sandbox = SandboxMiddleware()

    with Mark.local(tmp_path, middleware=[sandbox]) as mark:
        result = mark.runtime.middleware.run(
            "sandbox_execute",
            runtime=mark.runtime,
            payload={"code": "print('hello')"},
            handler=lambda context: {"ok": True},
        )

    assert result["ok"] is False
    assert result["sandbox_mode"] == "blocked/local-disabled"
