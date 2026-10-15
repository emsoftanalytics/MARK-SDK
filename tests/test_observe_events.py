# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 MARK Contributors
"""mark.observe() events and delegation: ObserveEvent schema, HOOK_OBSERVE_EVENT
emission with and without an LLM, and SimpleMemory.observe().
"""
from __future__ import annotations

from pathlib import Path

from mark import Mark, ObserveEvent, ObserveMiddleware
from mark.embeddings import HashEmbeddingProvider
from mark.memory.runtime import MarkRuntime
from mark.plugins import HOOK_OBSERVE_EVENT

from _helpers_observe import _ELENA_JSON_RESPONSE, _FakeLLM


# ---------------------------------------------------------------------------
# HOOK_OBSERVE_EVENT — schema and emission
# ---------------------------------------------------------------------------

def test_observe_event_schema_fields(tmp_path: Path) -> None:
    llm = _FakeLLM(_ELENA_JSON_RESPONSE)
    runtime = MarkRuntime.local(store_path=tmp_path / "mem.db",
                                embedder=HashEmbeddingProvider(dim=32),
                                llm=llm,
                                middleware=[ObserveMiddleware()])
    mem = runtime.memory("director")
    result = mem.observe("Elena enters the warehouse.", session_id="ep-01")

    event = ObserveEvent(
        agent_id    = "director",
        fragment_id = result.fragment_id,
        content     = "Elena enters the warehouse.",
        session_id  = "ep-01",
        node_labels = [n.label for n in result.nodes],
        edge_count  = len(result.edges),
        inferred    = result.inferred,
    )
    assert event.agent_id == "director"
    assert event.inferred is True
    assert "Elena" in event.node_labels
    assert event.edge_count >= 3   # deterministic + LLM may produce more
    assert event.timestamp is not None
    runtime.shutdown()


def test_hook_observe_event_fires(tmp_path: Path) -> None:
    """Registering a HOOK_OBSERVE_EVENT plugin causes it to receive ObserveEvent."""
    from mark.plugins import MarkCorePlugin, PluginRegistry

    received: list[ObserveEvent] = []

    class ObservatoryPlugin(MarkCorePlugin):
        @property
        def name(self) -> str: return "test-observatory"
        @property
        def version(self) -> str: return "0.0.1"
        @property
        def provides(self) -> set[str]: return {HOOK_OBSERVE_EVENT}
        def get_hook(self, hook_name: str):
            if hook_name == HOOK_OBSERVE_EVENT:
                return lambda event: received.append(event)
            raise KeyError(hook_name)

    llm = _FakeLLM(_ELENA_JSON_RESPONSE)
    runtime = MarkRuntime.local(
        store_path=tmp_path / "mem.db",
        embedder=HashEmbeddingProvider(dim=32),
        plugins=[ObservatoryPlugin()],
        llm=llm,
        middleware=[ObserveMiddleware()],
    )
    mem = runtime.memory("director")
    mem.observe("Elena enters the warehouse.", session_id="ep-01")

    assert len(received) == 1
    event = received[0]
    assert isinstance(event, ObserveEvent)
    assert event.agent_id == "director"
    assert event.session_id == "ep-01"
    assert event.inferred is True
    runtime.shutdown()


def test_hook_observe_event_still_fires_without_llm(tmp_path: Path) -> None:
    """Observatory plugin receives events even when no LLM is configured."""
    from mark.plugins import MarkCorePlugin

    received: list[ObserveEvent] = []

    class ObservatoryPlugin(MarkCorePlugin):
        @property
        def name(self) -> str: return "test-observatory"
        @property
        def version(self) -> str: return "0.0.1"
        @property
        def provides(self) -> set[str]: return {HOOK_OBSERVE_EVENT}
        def get_hook(self, hook_name: str):
            return lambda event: received.append(event)

    runtime = MarkRuntime.local(
        store_path=tmp_path / "mem.db",
        plugins=[ObservatoryPlugin()],
    )
    mem = runtime.memory("director")
    mem.observe("The sky is blue.")

    assert len(received) == 1
    assert received[0].inferred is False
    runtime.shutdown()


# ---------------------------------------------------------------------------
# SimpleMemory.observe() delegation
# ---------------------------------------------------------------------------

def test_simple_memory_observe_without_llm(tmp_path: Path) -> None:
    """Deterministic extractor runs on simple memory observe without LLM."""
    with Mark.local(project_path=tmp_path, middleware=[ObserveMiddleware()]) as mark:
        result = mark.memory.observe(
            "Elena enters the warehouse.",
            session_id="ep-01",
            importance=0.8,
        )
        assert result.fragment_id
        # Deterministic extractor finds Elena (subject) and warehouse (location)
        assert result.inferred is True
        assert result.session_id == "ep-01"


def test_simple_memory_observe_tagged_as_observe_block(tmp_path: Path) -> None:
    with Mark.local(project_path=tmp_path) as mark:
        result = mark.memory.observe("Some observation.")
        stored = mark.runtime.store.get(result.fragment_id)
        assert stored is not None
        assert "block:observe" in stored.tags
