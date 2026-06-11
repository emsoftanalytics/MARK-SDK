# SPDX-License-Identifier: MIT
# Copyright (c) 2025 MARK Contributors
#
# Import-surface smoke test.
# Uses only the public API exposed at the package boundary — no internal imports.
# Validates that the installed package works as documented in README.md.
from __future__ import annotations

from pathlib import Path


def test_mark_local_full_public_api(tmp_path: Path) -> None:
    """Mark.local() creates a working runtime via the public API only."""
    from mark import Mark

    with Mark.local(project_path=tmp_path) as mark:
        # Config
        assert mark.config.store_path == tmp_path / ".mark"
        assert (tmp_path / ".mark" / "memory.db").exists()

        # Simple memory API
        block = mark.memory.block("smoke", kind="test")
        rec = block.write("MARK smoke test record.", importance=0.9, confidence=0.9)
        assert rec.content == "MARK smoke test record."
        assert rec in block.records()

        bundle = mark.memory.retrieve("smoke test")
        assert "smoke test" in bundle.as_text().lower() or "MARK" in bundle.as_text()

        # Runtime engine is accessible
        assert mark.runtime is not None

        # Policies and skills registries
        assert "default" in mark.policies.names()
        assert "echo" in mark.skills.names()


def test_mark_runtime_direct_import(tmp_path: Path) -> None:
    """MarkRuntime is importable directly and works independently."""
    from mark import MarkRuntime

    runtime = MarkRuntime.local(store_path=tmp_path / "rt.db")
    mem = runtime.memory("smoke-agent")
    fid = mem.store_sync("Direct runtime smoke record.", importance=0.8)
    result = mem.retrieve_sync("smoke record")

    assert fid
    assert any("smoke record" in f.content for f in result.fragments)
    runtime.shutdown()


def test_all_public_types_importable() -> None:
    """Every type documented in the public API surface is importable."""
    from mark import (
        ContextualCompressor,
        DeterministicExtractor,
        ExtractionMerger,
        ExtractionResult,
        ExtractedEntity,
        ExtractedRelation,
        LLMContextualCompressor,
        LLMProvider,
        LLMStructuredExtractor,
        Mark,
        MarkRuntime,
        NoopCompressor,
        SimpleBlock,
        SimpleMemory,
        SimpleWindowCompressor,
        StructuredExtraction,
    )
    from mark.types import (
        MemoryFragment,
        MemoryScope,
        MemoryState,
        MemoryTier,
        MemoryBlock,
        MemoryEdge,
        MemoryNode,
        GapReport,
        GapSeverity,
        SourceAttribution,
    )
    from mark.plugins import (
        HOOK_SCORING,
        HOOK_POLICY_ENGINE,
        HOOK_PLASTICITY,
        HOOK_CONSOLIDATION,
        HOOK_RL_POLICY,
        HOOK_GOVERNANCE,
        HOOK_SYNC,
        MarkCorePlugin,
        PluginRegistry,
    )
    from mark.plasticity import (
        ExponentialDecay,
        HebbianReinforcement,
        EdgeCoActivation,
        MemoryPruner,
        StaticRouter,
    )
    from mark.rl import (
        ExperienceTuple,
        ReplayBuffer,
        RewardSignal,
        GovernanceSignalCollector,
    )

    # Just verify they imported — the names are the assertions
    assert Mark is not None
    assert MarkRuntime is not None
    assert ContextualCompressor is not None
    assert DeterministicExtractor is not None
    assert ExtractionMerger is not None
    assert ExtractionResult is not None
    assert ExtractedEntity is not None
    assert ExtractedRelation is not None
    assert LLMContextualCompressor is not None
    assert LLMProvider is not None
    assert LLMStructuredExtractor is not None
    assert NoopCompressor is not None
    assert SimpleWindowCompressor is not None
    assert StructuredExtraction is not None
    assert MemoryFragment is not None
    assert PluginRegistry is not None
    assert MemoryPruner is not None
    assert ReplayBuffer is not None


def test_adapter_public_surface_importable_without_optional_frameworks() -> None:
    """Bundled adapter namespaces import without installing framework extras."""
    from mark.adapters import BackendResult, LocalMarkBackend, MarkBackend
    from mark.adapters.langchain import MarkAgentMiddleware, MarkMemoryMiddleware, create_mark_tools
    from mark.adapters.langgraph import MarkState

    assert BackendResult is not None
    assert LocalMarkBackend is not None
    assert MarkBackend is not None
    assert MarkAgentMiddleware is not None
    assert MarkMemoryMiddleware is not None
    assert create_mark_tools is not None
    assert MarkState is not None


def test_local_adapter_backend_round_trip(tmp_path: Path) -> None:
    """LocalMarkBackend can write and retrieve through public MARK APIs."""
    import asyncio

    from mark import Mark
    from mark.adapters import LocalMarkBackend

    async def run() -> None:
        with Mark.local(project_path=tmp_path) as mark:
            backend = LocalMarkBackend(mark, default_agent_id="smoke-agent")
            written = await backend.write("Bundled adapter smoke memory.", importance=0.8)
            retrieved = await backend.retrieve("adapter smoke memory")

            assert written.ok, written.error
            assert retrieved.ok, retrieved.error
            if hasattr(retrieved.value, "as_text"):
                text = retrieved.value.as_text()
            else:
                text = retrieved.value.as_context()
            assert "adapter smoke" in text.lower()

    asyncio.run(run())
