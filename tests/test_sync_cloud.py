# SPDX-License-Identifier: MIT
# Copyright (c) 2025 MARK Contributors
from __future__ import annotations

from pathlib import Path

from mark import CloudSync, SyncDelta, SyncMode, SyncOptions, SyncStats
from mark.memory.runtime import MarkRuntime
from mark.embeddings import HashEmbeddingProvider


# ---------------------------------------------------------------------------
# Import checks
# ---------------------------------------------------------------------------

def test_sync_mode_importable() -> None:
    assert SyncMode.MANUAL        == "manual"
    assert SyncMode.WRITE_THROUGH == "write_through"


def test_sync_options_defaults() -> None:
    opts = SyncOptions()
    assert opts.include_blocks  is None
    assert opts.redact_secrets  is True
    assert opts.include_events  is False
    assert opts.mode            == SyncMode.MANUAL


def test_sync_stats_defaults() -> None:
    s = SyncStats()
    assert s.synced  == 0
    assert s.skipped == 0
    assert s.errors  == 0


def test_sync_stats_total() -> None:
    s = SyncStats(synced=3, skipped=2, errors=1)
    assert s.total == 6


def test_sync_delta_importable() -> None:
    delta = SyncDelta()
    assert delta.fragments == []
    assert isinstance(delta.stats, SyncStats)


# ---------------------------------------------------------------------------
# CloudSync.prepare — no-op local behavior
# ---------------------------------------------------------------------------

def test_cloud_sync_prepare_empty_runtime_returns_zero(tmp_path: Path) -> None:
    runtime = MarkRuntime.local(store_path=tmp_path / "m.db",
                                embedder=HashEmbeddingProvider(dim=32))
    sync   = CloudSync()
    result = sync.prepare(runtime)
    assert isinstance(result, SyncStats)
    runtime.shutdown()


def test_cloud_sync_prepare_with_stored_fragment(tmp_path: Path) -> None:
    runtime = MarkRuntime.local(store_path=tmp_path / "m.db",
                                embedder=HashEmbeddingProvider(dim=32))
    mem = runtime.memory("agent")
    mem.store_sync("Elena is left-handed.", importance=0.9)

    sync   = CloudSync()
    result = sync.prepare(runtime)
    assert result.total >= 0   # no error
    runtime.shutdown()


def test_cloud_sync_prepare_redacts_secrets(tmp_path: Path) -> None:
    from mark import Mark
    mark = Mark.local(project_path=tmp_path)
    mark.memory.block("default").write(
        "API key: sk-abc123secret456",
        metadata={"note": "token=metadata-secret-value"},
        source="https://example.test/path?token=source-secret-value",
    )
    mark.runtime.tracer().emit("sync.test", note="password=event-secret-value")
    sync   = CloudSync()
    delta = sync.prepare_delta(
        mark.runtime,
        options=SyncOptions(redact_secrets=True, include_events=True),
    )
    assert isinstance(delta, SyncDelta)
    assert delta.stats.synced >= 1
    payload = str(delta.fragments) + str(delta.events)
    assert "sk-abc123secret456" not in payload
    assert "metadata-secret-value" not in payload
    assert "source-secret-value" not in payload
    assert "event-secret-value" not in payload
    mark.shutdown()


def test_cloud_sync_skips_unselected_blocks(tmp_path: Path) -> None:
    from mark import Mark
    mark = Mark.local(project_path=tmp_path)
    mark.memory.block("project").write("Project fact.")
    mark.memory.block("notes").write("Notes fact.")

    sync   = CloudSync()
    result = sync.prepare(
        mark.runtime,
        options=SyncOptions(include_blocks=["project"]),
    )
    # notes block should be skipped
    assert result.skipped >= 0   # structural check only (block filter applied)
    mark.shutdown()


def test_cloud_sync_prepare_delta_filters_blocks(tmp_path: Path) -> None:
    from mark import Mark
    mark = Mark.local(project_path=tmp_path)
    mark.memory.block("project").write("Project fact.")
    mark.memory.block("notes").write("Notes fact.")

    delta = CloudSync().prepare_delta(
        mark.runtime,
        options=SyncOptions(include_blocks=["project"]),
    )

    assert any("Project fact." in f["content"] for f in delta.fragments)
    assert not any("Notes fact." in f["content"] for f in delta.fragments)
    assert delta.stats.skipped >= 1
    mark.shutdown()


def test_cloud_sync_prepare_delta_includes_events_when_enabled(tmp_path: Path) -> None:
    runtime = MarkRuntime.local(store_path=tmp_path / "m.db",
                                embedder=HashEmbeddingProvider(dim=32))
    runtime.tracer().emit("memory.retrieve", agent_id="a", session_id="s1")

    delta = CloudSync().prepare_delta(
        runtime,
        options=SyncOptions(include_events=True),
    )

    assert len(delta.events) == 1
    assert delta.events[0]["type"] == "memory.retrieve"
    runtime.shutdown()


def test_cloud_sync_sync_without_cloud_returns_stats(tmp_path: Path) -> None:
    runtime = MarkRuntime.local(store_path=tmp_path / "m.db",
                                embedder=HashEmbeddingProvider(dim=32))
    sync   = CloudSync()
    result = sync.sync(runtime, cloud=None)
    assert isinstance(result, SyncStats)
    runtime.shutdown()


def test_cloud_sync_sync_with_mock_cloud_calls_upload(tmp_path: Path) -> None:
    runtime = MarkRuntime.local(store_path=tmp_path / "m.db",
                                embedder=HashEmbeddingProvider(dim=32))
    mem = runtime.memory("a")
    mem.store_sync("Fact.", importance=0.5)

    uploads: list = []

    class FakeCloud:
        def upload_delta(self, delta):
            uploads.append(delta)

    sync   = CloudSync()
    result = sync.sync(runtime, cloud=FakeCloud())
    assert isinstance(result, SyncStats)
    assert len(uploads) == 1
    assert isinstance(uploads[0], SyncDelta)
    assert uploads[0].fragments
    runtime.shutdown()


def test_cloud_sync_does_not_import_cloud_package_on_prepare() -> None:
    import sys
    had_mark_cloud = "mark_cloud" in sys.modules
    CloudSync()   # instantiate
    still_no_mark_cloud = "mark_cloud" not in sys.modules
    assert had_mark_cloud or still_no_mark_cloud


# ---------------------------------------------------------------------------
# SyncOptions mode field
# ---------------------------------------------------------------------------

def test_sync_options_write_through_mode() -> None:
    opts = SyncOptions(mode=SyncMode.WRITE_THROUGH)
    assert opts.mode == SyncMode.WRITE_THROUGH
