# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 MARK Contributors
"""store_sync derives the memory kind from a leading label, like remember() does."""
from __future__ import annotations

from mark.embeddings import HashEmbeddingProvider
from mark.memory.runtime import MarkRuntime


def _memory():
    return MarkRuntime.local(store_path=":memory:", embedder=HashEmbeddingProvider(dim=32)).memory("k")


def test_store_sync_tags_the_kind_of_a_labelled_note() -> None:
    memory = _memory()
    fid = memory.store_sync("TODO: add retries to the ingest worker")
    assert "kind:todo" in memory.get(fid).tags
    plain = memory.store_sync("atlas-core runs on port 8080.")
    assert not any(t.startswith("kind:") for t in memory.get(plain).tags)


def test_an_explicit_kind_tag_wins() -> None:
    memory = _memory()
    fid = memory.store_sync("TODO: add retries", tags=["kind:decision"])
    assert [t for t in memory.get(fid).tags if t.startswith("kind:")] == ["kind:decision"]
