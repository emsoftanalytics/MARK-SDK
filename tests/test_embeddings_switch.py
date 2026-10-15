# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 MARK Contributors
"""Switching embedders on an existing store re-embeds it into the new space.

Covers the three shapes a switch can take after ``embedder="auto"`` starts
picking a different model: a new dimension known up front, a new model with
the *same* dimension (only the fingerprint can tell), and a provider that only
learns its dimension on the first call (fingerprint ``...:None``).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from mark.embeddings import HashEmbeddingProvider
from mark.memory.runtime import MarkRuntime

FACTS = [
    "The billing service runs on port 9443.",
    "Invoices are archived to cold storage after 90 days.",
    "The mobile app is written in Kotlin.",
    "Deployments happen every Tuesday at 14:00 UTC.",
    "The on-call rotation is managed in PagerDuty.",
]
KEY = "embedder_fingerprint:switch"


class _OtherModel(HashEmbeddingProvider):
    """A different embedding space: hash features with a salt, any dimension.

    ``lazy=True`` mimics HTTP/ONNX providers whose dimension is unknown until
    the first embed call.
    """

    calls = 0

    def __init__(self, dim: int, *, lazy: bool = False) -> None:
        super().__init__(dim=dim)
        self._known = not lazy

    @property
    def dim(self) -> int | None:
        return self._dim if self._known else None

    @property
    def fingerprint(self) -> str:
        return f"other-model:{self.dim}"

    def embed(self, text: str) -> list[float]:
        type(self).calls += 1
        self._known = True
        return super().embed("salt " + text)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(text) for text in texts]


def _seed(path: Path) -> list[str]:
    runtime = MarkRuntime.local(store_path=path)  # historical default: hash
    try:
        mem = runtime.memory("switch")
        ids = [mem.store_sync(fact) for fact in FACTS]
        assert mem.retrieve_sync("billing port").fragments[0].id == ids[0]
        assert runtime.store.get_meta(KEY) == "mark.hash-v2:256"
        return ids
    finally:
        runtime.shutdown()


@pytest.mark.parametrize("dim, lazy", [(64, False), (256, False), (64, True), (256, True)])
def test_switch_from_hash_reembeds_existing_store(tmp_path: Path, dim: int, lazy: bool) -> None:
    path = tmp_path / "switch.db"
    ids = _seed(path)
    expected = _OtherModel(dim)

    runtime = MarkRuntime.local(store_path=path, embedder=_OtherModel(dim, lazy=lazy))
    try:
        mem = runtime.memory("switch")
        result = mem.retrieve_sync("Which port does the billing service use?")
        assert result.fragments[0].id == ids[0]
        assert runtime.store.get_meta(KEY) == f"other-model:{dim}"
        stored = dict(runtime.store.embeddings_for_agent("switch"))
        assert set(stored) == set(ids)
        for fid, fact in zip(ids, FACTS):
            assert stored[fid] == pytest.approx(expected.embed(fact)), fact
    finally:
        runtime.shutdown()


def test_switch_happens_once_then_vectors_are_reused(tmp_path: Path) -> None:
    path = tmp_path / "once.db"
    _seed(path)
    calls: list[int] = []
    for _ in range(2):
        _OtherModel.calls = 0
        runtime = MarkRuntime.local(store_path=path, embedder=_OtherModel(256, lazy=True))
        try:
            runtime.memory("switch").retrieve_sync("cold storage")
        finally:
            runtime.shutdown()
        calls.append(_OtherModel.calls)
    # First process: probe + re-embed of every fact + query.
    assert calls[0] >= len(FACTS) + 1
    # Second process: the query only (stored vectors reused, no probe).
    assert calls[1] == 1


def test_switch_back_to_hash_restores_hash_vectors(tmp_path: Path) -> None:
    path = tmp_path / "back.db"
    ids = _seed(path)
    other = MarkRuntime.local(store_path=path, embedder=_OtherModel(64))
    other.memory("switch").retrieve_sync("kotlin")
    other.shutdown()

    runtime = MarkRuntime.local(store_path=path, embedder="hash")
    try:
        result = runtime.memory("switch").retrieve_sync("mobile app language")
        assert result.fragments[0].id == ids[2]
        stored = dict(runtime.store.embeddings_for_agent("switch"))
        assert stored[ids[2]] == pytest.approx(HashEmbeddingProvider().embed(FACTS[2]))
        assert runtime.store.get_meta(KEY) == "mark.hash-v2:256"
    finally:
        runtime.shutdown()
