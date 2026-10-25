# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 Somtochukwu Owen Ikeanyi
"""After an embedder outage, one recall never stalls on the whole embedding backlog."""
from __future__ import annotations

import time

import pytest

from mark.embeddings import EmbeddingHTTPError
from mark.embeddings.base import HashEmbeddingProvider
from mark.intelligence import pending_embeddings
from mark.memory.runtime import MarkRuntime

AGENT = "a"


class _SlowFlaky(HashEmbeddingProvider):
    """Down (connection refused) or up but slow: ``delay`` seconds per text, like a contended server."""

    def __init__(self) -> None:
        super().__init__()
        self.down = False
        self.delay = 0.0
        self.texts = 0

    def embed_batch(self, texts):
        if self.down:
            raise EmbeddingHTTPError("could not connect", status=None)
        time.sleep(self.delay * len(texts))
        self.texts += len(texts)
        return [HashEmbeddingProvider.embed(self, text) for text in texts]

    def embed(self, text):
        return self.embed_batch([text])[0]


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    monkeypatch.setattr(pending_embeddings, "RETRY_SECONDS", 0.0)
    monkeypatch.setattr(pending_embeddings, "CATCHUP_SECONDS", 0.3)
    embedder = _SlowFlaky()
    rt = MarkRuntime.local(store_path=tmp_path / "m.db", embedder=embedder)
    rt.memory(AGENT).store_sync("the gateway runs on port 8443")
    yield rt, embedder
    rt.shutdown()


def test_recall_after_an_outage_embeds_the_backlog_in_bounded_steps(runtime) -> None:
    rt, embedder = runtime
    memory = rt.memory(AGENT)
    embedder.down = True
    for i in range(120):
        memory.store_sync(f"outage note {i}: zebra-{i} handled the queue")
    embedder.down, embedder.delay, embedder.texts = False, 0.01, 0  # back, but 10 ms per text
    start = time.monotonic()
    result = memory.retrieve_sync("zebra-7 queue")
    assert time.monotonic() - start < 1.0  # the full backlog would take 1.2 s
    assert result.fragments  # keyword search answers meanwhile
    assert 0 < embedder.texts < 120
    for _ in range(20):  # later recalls finish the backlog
        memory.retrieve_sync("zebra-7 queue")
        if embedder.texts >= 120:
            break
    assert embedder.texts >= 120
    before = embedder.texts
    memory.retrieve_sync("zebra-8 queue")
    assert embedder.texts - before <= 1  # nothing left but the query itself
