"""breadth #10: a network embedder outage never loses a write.

The memory is stored without an embedding (one warning per outage), recall
still finds it by keyword, and it is embedded once the embedder answers.
Configuration errors (4xx) still fail the write.
"""
from __future__ import annotations

import logging

import pytest

from mark.embeddings import EmbeddingHTTPError
from mark.embeddings.base import HashEmbeddingProvider
from mark.intelligence import pending_embeddings
from mark.memory.runtime import MarkRuntime

AGENT = "a"


class _Flaky(HashEmbeddingProvider):
    """A hash embedder that can be switched off like an unreachable server."""

    def __init__(self) -> None:
        super().__init__()
        self.down: int | None = None  # None: up; else the HTTP status (0: connection refused)
        self.calls = 0

    def _check(self) -> None:
        self.calls += 1
        if self.down is not None:
            raise EmbeddingHTTPError("could not connect to http://127.0.0.1:1 after 3 attempts",
                                     status=self.down or None)

    def embed_batch(self, texts):
        self._check()
        return [super(_Flaky, self).embed(text) for text in texts]

    def embed(self, text):
        self._check()
        return super().embed(text)


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    monkeypatch.setattr(pending_embeddings, "RETRY_SECONDS", 0.0)
    embedder = _Flaky()
    rt = MarkRuntime.local(store_path=tmp_path / "m.db", embedder=embedder)
    rt.memory(AGENT).store_sync("the gateway runs on port 8443")
    yield rt, embedder
    rt.shutdown()


def test_writes_during_an_outage_are_stored_and_found_by_keyword(runtime, caplog) -> None:
    rt, embedder = runtime
    memory = rt.memory(AGENT)
    embedder.down = 0
    with caplog.at_level(logging.WARNING, logger="mark.retrieval"):
        fid = memory.store_sync("zebra-42 handled the outage")
        other = memory.store_sync("zebra-43 took the second shift")
    assert fid and other and rt.store.count(AGENT) == 3
    assert caplog.text.count("embedder unavailable") == 1  # one warning per outage
    assert rt.store.get(fid).embedding is None  # pending marker
    hits = [f.id for f in memory.retrieve_sync("zebra-42").fragments]
    assert fid in hits  # keyword channel answers while vectors cannot
    embedder.down = None  # back: the next sync embeds the pending memories
    assert memory.retrieve_sync("zebra-42 outage").fragments[0].id == fid
    assert rt.store.get(fid).embedding is not None and fid in rt.index


def test_pending_memories_are_embedded_by_a_new_process(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(pending_embeddings, "RETRY_SECONDS", 0.0)
    embedder = _Flaky()
    rt = MarkRuntime.local(store_path=tmp_path / "m.db", embedder=embedder)
    embedder.down = 503
    fid = rt.memory(AGENT).store_sync("deferred while the server returned 503")
    rt.shutdown()
    rt = MarkRuntime.local(store_path=tmp_path / "m.db", embedder=_Flaky())
    assert rt.memory(AGENT).retrieve_sync("deferred server 503").fragments[0].id == fid
    assert rt.store.get(fid).embedding is not None
    rt.shutdown()


def test_configuration_errors_still_fail_the_write(runtime) -> None:
    rt, embedder = runtime
    embedder.down = 401
    with pytest.raises(EmbeddingHTTPError):
        rt.memory(AGENT).store_sync("never stored: bad API key")
    assert rt.store.count(AGENT) == 1


def test_no_retry_storm_during_an_outage(runtime, monkeypatch) -> None:
    rt, embedder = runtime
    monkeypatch.setattr(pending_embeddings, "RETRY_SECONDS", 60.0)
    embedder.down = 0
    memory = rt.memory(AGENT)
    memory.store_sync("first write in the outage")
    calls = embedder.calls
    for i in range(5):
        memory.store_sync(f"write {i} in the outage")
        memory.retrieve_sync(f"write {i}")
    assert embedder.calls == calls  # the outage cost one failed call, not one per operation
