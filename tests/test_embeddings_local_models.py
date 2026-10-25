# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 Somtochukwu Owen Ikeanyi
"""Optional local-model embedding providers (sentence-transformers, fastembed,
model2vec), faked so no downloads happen.
"""
from __future__ import annotations

import sys
import types

import pytest

from mark.embeddings import (
    FastEmbedProvider,
    Model2VecProvider,
    SentenceTransformerEmbeddingProvider,
)


# ── optional local model providers (faked; no downloads) ─────────────────────


class _FakeArray:
    def __init__(self, rows: list[list[float]]) -> None:
        self.rows = rows

    def tolist(self) -> list[list[float]]:
        return self.rows


def test_sentence_transformers_import_and_batch(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict] = []

    class SentenceTransformer:
        def __init__(self, name: str, **kwargs: object) -> None:
            self.name = name

        def get_sentence_embedding_dimension(self) -> int:
            return 2

        def encode(self, texts: list[str], **kwargs: object) -> _FakeArray:
            calls.append({"texts": texts, **kwargs})
            return _FakeArray([[1.0, 0.0] for _ in texts])

    monkeypatch.setitem(sys.modules, "sentence_transformers", types.SimpleNamespace(SentenceTransformer=SentenceTransformer))
    provider = SentenceTransformerEmbeddingProvider("tiny-model")
    assert provider.dim == 2
    assert provider.embed_batch(["a", "b"]) == [[1.0, 0.0], [1.0, 0.0]]
    assert calls[0]["normalize_embeddings"] is True and calls[0]["texts"] == ["a", "b"]
    assert provider.embed("c") == [1.0, 0.0]
    assert provider.fingerprint == "sentence-transformers:tiny-model:2"


def test_fastembed_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    class TextEmbedding:
        def __init__(self, model_name: str, **kwargs: object) -> None:
            self.model_name = model_name

        def embed(self, texts: list[str], batch_size: int = 256):
            for _ in texts:
                yield _FakeArray([3.0, 4.0]).rows

    monkeypatch.setitem(sys.modules, "fastembed", types.SimpleNamespace(TextEmbedding=TextEmbedding))
    provider = FastEmbedProvider()
    assert provider.embed_batch(["a", "b"]) == [pytest.approx([0.6, 0.8])] * 2
    assert provider.dim == 2
    assert provider.fingerprint == "fastembed:BAAI/bge-small-en-v1.5:2"


def test_model2vec_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    class StaticModel:
        dim = 2

        @classmethod
        def from_pretrained(cls, name: str, **kwargs: object) -> "StaticModel":
            return cls()

        def encode(self, texts: list[str]) -> _FakeArray:
            return _FakeArray([[0.0, 2.0] for _ in texts])

    monkeypatch.setitem(sys.modules, "model2vec", types.SimpleNamespace(StaticModel=StaticModel))
    provider = Model2VecProvider()
    assert provider.dim == 2
    assert provider.embed("x") == [0.0, 1.0]
    assert provider.fingerprint == "model2vec:minishlab/potion-base-8M:2"


@pytest.mark.parametrize(
    ("cls", "module", "extra"),
    [
        (FastEmbedProvider, "fastembed", "mark-sdk[fastembed]"),
        (Model2VecProvider, "model2vec", "mark-sdk[model2vec]"),
        (SentenceTransformerEmbeddingProvider, "sentence_transformers", "mark-sdk[sentence-transformers]"),
    ],
)
def test_optional_providers_name_their_extra(monkeypatch: pytest.MonkeyPatch, cls, module: str, extra: str) -> None:
    monkeypatch.setitem(sys.modules, module, None)
    with pytest.raises(ImportError, match=extra.replace("[", r"\[").replace("]", r"\]")):
        cls()
