# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 MARK Contributors
"""The store remembers its embedder (dev re-test major 3).

A Model2Vec store opened by the CLI or ``Mark.local(path)`` used the hash
embedder and silently re-embedded everything. ``Model2VecProvider`` below is
a deterministic stand-in (named like the real class, so the store records a
``model2vec:`` spec); ``_build_stored`` is patched to say whether the model
"is installed" here.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
from pathlib import Path

import pytest

import mark.runtime_embedder as stored_mod
from mark import Mark
from mark.cli import main
from mark.embeddings.base import EmbeddingProvider


class Model2VecProvider(EmbeddingProvider):
    model_name = "fake/potion"

    @property
    def dim(self) -> int:
        return 8

    @property
    def fingerprint(self) -> str:
        return "model2vec:fake/potion:8"

    def embed(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.lower().encode()).digest()
        vec = [b / 255.0 + 0.01 for b in digest[:8]]
        norm = math.sqrt(sum(v * v for v in vec))
        return [v / norm for v in vec]


def _vectors(mark: Mark) -> dict[str, list[float] | None]:
    return {f.id: f.embedding for f in mark.runtime.store.list_by_agent("__mark__", limit=None)}


@pytest.fixture()
def store_dir(tmp_path: Path) -> Path:
    with Mark.local(tmp_path, embedder=Model2VecProvider()) as mark:
        mark.memory.remember("The billing service runs on port 4444.")
        mark.memory.remember("Deploys happen on Fridays.")
        assert all(_vectors(mark).values())
        meta = json.loads(mark.runtime.store.get_meta(stored_mod.STORE_EMBEDDER_KEY))
    assert meta == {"spec": "model2vec:fake/potion", "fingerprint": "model2vec:fake/potion:8"}
    return tmp_path


def test_reopen_without_embedder_uses_the_stored_one(store_dir: Path, monkeypatch) -> None:
    monkeypatch.setattr(stored_mod, "_build_stored",
                        lambda spec: Model2VecProvider() if spec == "model2vec:fake/potion" else None)
    with Mark.local(store_dir) as mark:
        before = _vectors(mark)
        info = mark.memory.embedder_info()
        assert info.source == "store" and info.fingerprint == "model2vec:fake/potion:8"
        assert mark.memory.recall("billing port").fragments
        assert _vectors(mark) == before                       # nothing re-embedded


def test_unloadable_stored_embedder_opens_lexical_only_with_one_warning(store_dir: Path, monkeypatch,
                                                                        caplog) -> None:
    monkeypatch.setattr(stored_mod, "_build_stored", lambda spec: None)
    monkeypatch.setattr(stored_mod, "_warned", set())
    with caplog.at_level(logging.WARNING, logger="mark"):
        with Mark.local(store_dir) as mark:
            before = _vectors(mark)
            info = mark.memory.embedder_info()
            assert not info.semantic and "unavailable" in info.name
            top = mark.memory.recall("billing service port").fragments
            assert top and "4444" in top[0].content               # keyword search still answers
            new = mark.memory.remember("Standups are at 9:30.")
            vectors = _vectors(mark)
            assert {k: v for k, v in vectors.items() if k != new} == before   # untouched
    warnings = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert len(warnings) == 1 and "cannot be loaded" in warnings[0] and "model2vec:fake/potion" in warnings[0] and "pip install" in warnings[0]
    with Mark.local(store_dir) as mark:                          # still remembered, not downgraded
        meta = json.loads(mark.runtime.store.get_meta(stored_mod.STORE_EMBEDDER_KEY))
    assert meta["spec"] == "model2vec:fake/potion"


def test_cli_uses_the_stored_embedder(store_dir: Path, monkeypatch, capsys) -> None:
    monkeypatch.setattr(stored_mod, "_build_stored",
                        lambda spec: Model2VecProvider() if spec == "model2vec:fake/potion" else None)
    assert main(["--no-server", "--db", str(store_dir), "stats"]) == 0
    assert "model2vec:fake/potion (semantic, store)" in capsys.readouterr().out


def test_explicit_other_embedder_re_embeds_and_becomes_the_stored_one(store_dir: Path) -> None:
    with Mark.local(store_dir, embedder="hash") as mark:
        assert mark.memory.recall("billing port").fragments
        meta = json.loads(mark.runtime.store.get_meta(stored_mod.STORE_EMBEDDER_KEY))
    assert meta["spec"] == "hash"
    with Mark.local(store_dir) as mark:
        assert mark.memory.embedder_info().source == "store"


def test_new_store_records_the_default_and_legacy_stores_are_recognised(tmp_path: Path) -> None:
    with Mark.local(tmp_path / "new") as mark:
        assert json.loads(mark.runtime.store.get_meta(stored_mod.STORE_EMBEDDER_KEY))["spec"] == "hash"
    with Mark.local(tmp_path / "old", embedder=Model2VecProvider()) as mark:
        mark.memory.remember("x marks the spot")
        mark.memory.recall("spot")
        store = mark.runtime.store
        with store._writing(structural=False) as conn:          # what an a9 store looks like
            conn.execute("DELETE FROM schema_info WHERE key = ?", (stored_mod.STORE_EMBEDDER_KEY,))
        assert stored_mod.stored_embedder(store) == {"spec": "model2vec:fake/potion",
                                                     "fingerprint": "model2vec:fake/potion:8"}
