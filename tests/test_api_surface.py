# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 Somtochukwu Owen Ikeanyi
"""D4 embedder visibility and D5 curated top-level exports."""
from __future__ import annotations

import logging
import subprocess
import sys

import pytest

import mark
from mark.embeddings import auto
from mark.embeddings.base import HashEmbeddingProvider
from mark.embeddings.info import embedder_info
from mark.errors import EmbedderUnavailableError, MarkError


def test_embedder_info_reports_name_dim_semantic_and_source(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(auto.ENV_VAR, raising=False)
    monkeypatch.setattr(auto, "_LOCAL_CANDIDATES", [])
    default = embedder_info()
    assert (default.name, default.dim, default.semantic, default.source) == ("hash", 256, False, "default")
    assert embedder_info("hash:64").source == "spec" and embedder_info("auto").source == "auto"
    assert embedder_info(HashEmbeddingProvider()).source == "custom"
    monkeypatch.setenv(auto.ENV_VAR, "hash:32")
    assert (embedder_info().dim, embedder_info().source) == (32, "env")
    assert "lexical" in embedder_info().as_text() and embedder_info().to_dict()["dim"] == 32


def test_runtime_objects_expose_the_active_embedder(tmp_path) -> None:
    with mark.Mark.local(tmp_path) as instance:
        assert instance.embedder_info().name == "hash"
        assert instance.memory.embedder_info() == instance.runtime.memory("x").embedder_info()


def test_auto_fallback_to_hash_warns_once(monkeypatch: pytest.MonkeyPatch, caplog) -> None:
    monkeypatch.setattr(auto, "_LOCAL_CANDIDATES", [])
    monkeypatch.setattr(auto, "_warned_fallback", False)
    with caplog.at_level(logging.WARNING, logger="mark.embeddings"):
        auto.auto_embedder("auto")
        auto.auto_embedder("auto")
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1 and "mark-sdk[model2vec]" in warnings[0].getMessage()


def test_model_download_errors_mention_hf_home(monkeypatch: pytest.MonkeyPatch) -> None:
    def denied() -> None:
        raise PermissionError("[Errno 13] Permission denied: '/root/.cache/huggingface'")

    with pytest.raises(EmbedderUnavailableError, match="HF_HOME") as info:
        auto._loading("minishlab/potion-base-8M", denied)
    assert isinstance(info.value, (MarkError, RuntimeError))
    with pytest.raises(ImportError):  # a missing extra keeps its own install hint
        auto._loading("m", lambda: (_ for _ in ()).throw(ImportError("pip install mark-sdk[x]")))


def test_top_level_exports_are_curated_and_old_names_still_work() -> None:
    assert len(mark.__all__) <= 30 and "Mark" in mark.__all__ and "embedder_info" in mark.__all__
    for name in mark.__all__:
        assert getattr(mark, name) is not None
    with pytest.warns(DeprecationWarning, match="mark.middlewares.observe"):
        from mark import ObserveMiddleware  # noqa: F401
    assert "ObserveMiddleware" not in dir(mark)


def test_import_mark_stays_lazy() -> None:
    code = ("import sys, mark; heavy = [m for m in ('mark.memory', 'mark.runtime', 'pydantic', 'numpy')"
            " if m in sys.modules]; print(heavy)")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    assert out.strip() == "[]"
