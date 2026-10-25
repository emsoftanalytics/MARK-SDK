# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 Somtochukwu Owen Ikeanyi
"""auto_embedder: selection order, spec / MARK_EMBEDDER parsing, no implicit network.

Everything here is offline: optional packages are faked or reported as not
installed, and any HTTP or socket use fails the test. This machine may run an
Ollama server, so "no implicit network" is enforced, not assumed.
"""
from __future__ import annotations

import logging
import socket

import pytest

from mark.embeddings import (
    EmbeddingProvider,
    HashEmbeddingProvider,
    OllamaEmbeddingProvider,
    OpenAICompatibleEmbeddingProvider,
    auto,
    auto_embedder,
    resolve_embedder,
)
from mark.memory.runtime import MarkRuntime


class _Fake(HashEmbeddingProvider):
    """Stand-in local model with its own embedding-space identity."""

    def __init__(self, name: str) -> None:
        super().__init__(dim=32)
        self.name = name

    @property
    def fingerprint(self) -> str:
        return f"fake:{self.name}:32"


@pytest.fixture(autouse=True)
def _offline(monkeypatch: pytest.MonkeyPatch) -> None:
    """No env override, no optional packages, and any network use is an error."""
    monkeypatch.delenv(auto.ENV_VAR, raising=False)
    monkeypatch.delenv(auto.API_KEY_ENV_VAR, raising=False)
    monkeypatch.setattr(auto, "_installed", lambda module: False)

    def _no_network(*args: object, **kwargs: object) -> None:
        raise AssertionError("auto_embedder must not touch the network")

    monkeypatch.setattr(socket, "create_connection", _no_network)
    monkeypatch.setattr("mark.embeddings._http.post_json", _no_network)


# ── Selection order ──────────────────────────────────────────────────────────


def test_nothing_installed_is_exactly_the_hash_default() -> None:
    provider = auto_embedder()
    assert type(provider) is HashEmbeddingProvider
    assert provider.fingerprint == HashEmbeddingProvider().fingerprint == "mark.hash-v3:256"
    assert provider.embed("same vector") == HashEmbeddingProvider().embed("same vector")


def test_real_builders_skip_missing_packages() -> None:
    assert auto._try_model2vec(auto.DEFAULT_MODEL2VEC_MODEL, True) is None
    assert auto._try_fastembed(auto.DEFAULT_FASTEMBED_MODEL, True) is None


def test_first_available_candidate_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(auto, "_LOCAL_CANDIDATES", [
        ("model2vec", "m", lambda model, dl: _Fake("m2v")),
        ("fastembed", "f", lambda model, dl: _Fake("fe")),
    ])
    assert auto_embedder().fingerprint == "fake:m2v:32"


def test_unavailable_or_broken_candidates_fall_through(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    def broken(model: str, dl: bool) -> EmbeddingProvider:
        raise RuntimeError("corrupt install")

    monkeypatch.setattr(auto, "_LOCAL_CANDIDATES", [
        ("model2vec", "m", lambda model, dl: None),
        ("fastembed", "f", broken),
    ])
    with caplog.at_level(logging.INFO, logger="mark.embeddings"):
        provider = auto_embedder()
    assert type(provider) is HashEmbeddingProvider
    assert "fastembed unavailable" in caplog.text
    assert "using mark.hash-v3:256" in caplog.text


def test_allow_download_is_passed_to_candidates(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[bool] = []
    monkeypatch.setattr(auto, "_LOCAL_CANDIDATES", [("x", "m", lambda model, dl: seen.append(dl))])
    auto_embedder()
    auto_embedder(allow_download=True)
    assert seen == [False, True]


def test_model2vec_uncached_is_skipped_unless_download_allowed(monkeypatch: pytest.MonkeyPatch) -> None:
    built: list[tuple[str, str | None]] = []

    class Model2VecProvider(_Fake):
        def __init__(self, model: str, *, local_path: str | None = None) -> None:
            super().__init__(model)
            built.append((model, local_path))

    monkeypatch.setattr(auto, "_installed", lambda module: module == "model2vec")
    monkeypatch.setattr("mark.embeddings.model2vec.Model2VecProvider", Model2VecProvider)
    monkeypatch.setattr(auto, "_cached_model2vec_path", lambda model: None)
    assert type(auto_embedder()) is HashEmbeddingProvider
    assert built == []

    assert auto_embedder(allow_download=True).fingerprint == f"fake:{auto.DEFAULT_MODEL2VEC_MODEL}:32"
    monkeypatch.setattr(auto, "_cached_model2vec_path", lambda model: "/cache/snap")
    auto_embedder()
    assert built[-1] == (auto.RETRIEVAL_MODEL2VEC_MODEL, "/cache/snap")  # best cached model wins


def test_fastembed_offline_load_and_dimension_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    kwargs_seen: list[dict] = []

    class FastEmbedProvider(_Fake):
        def __init__(self, model: str, **kwargs: object) -> None:
            super().__init__(model)
            kwargs_seen.append(kwargs)
            self.probed = 0

        def embed(self, text: str) -> list[float]:
            self.probed += 1
            return super().embed(text)

    monkeypatch.setattr(auto, "_installed", lambda module: module == "fastembed")
    monkeypatch.setattr("mark.embeddings.fastembed.FastEmbedProvider", FastEmbedProvider)
    provider = auto_embedder()
    assert isinstance(provider, FastEmbedProvider) and provider.probed == 1
    assert kwargs_seen == [{"local_files_only": True}]
    auto_embedder(allow_download=True)
    assert kwargs_seen[-1] == {}


def test_cached_retrieval_model_preferred_but_never_downloaded(monkeypatch: pytest.MonkeyPatch) -> None:
    class Model2VecProvider(_Fake):
        def __init__(self, model: str, *, local_path: str | None = None) -> None:
            super().__init__(model)

    monkeypatch.setattr(auto, "_installed", lambda module: module == "model2vec")
    monkeypatch.setattr("mark.embeddings.model2vec.Model2VecProvider", Model2VecProvider)
    monkeypatch.setattr(auto, "_cached_model2vec_path", lambda m: "/c" if m == auto.RETRIEVAL_MODEL2VEC_MODEL else None)
    assert auto_embedder().fingerprint == f"fake:{auto.RETRIEVAL_MODEL2VEC_MODEL}:32"
    monkeypatch.setattr(auto, "_cached_model2vec_path", lambda m: None)
    # Download allowed: only the small default model is fetched.
    assert auto_embedder(allow_download=True).fingerprint == f"fake:{auto.DEFAULT_MODEL2VEC_MODEL}:32"


def test_fastembed_not_cached_is_skipped_quietly(monkeypatch: pytest.MonkeyPatch) -> None:
    class FastEmbedProvider(_Fake):
        def __init__(self, model: str, **kwargs: object) -> None:
            raise ValueError("Could not load model from any source.")

    monkeypatch.setattr(auto, "_installed", lambda module: module == "fastembed")
    monkeypatch.setattr("mark.embeddings.fastembed.FastEmbedProvider", FastEmbedProvider)
    assert auto._try_fastembed(auto.DEFAULT_FASTEMBED_MODEL, False) is None
    with pytest.raises(ValueError):
        auto._try_fastembed(auto.DEFAULT_FASTEMBED_MODEL, True)


def test_cached_model2vec_path_accepts_local_directory(tmp_path) -> None:
    assert auto._cached_model2vec_path(str(tmp_path)) == str(tmp_path)


# ── Spec / MARK_EMBEDDER parsing ─────────────────────────────────────────────


@pytest.mark.parametrize("spec, expected", [
    ("auto", ("auto", "", "")),
    ("hash:64", ("hash", "64", "")),
    ("model2vec:minishlab/potion-retrieval-32M", ("model2vec", "minishlab/potion-retrieval-32M", "")),
    ("ollama:nomic-embed-text-v2-moe:latest", ("ollama", "nomic-embed-text-v2-moe:latest", "")),
    ("OLLAMA:bge-m3|http://gpu:11434", ("ollama", "bge-m3", "http://gpu:11434")),
    ("openai:http://localhost:8080/v1|qwen3-embed", ("openai", "http://localhost:8080/v1", "qwen3-embed")),
])
def test_parse_spec(spec: str, expected: tuple[str, str, str]) -> None:
    assert auto.parse_embedder_spec(spec) == expected


@pytest.mark.parametrize("spec", ["", "bert", "openai:http://x/v1", "openai:|m", "hash:big"])
def test_bad_specs_raise(spec: str) -> None:
    with pytest.raises(ValueError):
        auto.parse_embedder_spec(spec)


def test_env_var_selects_ollama_lazily(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(auto.ENV_VAR, "ollama:nomic-embed-text-v2-moe:latest")
    provider = auto_embedder()  # constructing must not call the server
    assert isinstance(provider, OllamaEmbeddingProvider)
    assert provider.model == "nomic-embed-text-v2-moe:latest"
    assert provider.base_url == "http://localhost:11434"


def test_env_var_selects_openai_with_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(auto.ENV_VAR, "openai:http://127.0.0.1:1234/v1|embed-small")
    monkeypatch.setenv(auto.API_KEY_ENV_VAR, "sk-test")
    provider = auto_embedder()
    assert isinstance(provider, OpenAICompatibleEmbeddingProvider)
    assert (provider.base_url, provider.model) == ("http://127.0.0.1:1234/v1", "embed-small")
    assert "sk-test" not in repr(provider) and "api_key=<set>" in repr(provider)


def test_prefer_overrides_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(auto.ENV_VAR, "ollama:x")
    assert auto_embedder("hash:64").fingerprint == "mark.hash-v3:64"


def test_explicit_local_kind_raises_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys

    monkeypatch.setitem(sys.modules, "model2vec", None)  # makes the import fail
    with pytest.raises(ImportError):
        auto_embedder("model2vec")


# ── Never implicit network ───────────────────────────────────────────────────


def test_network_providers_are_never_chosen_implicitly(monkeypatch: pytest.MonkeyPatch) -> None:
    assert all(name not in auto._NETWORK_KINDS for name, _, _ in auto._LOCAL_CANDIDATES)
    monkeypatch.setattr(auto, "_installed", lambda module: True)  # broken installs everywhere
    monkeypatch.setattr(auto, "_cached_model2vec_path", lambda model: None)
    for provider in (auto_embedder(), resolve_embedder("auto"), resolve_embedder(None)):
        assert not isinstance(provider, (OllamaEmbeddingProvider, OpenAICompatibleEmbeddingProvider))
    assert auto.is_network_spec("ollama:m") and not auto.is_network_spec("auto")


# ── resolve_embedder / MarkRuntime.local wiring ──────────────────────────────


def test_resolve_embedder_forms(monkeypatch: pytest.MonkeyPatch) -> None:
    custom = _Fake("custom")
    assert resolve_embedder(custom) is custom
    assert type(resolve_embedder(None)) is HashEmbeddingProvider
    assert resolve_embedder("hash:128").dim == 128
    assert resolve_embedder(" AUTO ").fingerprint == "mark.hash-v3:256"
    monkeypatch.setenv(auto.ENV_VAR, "hash:64")
    assert resolve_embedder(None).fingerprint == "mark.hash-v3:64"  # explicit operator opt-in
    with pytest.raises(TypeError):
        resolve_embedder(42)


def test_runtime_local_accepts_auto_string() -> None:
    runtime = MarkRuntime.local(embedder="auto")
    try:
        assert runtime.embedder.fingerprint == "mark.hash-v3:256"
        mem = runtime.memory("auto")
        fid = mem.store_sync("The billing service runs on port 9443.")
        assert mem.retrieve_sync("billing service port").fragments[0].id == fid
    finally:
        runtime.shutdown()


def test_default_runtime_embedder_unchanged() -> None:
    runtime = MarkRuntime.local()
    try:
        assert type(runtime.embedder) is HashEmbeddingProvider
    finally:
        runtime.shutdown()
