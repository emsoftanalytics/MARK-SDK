# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 Somtochukwu Owen Ikeanyi
"""The store remembers its embedder.

A store embedded with Model2Vec used to be opened by the CLI, MCP or a plain
``Mark.local(path)`` with the default hash embedder, which silently
re-embedded every memory (and back again on the next switch). Now:

* **Remembered.** The embedder's spec (``"model2vec:minishlab/potion-base-8M"``,
  ``"hash"``...) and fingerprint are stored in the store's meta table
  (key :data:`STORE_EMBEDDER_KEY`) when the store is created, and updated
  when an explicit choice replaces them. Stores from earlier versions are
  recognised from the per-agent fingerprints they already hold.
* **Reused.** Opening without an explicit embedder (``embedder=None`` or
  ``"auto"``, no ``MARK_EMBEDDER``) builds the stored one, without
  downloading anything; :func:`mark.embeddings.info.embedder_info` reports
  ``source="store"``.
* **Never silently replaced.** When the stored embedder cannot be built
  here (its extra is not installed, its model is not cached, it was a custom
  provider instance), the store opens **lexical-only** with one warning that
  names it and how to install it. The stored vectors are left untouched and
  new memories wait for an embedding until the embedder is available again.
* **Explicit wins.** ``embedder=<other spec or provider>`` or
  ``MARK_EMBEDDER`` re-embeds the store with that embedder (as before) and
  becomes the stored one.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any

from mark.embeddings.auto import ENV_VAR, _tag, parse_embedder_spec, resolve_embedder
from mark.embeddings.base import EmbeddingProvider, HashEmbeddingProvider

logger = logging.getLogger("mark.embeddings")

#: schema_info key holding ``{"spec": ..., "fingerprint": ...}``.
STORE_EMBEDDER_KEY = "store_embedder"
#: ``EmbedderInfo.source`` of an embedder taken from the store.
SOURCE_STORE = "store"
_AGENT_KEY_PREFIX = "embedder_fingerprint:"
_warned: set[str] = set()


class StoredEmbedderUnavailable(ConnectionError):
    """The store's embedder cannot be built here (recall is lexical-only meanwhile)."""


class UnavailableEmbedder(EmbeddingProvider):
    """Stands in for a stored embedder that cannot be loaded: same fingerprint, no vectors.

    Every ``embed`` call raises :class:`StoredEmbedderUnavailable` (a
    ``ConnectionError``, which retrieval treats as "embedder down": keyword
    search answers, new memories are embedded later). Keeping the stored
    fingerprint means nothing is re-embedded.
    """

    semantic = False

    def __init__(self, spec: str | None, fingerprint: str) -> None:
        self.spec = spec
        self._fingerprint = fingerprint
        tail = fingerprint.rsplit(":", 1)[-1]
        self._dim = int(tail) if tail.isdigit() else None
        self.model_name = f"{spec or fingerprint} (unavailable: lexical-only)"

    @property
    def dim(self) -> int | None:
        return self._dim

    @property
    def fingerprint(self) -> str:
        return self._fingerprint

    def embed(self, text: str) -> list[float]:
        raise StoredEmbedderUnavailable(f"stored embedder {self.spec or self._fingerprint} is not available")

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]


def spec_of(provider: Any) -> str | None:
    """The spec string that rebuilds *provider*, or None (a custom provider instance)."""
    if isinstance(provider, UnavailableEmbedder):
        return provider.spec
    if isinstance(provider, HashEmbeddingProvider):
        return "hash" if provider.dim == 256 else f"hash:{provider.dim}"
    kind = type(provider).__name__
    model = getattr(provider, "model_name", None) or getattr(provider, "model", None)
    if kind == "Model2VecProvider":
        return f"model2vec:{model}"
    if kind == "FastEmbedProvider":
        return f"fastembed:{model}"
    if kind == "OllamaEmbeddingProvider":
        return f"ollama:{model}|{provider.base_url}"
    if kind == "OpenAICompatibleEmbeddingProvider":
        return f"openai:{provider.base_url}|{model}"
    return None


def _spec_from_fingerprint(fingerprint: str) -> str | None:
    """A spec for a legacy per-agent fingerprint (local kinds only; servers need their URL)."""
    kind, _, rest = fingerprint.partition(":")
    model, _, dim = rest.rpartition(":")
    if kind.startswith("mark.hash"):
        return "hash" if rest == "256" else f"hash:{rest}"
    if kind in ("model2vec", "fastembed") and model and dim.isdigit():
        return f"{kind}:{model}"
    return None


def stored_embedder(store: Any) -> dict[str, Any] | None:
    """``{"spec", "fingerprint"}`` the store was embedded with, or None for a new store."""
    get_meta = getattr(store, "get_meta", None)
    if get_meta is None:
        return None
    rows = store._fetchall("SELECT value FROM schema_info WHERE key LIKE ?", (_AGENT_KEY_PREFIX + "%",))
    prints = sorted({str(row[0]) for row in rows})
    semantic = [p for p in prints if not p.startswith("mark.hash")]
    agents_print = semantic[0] if semantic else (prints[-1] if prints else None)
    raw = get_meta(STORE_EMBEDDER_KEY)
    if raw:
        try:
            data = json.loads(raw)
        except ValueError:
            data = None
        if isinstance(data, dict) and data.get("fingerprint"):
            if str(data["fingerprint"]).endswith(":None") and agents_print:
                data["fingerprint"] = agents_print  # dimension learnt by the first embed
            return data
    if agents_print is None:
        return None
    # A store from 0.2.0a9 or earlier: its agents' fingerprints tell.
    return {"spec": _spec_from_fingerprint(agents_print), "fingerprint": agents_print}


def _build_stored(spec: str) -> EmbeddingProvider | None:
    """Build the stored embedder without downloading; None when it cannot be built here."""
    from mark.embeddings import auto

    try:
        kind, arg, _ = parse_embedder_spec(spec)
        if kind == "model2vec":
            return auto._try_model2vec(arg or auto.DEFAULT_MODEL2VEC_MODEL, False)
        if kind == "fastembed":
            return auto._try_fastembed(arg or auto.DEFAULT_FASTEMBED_MODEL, False)
        return auto.embedder_from_spec(spec)  # hash, or a server named with its URL
    except Exception as exc:  # noqa: BLE001 - missing extra, broken install: report, don't crash
        logger.debug("stored embedder %s could not be built: %s", spec, exc)
        return None


def _install_hint(spec: str | None) -> str:
    kind = (spec or "").partition(":")[0]
    if kind in ("model2vec", "fastembed"):
        return (f'install it with pip install "mark-sdk[{kind}]" and make sure the model is downloaded '
                f'(open once with embedder="{spec}"), or pass embedder="{spec}" explicitly')
    return f'pass the same embedder again (embedder=...{f" or {spec!r}" if spec else ""})'


def embedder_for_store(store: Any, requested: object | None) -> EmbeddingProvider:
    """The embedder to open *store* with (see the module docstring); records it in the store."""
    stored = stored_embedder(store)
    wants_default = requested is None or (isinstance(requested, str) and requested.strip().lower() == "auto")
    explicit = not wants_default or bool(os.environ.get(ENV_VAR, "").strip())
    if stored is not None and not explicit:
        spec = stored.get("spec")
        provider = _build_stored(spec) if spec else None
        if provider is not None:
            return _tag(provider, SOURCE_STORE)
        key = f"{getattr(store, 'path', '')}|{stored['fingerprint']}"
        if key not in _warned:
            _warned.add(key)
            logger.warning(
                "this store was embedded with %s, which cannot be loaded here; recall is keyword-only "
                "(stored vectors are kept, new memories are embedded once it is available): %s",
                spec or stored["fingerprint"], _install_hint(spec))
        return _tag(UnavailableEmbedder(spec, str(stored["fingerprint"])), SOURCE_STORE)
    provider = resolve_embedder(requested)
    if stored is not None and stored.get("fingerprint") != getattr(provider, "fingerprint", None):
        logger.info("store embedder changes from %s to %s; memories are re-embedded",
                    stored.get("spec") or stored.get("fingerprint"), spec_of(provider) or provider)
    remember_embedder(store, provider)
    return provider


def remember_embedder(store: Any, provider: Any) -> None:
    """Record *provider* as the store's embedder (its spec, and its fingerprint once known)."""
    fingerprint = str(getattr(provider, "fingerprint", "") or "")
    set_meta = getattr(store, "set_meta", None)
    if set_meta is None or not fingerprint or (fingerprint.endswith(":None") and spec_of(provider) is None):
        return
    value = json.dumps({"spec": spec_of(provider), "fingerprint": fingerprint})
    if store.get_meta(STORE_EMBEDDER_KEY) != value:
        set_meta(STORE_EMBEDDER_KEY, value)


def open_store(store_path: Any, encryption: Any, requested: object | None,
               **upgrade: Any) -> tuple[Any, EmbeddingProvider]:
    """``(LocalMemoryStore, embedder)`` for ``MarkRuntime.local``; the store is closed on failure.

    A store in MARK's own ``.mark`` folder left group/world-readable by an
    older version is made private first (:func:`mark._private_fs.tighten_mark_store`).
    *upgrade* (``backup_on_upgrade``, ``backup_dir``) goes to the store.
    """
    from mark._private_fs import tighten_mark_store
    from mark.store.sqlite import LocalMemoryStore

    if str(store_path) != ":memory:":
        for path in tighten_mark_store(store_path):
            logger.info("made %s private (owner-only), as MARK creates it", path)
    store = LocalMemoryStore(store_path, encryption=encryption, **upgrade)
    try:
        return store, embedder_for_store(store, requested)
    except BaseException:
        store.close()
        raise


def quiet_unavailable(runtime: Any) -> Any:
    """With an :class:`UnavailableEmbedder`, mark retrieval's outage as known (no second warning)."""
    if isinstance(getattr(runtime, "embedder", None), UnavailableEmbedder):
        runtime.pipeline.__dict__["_embed_outage"] = True
    return runtime
