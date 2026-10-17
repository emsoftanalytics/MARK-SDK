"""Pick the strongest embedding provider that is available *without* network access.

``auto_embedder()`` walks an in-process candidate list and returns the first
provider that can be built locally:

1. Model2Vec (``potion`` static models) — if ``model2vec`` is importable and
   the model is already in the Hugging Face cache: ``potion-retrieval-32M`` if
   cached, else ``potion-base-8M``. Only the latter is ever downloaded, and only
   with ``allow_download=True``.
2. FastEmbed (ONNX) — if ``fastembed`` is importable and the model is cached
   (``local_files_only``), or downloads are allowed.
3. :class:`HashEmbeddingProvider` — the zero-dependency lexical fallback.

Invariants (each one is tested):

- **No implicit network.** Ollama / OpenAI-compatible servers are never probed
  or chosen automatically, even when one is running on localhost. They are used
  only when explicitly named: ``prefer="ollama:<model>"``,
  ``prefer="openai:<base_url>|<model>"``, or the same string in the
  ``MARK_EMBEDDER`` environment variable.
- **Nothing installed, no env var => exactly the hash embedder** (same
  ``mark.hash-v2:256`` fingerprint as the historical default).
- **Lazy.** This module imports only the standard library and the embeddings
  base at import time; optional packages are imported inside candidate builders.

Spec syntax (``prefer`` / ``MARK_EMBEDDER``)::

    auto                         best local provider (the default)
    hash | hash:<dim>            lexical hash embedder
    model2vec[:<model>]          Model2Vec (explicit => download allowed)
    fastembed[:<model>]          FastEmbed (explicit => download allowed)
    ollama[:<model>][|<url>]     Ollama /api/embed (network, explicit only)
    openai:<base_url>|<model>    OpenAI-compatible /embeddings (network, explicit only)

The OpenAI-compatible API key, if any, is read from ``MARK_EMBEDDER_API_KEY``.
"""
from __future__ import annotations

import importlib.util
import logging
import os
from pathlib import Path
from typing import Callable

from mark.embeddings.base import EmbeddingProvider, HashEmbeddingProvider

logger = logging.getLogger("mark.embeddings")

ENV_VAR = "MARK_EMBEDDER"
API_KEY_ENV_VAR = "MARK_EMBEDDER_API_KEY"
DEFAULT_MODEL2VEC_MODEL = "minishlab/potion-base-8M"
RETRIEVAL_MODEL2VEC_MODEL = "minishlab/potion-retrieval-32M"
DEFAULT_FASTEMBED_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_OLLAMA_MODEL = "nomic-embed-text"
# Files a Model2Vec repo needs; all must be cached for an offline load.
_MODEL2VEC_FILES = ("config.json", "model.safetensors", "tokenizer.json")
_NETWORK_KINDS = frozenset({"ollama", "openai"})
_KINDS = frozenset({"auto", "hash", "model2vec", "fastembed"}) | _NETWORK_KINDS


def parse_embedder_spec(spec: str) -> tuple[str, str, str]:
    """Split a spec into ``(kind, arg, extra)``; raises ``ValueError`` if malformed.

    Only the first ``:`` separates the kind, because model names (``nomic:latest``)
    and URLs contain colons. ``|`` separates the second argument.
    """
    text = spec.strip()
    kind, _, rest = text.partition(":")
    kind = kind.strip().lower()
    if kind not in _KINDS:
        raise ValueError(f"unknown embedder spec {spec!r}; expected one of {sorted(_KINDS)}")
    arg, _, extra = rest.partition("|")
    arg, extra = arg.strip(), extra.strip()
    if kind == "openai" and (not arg or not extra):
        raise ValueError(f"openai embedder spec must be 'openai:<base_url>|<model>', got {spec!r}")
    if kind == "hash" and arg and not arg.isdigit():
        raise ValueError(f"hash embedder spec must be 'hash' or 'hash:<dim>', got {spec!r}")
    return kind, arg, extra


def is_network_spec(spec: str) -> bool:
    """True when the spec names a provider that talks to a server."""
    return parse_embedder_spec(spec)[0] in _NETWORK_KINDS


# ── Local candidate builders (return None when unavailable) ──────────────────


def _installed(module: str) -> bool:
    """Whether an optional package is importable, without importing it."""
    return importlib.util.find_spec(module) is not None


def _cached_model2vec_path(model: str) -> str | None:
    """Local directory holding ``model``, without touching the network."""
    path = Path(model).expanduser()
    if path.is_dir():
        return str(path)
    try:
        from huggingface_hub import try_to_load_from_cache  # type: ignore[import-not-found]
    except ImportError:
        return None
    found = [try_to_load_from_cache(model, name) for name in _MODEL2VEC_FILES]
    if all(isinstance(item, str) for item in found):
        return str(Path(found[0]).parent)  # type: ignore[arg-type]
    return None


def _try_model2vec(model: str, allow_download: bool) -> EmbeddingProvider | None:
    if not _installed("model2vec"):
        return None
    from mark.embeddings.model2vec import Model2VecProvider

    local = _cached_model2vec_path(model)
    if local is None and not allow_download:
        logger.debug("model2vec installed but %s is not cached; skipping (allow_download=False)", model)
        return None
    return Model2VecProvider(model, local_path=local)


def _try_fastembed(model: str, allow_download: bool) -> EmbeddingProvider | None:
    if not _installed("fastembed"):
        return None
    from mark.embeddings.fastembed import FastEmbedProvider

    if allow_download:
        provider = FastEmbedProvider(model)
    else:
        try:
            provider = FastEmbedProvider(model, local_files_only=True)
        except Exception as exc:  # fastembed raises ValueError when the model is not cached
            logger.info("fastembed installed but %s is not cached (%s); skipping", model, exc)
            return None
    # Learn the dimension now so the fingerprint is complete before any store
    # compares it (a ``:None`` fingerprint cannot trigger re-embedding).
    provider.embed("dimension probe")
    return provider


_LocalBuilder = Callable[[str, bool], "EmbeddingProvider | None"]

# Order is the preference order of ``auto``. Tests patch this list.
# potion-retrieval-32M (125 MB) scored best among Model2Vec models on the
# paraphrase set but is used only when already cached; the default download
# (allow_download=True) is the 30 MB potion-base-8M.
_LOCAL_CANDIDATES: list[tuple[str, str, _LocalBuilder]] = [
    ("model2vec", RETRIEVAL_MODEL2VEC_MODEL, lambda model, _download: _try_model2vec(model, False)),
    ("model2vec", DEFAULT_MODEL2VEC_MODEL, _try_model2vec),
    ("fastembed", DEFAULT_FASTEMBED_MODEL, _try_fastembed),
]


# ── Public API ───────────────────────────────────────────────────────────────


def embedder_from_spec(spec: str, *, allow_download: bool = False) -> EmbeddingProvider:
    """Build the provider a spec names. Explicit local kinds may download; ``auto`` may not
    unless ``allow_download`` is set. Explicit kinds raise if they cannot be built."""
    kind, arg, extra = parse_embedder_spec(spec)
    if kind == "auto":
        return _auto_local(allow_download)
    if kind == "hash":
        return HashEmbeddingProvider(int(arg)) if arg else HashEmbeddingProvider()
    if kind == "model2vec":
        from mark.embeddings.model2vec import Model2VecProvider

        model = arg or DEFAULT_MODEL2VEC_MODEL
        return Model2VecProvider(model, local_path=_cached_model2vec_path(model))
    if kind == "fastembed":
        from mark.embeddings.fastembed import FastEmbedProvider

        return FastEmbedProvider(arg or DEFAULT_FASTEMBED_MODEL)
    if kind == "ollama":
        from mark.embeddings.ollama import OllamaEmbeddingProvider

        model = arg or DEFAULT_OLLAMA_MODEL
        return OllamaEmbeddingProvider(model, extra) if extra else OllamaEmbeddingProvider(model)
    from mark.embeddings.openai_compatible import OpenAICompatibleEmbeddingProvider

    return OpenAICompatibleEmbeddingProvider(base_url=arg, model=extra, api_key=os.environ.get(API_KEY_ENV_VAR) or None)


def _auto_local(allow_download: bool) -> EmbeddingProvider:
    for name, model, build in _LOCAL_CANDIDATES:
        try:
            provider = build(model, allow_download)
        except Exception as exc:  # a broken optional install must never break startup
            logger.warning("auto_embedder: %s unavailable (%s: %s); trying next", name, type(exc).__name__, exc)
            continue
        if provider is not None:
            logger.info("auto_embedder: using %s (%s)", name, provider.fingerprint)
            return provider
        logger.debug("auto_embedder: %s not available", name)
    provider = HashEmbeddingProvider()
    logger.info("auto_embedder: no local model available; using %s", provider.fingerprint)
    return provider


def auto_embedder(prefer: str | None = None, *, allow_download: bool = False) -> EmbeddingProvider:
    """Return the best available embedder.

    ``prefer`` (or ``MARK_EMBEDDER`` when ``prefer`` is None) is a spec string —
    see the module docstring. With neither set, the in-process candidates are
    tried in order and the hash embedder is the final fallback. Network-backed
    providers are returned only when a spec names them explicitly.
    """
    source = "prefer"
    if prefer is None:
        prefer = os.environ.get(ENV_VAR, "").strip() or None
        source = ENV_VAR
    if prefer is None:
        return _auto_local(allow_download)
    provider = embedder_from_spec(prefer, allow_download=allow_download)
    if parse_embedder_spec(prefer)[0] != "auto":
        logger.info("auto_embedder: using %s from %s=%r", provider.fingerprint, source, prefer)
    return provider


def resolve_embedder(embedder: object | None) -> EmbeddingProvider:
    """Normalise the ``embedder=`` argument of ``MarkRuntime.local`` / ``Mark.local``.

    - a provider instance is returned unchanged;
    - a string is a spec (``"auto"``, ``"hash"``, ``"ollama:<model>"`` ...);
    - ``None`` keeps the historical default — the hash embedder — unless the
      operator opted in through ``MARK_EMBEDDER`` (an explicit choice, so it
      is honoured). Installed packages alone never change the default.
    """
    if embedder is None:
        env = os.environ.get(ENV_VAR, "").strip()
        return auto_embedder(env) if env else HashEmbeddingProvider()
    if isinstance(embedder, str):
        return auto_embedder(None if embedder.strip().lower() == "auto" else embedder)
    if not isinstance(embedder, EmbeddingProvider) and not hasattr(embedder, "embed"):
        raise TypeError(f"embedder must be an EmbeddingProvider, a spec string or None, got {type(embedder).__name__}")
    return embedder  # type: ignore[return-value]
