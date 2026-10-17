"""Embedding provider protocol and the deterministic hash fallback."""
from __future__ import annotations

import hashlib
import math
import re
from abc import ABC, abstractmethod
from functools import lru_cache


class EmbeddingProvider(ABC):
    """Interface for local or developer-provided embedding models.

    Subclasses implement ``embed`` and ``dim``. Providers that can embed many
    texts in one model call should override ``embed_batch`` — it is the
    batching hook every MARK component uses for bulk work.
    """

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        """Return an L2-normalized vector for one text."""

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed several texts; override to batch them into one model call."""
        return [self.embed(text) for text in texts]

    @property
    @abstractmethod
    def dim(self) -> int | None:
        """Embedding dimension if known."""

    @property
    def fingerprint(self) -> str:
        """Stable identity of the embedding space (model + dimension).

        Vectors produced under different fingerprints are not comparable;
        stores use this to decide when stored embeddings must be recomputed.
        """
        return f"{type(self).__module__}.{type(self).__qualname__}:{self.dim}"

    @property
    def semantic(self) -> bool:
        """True when similar *meanings* (not just shared words) embed close together.

        Retrieval uses this to decide how much to trust the vector channel
        over lexical matches. Model-based providers are semantic, so that is
        the default; purely lexical embedders (feature hashing) override it.
        """
        return True


# Small English stopword list: function words that carry no topical signal.
_STOPWORDS = frozenset(
    """
    a an and are as at be been but by can could did do does for from had has have
    he her hers him his how i if in into is it its me my no nor not of on or our
    ours she so than that the their theirs them then there these they this those
    to too us was we were what when where which who whom why will with would you
    your yours
    """.split()
)
_WORD_RE = re.compile(r"\w+")
_POSSESSIVE_RE = re.compile(r"(\w)['’]s\b")

_UNIGRAM_WEIGHT = 1.0
_BIGRAM_WEIGHT = 1.0
_CHAR_WEIGHT = 0.5


@lru_cache(maxsize=1 << 16)
def _feature_hash(feature: str) -> int:
    return int.from_bytes(hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest(), "little")


def _hash_features(text: str) -> dict[str, float]:
    """Weighted feature counts: unigrams, adjacent bigrams, char 3-grams."""
    words = _WORD_RE.findall(_POSSESSIVE_RE.sub(r"\1", text.lower()))
    content = [word for word in words if word not in _STOPWORDS] or words
    counts: dict[str, float] = {}
    for word in content:
        key = "w:" + word
        counts[key] = counts.get(key, 0.0) + 1.0
        padded = f"<{word}>"
        for start in range(len(padded) - 2):
            key = "c:" + padded[start:start + 3]
            counts[key] = counts.get(key, 0.0) + 1.0
    for left, right in zip(content, content[1:]):
        key = f"b:{left} {right}"
        counts[key] = counts.get(key, 0.0) + 1.0
    return counts


class HashEmbeddingProvider(EmbeddingProvider):
    """
    Deterministic CPU-only embedder for tests and zero-dependency local demos.

    v2 features: lowercase ``\\w+`` words (possessive 's stripped, English
    stopwords dropped), adjacent-word bigrams, and boundary-marked character
    3-grams (weight 0.5), combined with signed feature hashing (blake2b),
    sublinear term frequency, and L2 normalisation.

    Production local usage should pass a stronger provider, such as
    OpenAICompatibleEmbeddingProvider (llama.cpp, LM Studio, vLLM, Ollama),
    FastEmbedProvider, Model2VecProvider, or sentence-transformers.
    """

    def __init__(self, dim: int = 256) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        self._dim = dim

    def embed(self, text: str) -> list[float]:
        """Return the embedding vector for the text."""
        dim = self._dim
        vector = [0.0] * dim
        for feature, count in _hash_features(text).items():
            kind = feature[0]
            weight = _CHAR_WEIGHT if kind == "c" else _BIGRAM_WEIGHT if kind == "b" else _UNIGRAM_WEIGHT
            digest = _feature_hash(feature)
            value = weight * (1.0 + math.log(count))
            if digest >> 63:
                value = -value
            vector[digest % dim] += value
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            return vector
        return [value / norm for value in vector]

    @property
    def dim(self) -> int:
        """Return the embedding dimension."""
        return self._dim

    @property
    def fingerprint(self) -> str:
        """Identity of the v2 hash embedding space."""
        return f"mark.hash-v2:{self._dim}"

    @property
    def semantic(self) -> bool:
        """Hashed words and n-grams: lexical similarity only."""
        return False
