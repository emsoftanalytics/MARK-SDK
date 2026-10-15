"""Embedding providers: hash fallback, local models, HTTP servers, and ``auto_embedder``."""
from mark.embeddings._http import EmbeddingHTTPError
from mark.embeddings.auto import auto_embedder, resolve_embedder
from mark.embeddings.base import EmbeddingProvider, HashEmbeddingProvider
from mark.embeddings.cache import CachedEmbeddingProvider
from mark.embeddings.fastembed import FastEmbedProvider
from mark.embeddings.model2vec import Model2VecProvider
from mark.embeddings.ollama import OllamaEmbeddingProvider
from mark.embeddings.openai_compatible import OpenAICompatibleEmbeddingProvider
from mark.embeddings.sentence_transformers import SentenceTransformerEmbeddingProvider

__all__ = [
    "CachedEmbeddingProvider",
    "EmbeddingHTTPError",
    "EmbeddingProvider",
    "FastEmbedProvider",
    "HashEmbeddingProvider",
    "Model2VecProvider",
    "OllamaEmbeddingProvider",
    "OpenAICompatibleEmbeddingProvider",
    "SentenceTransformerEmbeddingProvider",
    "auto_embedder",
    "resolve_embedder",
]
