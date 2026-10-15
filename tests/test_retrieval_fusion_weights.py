# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 MARK Contributors
"""Embedder-aware fusion: semantic embedders get score fusion, hash keeps RRF.

A deterministic fake "semantic" embedder maps words to concepts (automobile
and parking share a concept; digits are invisible to it), which reproduces
the two cases measured in benchmarks/embeddings: a paraphrase whose
distractor shares one stray word with the query, and near-duplicate ids
that only the lexical channel can tell apart.
"""
from __future__ import annotations

import math
import re
import zlib

import pytest

from mark.embeddings import CachedEmbeddingProvider, EmbeddingProvider, HashEmbeddingProvider
from mark.index import VectorSearchResult
from mark.intelligence.fusion_scoring import lexical_share, normalised_cosines, uses_score_fusion
from mark.intelligence.retrieval_policy import FusionConfig
from mark.memory.runtime import MarkRuntime

_CONCEPTS = {
    "automobile": "car", "car": "car", "parking": "car", "garage": "car",
    "spring": "season", "june": "season", "leave": "leave", "requests": "request",
    "ticket": "ticket", "jira": "ticket", "tracks": "ticket", "memory": "memory",
}
_DIMS = sorted(set(_CONCEPTS.values()))
_NOISE = 32


class ConceptEmbedder(EmbeddingProvider):
    """Bag of concepts; unknown words land in weak hashed buckets, digits vanish."""

    def embed(self, text: str) -> list[float]:
        vector = [0.0] * (len(_DIMS) + _NOISE)
        for word in re.findall(r"[a-z]+", text.lower()):  # digits never reach the vector
            concept = _CONCEPTS.get(word)
            if concept:
                vector[_DIMS.index(concept)] += 1.0
            elif len(word) > 3:
                vector[len(_DIMS) + zlib.crc32(word.encode()) % _NOISE] += 0.3
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / norm for v in vector]

    @property
    def dim(self) -> int:
        return len(_DIMS) + _NOISE


FILLER = [
    "The cafeteria serves soup on Mondays.",
    "Backups run every night at two.",
    "The printer on the second floor jams.",
]


def _ranked(embedder: EmbeddingProvider, facts: list[str], query: str, fusion: FusionConfig) -> list[str]:
    runtime = MarkRuntime.local(embedder=embedder)
    runtime.pipeline.configure_fusion(fusion)
    try:
        mem = runtime.memory("fusion")
        for fact in facts:
            mem.store_sync(fact)
        return [f.content for f in mem.retrieve_sync(query).fragments]
    finally:
        runtime.shutdown()


# ── Paraphrase vs stray-word distractor ───────────────────────────────────────

GOLD = "The parking garage is closed for repairs until June."
DISTRACTOR = "Leave requests for June go to HR."
PARAPHRASE = "Where can I leave my automobile this spring?"


def test_semantic_paraphrase_beats_stray_word_distractor() -> None:
    ranked = _ranked(ConceptEmbedder(), [GOLD, DISTRACTOR, *FILLER], PARAPHRASE, FusionConfig())
    assert ranked[0] == GOLD


def test_equal_rrf_reproduces_the_distractor_failure() -> None:
    # Guards the test above: legacy fusion really does lose this case.
    ranked = _ranked(ConceptEmbedder(), [GOLD, DISTRACTOR, *FILLER], PARAPHRASE, FusionConfig(mode="rrf"))
    assert ranked[0] == DISTRACTOR


# ── Exact identifiers the embedder cannot see ─────────────────────────────────

TICKETS = [
    "Ticket JIRA-4521 tracks the memory leak.",
    "Ticket JIRA-4512 tracks the memory spike.",
    "Ticket JIRA-5421 tracks the memory crash.",
]


@pytest.mark.parametrize(("query", "gold"), [
    ("JIRA-4512", 1), ("What does JIRA-5421 track?", 2), ("ticket 4521", 0),
])
def test_exact_id_wins_with_semantic_embedder(query: str, gold: int) -> None:
    ranked = _ranked(ConceptEmbedder(), [*TICKETS, *FILLER], query, FusionConfig())
    assert ranked[0] == TICKETS[gold]


def test_anchor_weight_alone_carries_identifier_queries() -> None:
    # The embedder sees three identical tickets (a three-way cosine tie);
    # with plain lexical weight at zero, only the anchor share of term
    # coverage can put the right one first.
    facts = [*TICKETS, *FILLER]
    for query, gold in (("JIRA-4512", 1), ("JIRA-5421", 2), ("JIRA-4521", 0)):
        assert _ranked(ConceptEmbedder(), facts, query, FusionConfig(lexical_weight=0.0))[0] == TICKETS[gold]


# ── Hash embedder keeps RRF exactly ───────────────────────────────────────────

def test_hash_rankings_unchanged_by_auto_fusion() -> None:
    facts = [GOLD, DISTRACTOR, *TICKETS, *FILLER, "The billing service listens on port 9443."]
    for query in (PARAPHRASE, "JIRA-4512", "billing port", "when do backups run"):
        auto = _ranked(HashEmbeddingProvider(), facts, query, FusionConfig())
        legacy = _ranked(HashEmbeddingProvider(), facts, query, FusionConfig(mode="rrf"))
        assert auto == legacy


# ── Building blocks ───────────────────────────────────────────────────────────

def test_semantic_flag_by_provider() -> None:
    assert HashEmbeddingProvider().semantic is False
    assert CachedEmbeddingProvider(HashEmbeddingProvider()).semantic is False
    assert ConceptEmbedder().semantic is True
    assert uses_score_fusion(FusionConfig(), HashEmbeddingProvider()) is False
    assert uses_score_fusion(FusionConfig(), ConceptEmbedder()) is True
    assert uses_score_fusion(FusionConfig(mode="score"), HashEmbeddingProvider()) is True
    assert uses_score_fusion(FusionConfig(mode="rrf"), ConceptEmbedder()) is False


def test_fusion_config_validates() -> None:
    with pytest.raises(ValueError):
        FusionConfig(mode="bm25")
    with pytest.raises(ValueError):
        FusionConfig(lexical_weight=1.5)


def test_lexical_share_needs_anchor_in_memory() -> None:
    config = FusionConfig(lexical_weight=0.1, anchor_lexical_weight=0.7)
    # "help-desk" looks like an identifier but occurs nowhere: paraphrase path.
    assert lexical_share(config, ["help desk"], ["Support tickets get answers in 4h."]) == 0.1
    assert lexical_share(config, ["jira 4512"], TICKETS) == 0.7
    assert lexical_share(config, [], TICKETS) == 0.1


def test_normalised_cosines_are_relative_to_best_hit() -> None:
    hits = [VectorSearchResult("a", 0.6, ""), VectorSearchResult("b", 0.3, ""), VectorSearchResult("c", -0.1, "")]
    assert normalised_cosines(hits) == {"a": 1.0, "b": 0.5, "c": 0.0}
    assert normalised_cosines([VectorSearchResult("a", -0.2, "")]) == {"a": 0.0}
    assert normalised_cosines([]) == {}
