"""Retrieval policies, per-policy knobs, session filters and pipeline constants.

Everything here is plain configuration: which policy a caller asked for,
what that policy means in concrete numbers (top-k, graph depth, gap
threshold), and how a retrieval is scoped to a session, tag set or block.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol

from mark.intelligence.classifier import QueryAnalysis
from mark.types import MemoryScope, MemoryState, MemoryTier


class RetrievalPolicy(str, Enum):
    """Built-in retrieval policy names.

    FAST / BALANCED / DEEP are authoritative: the pipeline uses exactly that
    policy's knobs. AUTO lets the query classifier pick one per query.
    """
    FAST = "fast"
    BALANCED = "balanced"
    DEEP = "deep"
    AUTO = "auto"


class ReinforceMode(str, Enum):
    """How retrieval feeds back into plasticity.

    RELEVANT  record access for every result; boost importance and co-activate
              edges only for results with real query evidence (default)
    ALL       legacy behaviour: boost every returned result
    NONE      read-only retrieval; no writes at all
    """
    RELEVANT = "relevant"
    ALL = "all"
    NONE = "none"


@dataclass(frozen=True)
class RetrievalPolicySpec:
    """Concrete retrieval knobs for one policy (top_k, depth, thresholds)."""
    name: str
    display_name: str
    description: str = ""
    top_k: int = 10
    graph_depth: int = 1
    rerank_depth: int = 1
    min_confidence: float = 0.30
    enable_decomposition: bool = False
    max_subqueries: int = 1
    critic_enabled: bool = False
    enable_external: bool = False
    is_builtin: bool = True
    expansion_direction: str = "both"   # "outgoing" | "incoming" | "both"


BUILTIN_POLICIES: dict[str, RetrievalPolicySpec] = {
    "fast": RetrievalPolicySpec(
        name="fast",
        display_name="Fast Lookup",
        description="Single-hop local recall.",
        top_k=5,
        graph_depth=0,
        min_confidence=0.25,
        expansion_direction="outgoing",
    ),
    "balanced": RetrievalPolicySpec(
        name="balanced",
        display_name="Balanced",
        description="Hybrid retrieval plus one-hop graph expansion.",
        top_k=10,
        graph_depth=1,
        min_confidence=0.40,
        expansion_direction="both",
    ),
    "deep": RetrievalPolicySpec(
        name="deep",
        display_name="Deep Recall",
        description="Hybrid retrieval plus two-hop graph expansion.",
        top_k=15,
        graph_depth=2,
        min_confidence=0.55,
        expansion_direction="both",
    ),
}

# Only these states may ever be returned; candidates, rejected, deleted or
# hash-failed fragments are filtered out before and after scoring.
_RETRIEVABLE_STATES = {MemoryState.UNVERIFIED, MemoryState.VERIFIED, MemoryState.PROMOTED}

# Reciprocal Rank Fusion constant (Cormack et al., SIGIR 2009). Each channel
# contributes 1 / (RRF_K + rank), so fusion depends on ranks, not on the
# incomparable raw scores of BM25 and cosine similarity.
RRF_K = 60
# Candidates fetched per channel before fusion, as a multiple of top_k.
_CHANNEL_DEPTH_FACTOR = 5
_CHANNEL_DEPTH_MIN = 50
# Graph neighbours inherit a damped share of the seed score per hop.
_GRAPH_HOP_DECAY = 0.5

_FUSION_MODES = ("auto", "rrf", "score")


@dataclass(frozen=True)
class FusionConfig:
    """How the vector and lexical channels are combined into seed scores.

    mode                   "auto": RRF for lexical embedders (the hash
                           default), score fusion for semantic ones;
                           "rrf" / "score" force one strategy.
    lexical_weight         share of term coverage in score fusion for plain
                           natural-language queries.
    anchor_lexical_weight  share of term coverage when the query names an
                           identifier or name that occurs in a candidate.

    Defaults are measured (benchmarks/embeddings/tune_fusion.py): the
    lexical weight on half of the paraphrase set, checked on the other half;
    the anchor weight on the retention A/B log, whose questions all name a
    service or ticket (flat optimum 0.6-0.9). With a semantic embedder,
    equal-weight RRF let a memory sharing one stray word with the query
    outrank the vector channel's best paraphrase match.
    """
    mode: str = "auto"
    lexical_weight: float = 0.1
    anchor_lexical_weight: float = 0.7

    def __post_init__(self) -> None:
        if self.mode not in _FUSION_MODES:
            raise ValueError(f"fusion mode must be one of {_FUSION_MODES}, got {self.mode!r}")
        for name in ("lexical_weight", "anchor_lexical_weight"):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"{name} must be within [0, 1]")


@dataclass(frozen=True)
class SessionFilter:
    """Optional session-scoping parameters for RetrievalPipeline.retrieve().

    session_id      exact session match (e.g. "episode-03")
    session_prefix  prefix match (e.g. "season-01/" matches all season-01 sessions)
    tags            all listed tags must be present on returned fragments (AND)
    tier            restrict to a specific MemoryTier
    scope           restrict to a specific MemoryScope
    block_id        restrict to fragments belonging to one memory block
    block_ids       restrict to fragments belonging to any listed block

    Pass no arguments for whole-agent retrieval (default, cross-session).
    Fragments inside QUARANTINED blocks are always excluded regardless of
    the filters used.
    """
    session_id:     str | None         = None
    session_prefix: str | None         = None
    tags:           list[str] | None   = None
    tier:           MemoryTier | None  = None
    scope:          MemoryScope | None = None
    block_id:       str | None         = None
    block_ids:      list[str] | None   = None


# QueryAnalysis is the Pydantic model from intelligence.classifier; any
# object providing analyze()/classify() can stand in for the classifier.
class QueryAnalyzer(Protocol):
    """Protocol for query analysis/classification implementations."""
    def analyze(self, query: str) -> QueryAnalysis:
        """Analyze the query and return the analysis."""
        ...
    def classify(self, query: str) -> QueryAnalysis:
        """Classify the query and return the analysis."""
        ...
