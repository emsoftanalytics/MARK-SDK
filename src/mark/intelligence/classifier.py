# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 MARK Contributors
#
# QueryClassifier — heuristic query complexity classifier.
# Runs in <1 ms with no LLM call. Drives retrieval depth (top_k, graph_depth).
#
# HOOK_POLICY_ENGINE can override classifier decisions at runtime.
# A registered adaptive routing plugin can replace this heuristic and improve
# through agent usage over time.
"""Deterministic query classification used to pick retrieval policies."""
from __future__ import annotations

import re
from enum import Enum
from typing import List, Set

from pydantic import BaseModel, Field


class QueryComplexity(str, Enum):
    """Named retrieval-depth tier derived from the complexity float score."""
    SIMPLE   = "simple"    # 0.00–0.29 → top_k = 5,  graph_depth = 0
    BALANCED = "balanced"  # 0.30–0.59 → top_k = 10, graph_depth = 1
    DEEP     = "deep"      # 0.60–1.00 → top_k = 15, graph_depth = 2


class QueryAnalysis(BaseModel):
    """
    Rich result of classifying a query before retrieval.

    Drives top_k and graph_depth in RetrievalPipeline.
    HOOK_POLICY_ENGINE may override these values.
    """
    query:           str
    complexity:      float          # 0.0 (simple lookup) → 1.0 (multi-hop reasoning)
    policy:          QueryComplexity
    is_factual:      bool           # matches a simple WH-question pattern
    is_multi_hop:    bool           # requires chaining multiple facts
    estimated_top_k: int            # suggested fragments to retrieve
    keywords:        List[str]      # salient content words for keyword fallback
    reasoning:       str            # full scoring breakdown string
    reasons:         List[str]      = Field(default_factory=list)
    # Richer feature flags used by plasticity/routing hooks.
    has_temporal:    bool = False   # temporal references ("last week", "in 2024")
    has_comparison:  bool = False   # comparative intent ("vs", "compare")
    has_multi_entity: bool = False  # two or more named entities detected
    has_reasoning:   bool = False   # analytical intent ("why", "explain")
    has_negation:    bool = False   # negation ("not", "never", "without")
    question_depth:  int  = 1       # 1=simple lookup, 2=moderate, 3=deep reasoning


_STOP_WORDS: Set[str] = {
    "about", "after", "all", "also", "an", "and", "any", "are", "as", "at",
    "be", "because", "been", "but", "by", "can", "could", "did", "do", "does",
    "each", "for", "from", "get", "got", "had", "has", "have", "he", "her",
    "here", "him", "his", "how", "if", "in", "into", "is", "it", "its",
    "just", "like", "make", "me", "more", "my", "no", "not", "now", "of",
    "on", "one", "only", "or", "other", "our", "out", "over", "said", "same",
    "she", "should", "so", "some", "than", "that", "the", "their", "them",
    "then", "there", "these", "they", "this", "those", "through", "to", "too",
    "under", "up", "use", "was", "we", "were", "what", "when", "where",
    "which", "while", "who", "will", "with", "would", "you", "your",
}

_TEMPORAL    = re.compile(
    r"\b(yesterday|today|last\s+\w+|recently|in\s+\d{4}|since|before|after|ago|now)\b",
    re.IGNORECASE,
)
_COMPARISON  = re.compile(
    r"\b(vs\.?|versus|compare|comparison|difference|between|better|worse|prefer)\b",
    re.IGNORECASE,
)
_REASONING   = re.compile(
    r"\b(why|how|explain|analyse|analyze|elaborate|describe|summarise|summarize"
    r"|impact|effect|cause|reason|mechanism)\b",
    re.IGNORECASE,
)
_NEGATION    = re.compile(r"\b(not|never|without|except|unless|no\s+\w+)\b", re.IGNORECASE)
_ENTITY_CAPS = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b")


class QueryClassifier:
    """
    Heuristic query complexity classifier.

    Scores a query on 0–1 and maps it to a retrieval tier:
      SIMPLE   (0.00–0.29)  top_k=5,  graph_depth=0
      BALANCED (0.30–0.59)  top_k=10, graph_depth=1
      DEEP     (0.60–1.00)  top_k=15, graph_depth=2

    Register HOOK_POLICY_ENGINE or HOOK_LEARNING_ROUTER to activate an adaptive
    routing plugin.
    """

    _SIMPLE: List[str] = [
        r"^what is\b", r"^what are\b", r"^who is\b", r"^where is\b",
        r"^when (did|was|is|are)\b", r"^define\b", r"^list\b", r"^name\b",
    ]
    _COMPLEX: List[str] = [
        r"\b(contrast|evaluate)\b",
        r"\b(consequence|implication)\b",
        r"\b(step by step|walk me through|in detail)\b",
        r"\b(describe the relationship|similarities between)\b",
    ]
    _MULTI_HOP: List[str] = [
        r"\b(then|after that|subsequently|as a result|therefore|thus)\b",
        r"\b(because of|due to|led to|caused by|resulting in)\b",
        r"\b(which (then|also|further))\b",
        r"\b(first .+ then)\b",
    ]

    def classify(self, query: str) -> QueryAnalysis:
        """Classify the query and return the analysis."""
        q     = query.lower().strip().rstrip("?")
        words = q.split()
        reasons: List[str] = []

        simple_hits  = sum(1 for p in self._SIMPLE  if re.search(p, q))
        complex_hits = sum(1 for p in self._COMPLEX if re.search(p, q))
        multi_hop    = any(re.search(p, q) for p in self._MULTI_HOP)

        has_temporal     = bool(_TEMPORAL.search(query))
        has_comparison   = bool(_COMPARISON.search(query))
        has_reasoning    = bool(_REASONING.search(query))
        has_negation     = bool(_NEGATION.search(query))
        entities         = _ENTITY_CAPS.findall(query)
        has_multi_entity = len(set(entities)) >= 2

        word_count_norm = min(len(words) / 30.0, 1.0)
        clause_count    = q.count(",") + q.count(";") + q.count("?")

        if simple_hits:      reasons.append("simple_lookup")
        if complex_hits:     reasons.append("complex_intent")
        if multi_hop:        reasons.append("multi_hop")
        if len(words) > 10:  reasons.append("long_query")
        if clause_count >= 2: reasons.append("clause_heavy")
        if has_temporal:     reasons.append("temporal")
        if has_multi_entity: reasons.append("multi_entity")
        if has_negation:     reasons.append("negation")

        raw = min(1.0, (
            complex_hits      * 0.30
            + word_count_norm * 0.20
            + (0.20 if multi_hop        else 0.0)
            + (0.15 if has_temporal     else 0.0)
            + (0.15 if has_multi_entity else 0.0)
            + (0.10 if has_negation     else 0.0)
            + clause_count    * 0.05
            - simple_hits     * 0.15
        ))
        complexity = round(max(0.0, raw), 4)

        if complexity < 0.30:
            policy, top_k = QueryComplexity.SIMPLE,   5
        elif complexity < 0.60:
            policy, top_k = QueryComplexity.BALANCED, 10
        else:
            policy, top_k = QueryComplexity.DEEP,     15

        depth = (
            3 if (has_reasoning or has_comparison) else
            2 if (has_temporal or has_multi_entity or len(words) > 15) else
            1
        )
        keywords = [w for w in words if len(w) >= 4 and w not in _STOP_WORDS][:12]
        reasoning = (
            f"complexity={complexity:.4f} | policy={policy.value} | "
            f"simple={simple_hits}, complex={complex_hits}, multi_hop={multi_hop}, "
            f"temporal={has_temporal}, multi_entity={has_multi_entity}, "
            f"negation={has_negation}, words={len(words)}, clauses={clause_count}"
        )

        return QueryAnalysis(
            query            = query,
            complexity       = complexity,
            policy           = policy,
            is_factual       = simple_hits > 0,
            is_multi_hop     = multi_hop,
            estimated_top_k  = top_k,
            keywords         = keywords,
            reasoning        = reasoning,
            reasons          = reasons,
            has_temporal     = has_temporal,
            has_comparison   = has_comparison,
            has_multi_entity = has_multi_entity,
            has_reasoning    = has_reasoning,
            has_negation     = has_negation,
            question_depth   = depth,
        )

    def analyze(self, query: str) -> QueryAnalysis:
        """Alias for classify() — satisfies the QueryAnalyzer Protocol."""
        return self.classify(query)


# Module-level singleton for lightweight use without instantiation
query_classifier = QueryClassifier()
