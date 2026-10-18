"""Seed scoring strategies for hybrid retrieval: rank fusion and score fusion.

Two ways to turn channel evidence into one seed score per candidate:

- **RRF** (``rrf_seed_score``): equal-weight Reciprocal Rank Fusion blended
  with calibrated relevance. Right for lexical embedders such as the hash
  default, whose vector channel is itself a fuzzy word matcher: two lexical
  views voting by rank make exact ids, names and synonyms work.
- **Score fusion** (``score_fusion_seed``): a convex combination of the
  cosine relative to the query's best hit and query-term coverage. Right
  for semantic embedders: RRF with k=60 flattens the vector channel's margin (rank 1 and
  rank 5 differ by ~6%), so a distractor sharing one stray word with the
  query collected lexical credit and beat the true paraphrase match. Here
  lexical evidence only decides near-ties, unless the query names an anchor
  (identifier or name) that really occurs in a candidate; then term
  coverage gets the larger share so exact ids and named subjects win.

Everything is deterministic and costs no model calls.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping

from mark.index import VectorSearchResult
from mark.intelligence.query_understanding import _anchor_norm, _contains_anchor
from mark.intelligence.retrieval_policy import RRF_K, FusionConfig
from mark.types import MemoryFragment

# Seed factor for memories outside a date window the question names (soft, not a filter).
OUT_OF_WINDOW = 0.7


def uses_score_fusion(config: FusionConfig, embedder: object) -> bool:
    """Whether this embedder's vectors should lead fusion (see module docstring).

    Providers without a ``semantic`` attribute are assumed to be models.
    """
    if config.mode != "auto":
        return config.mode == "score"
    return bool(getattr(embedder, "semantic", True))


def normalised_cosines(hits: list[VectorSearchResult], also: Iterable[str] = ()) -> dict[str, float]:
    """Each cosine as a fraction of the query's best cosine (best hit = 1).

    Absolute cosine scales differ a lot between models (bge-small gold
    matches sit around 0.6, potion-retrieval around 0.3), so cosines are
    made relative to the best hit before meeting term coverage. A ratio,
    not min-max: in a large store the depth-th neighbour is nearly as close
    as the first, and min-max would blow tiny cosine gaps up to the full
    [0, 1] range (measured: worse ranks on the retention A/B log).

    Ids in ``also`` (lexical hits) that fell outside the vector top-depth
    get the weakest hit's value: they are at most that similar, and a 0
    would bury an exact lexical match among many near-identical records
    (a ticket's "closed" line behind fifty "in progress" lines).
    """
    if not hits:
        return {}
    best = max(hit.score for hit in hits)
    if best <= 0:
        scores = {hit.fragment_id: 0.0 for hit in hits}
    else:
        scores = {hit.fragment_id: max(0.0, hit.score / best) for hit in hits}
    floor = min(scores.values())
    for fragment_id in also:
        scores.setdefault(fragment_id, floor)
    return scores


def rrf_seed_score(fused: float, max_fused: float, relevance: float) -> float:
    """Legacy blend: normalised RRF rank evidence plus calibrated relevance."""
    return 0.5 * min(1.0, fused / max_fused) + 0.5 * relevance


def max_rrf(query_count: int) -> float:
    """Best possible fused score: rank 1 in both channels for every query text."""
    return query_count * 2.0 / (RRF_K + 1)


def lexical_share(config: FusionConfig, anchors: list[str], contents: Iterable[str]) -> float:
    """Coverage weight for this query: higher only when a query anchor occurs in memory.

    Gating on an actual occurrence (not just on the query's shape) keeps
    hyphenated words such as "help-desk", which look like identifiers but
    match nothing, on the paraphrase path.
    """
    if anchors:
        for content in contents:
            norm = _anchor_norm(content)
            if any(_contains_anchor(norm, anchor) for anchor in anchors):
                return config.anchor_lexical_weight
    return config.lexical_weight


def score_fusion_seed(vector_norm: float, coverage: float, weight: float) -> float:
    """Convex combination of normalised cosine and term coverage, in [0, 1]."""
    return (1.0 - weight) * vector_norm + weight * coverage


def merge_max(target: dict[str, float], update: Mapping[str, float]) -> None:
    """Keep the best value per id across query texts (in place)."""
    for key, value in update.items():
        if value > target.get(key, -1.0):
            target[key] = value


def literal_hit(query_terms: list[str], coverage: Mapping[str, float]) -> bool:
    """True when the query has terms no stemmer splits (emoji, CJK) and a memory has them all."""
    return any(not term.isascii() for term in query_terms) and any(cov >= 0.999 for cov in coverage.values())


def scored_text(fragment: MemoryFragment) -> str:
    """The text term coverage is computed on: a document chunk without its heading-path line.

    Chunks start with "# Title › Section" (see ``mark.documents.chunking``);
    those words are shared by every chunk of the document, so counting them
    made the chunk holding the answer tie with every other chunk.
    """
    content = fragment.content
    if content.startswith("# ") and (fragment.metadata or {}).get("document_id"):
        newline = content.find("\n")
        if newline != -1:
            return content[newline + 1:]
    return content
