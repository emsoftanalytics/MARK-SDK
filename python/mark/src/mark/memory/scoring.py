"""Keyword-overlap scoring helpers for simple retrieval."""
from __future__ import annotations

import re
from datetime import datetime, timezone

from mark.memory.record import MemoryRecord

TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+")

# Transparent legacy fallback weights for the old block/record API.
# These are not production-calibrated MARK Cloud weights, and this scorer is
# not used by the current SQLite fragment RetrievalPipeline.
_OVERLAP_WEIGHT = 0.52
_IMPORTANCE_WEIGHT = 0.22
_CONFIDENCE_WEIGHT = 0.16
_RECENCY_WEIGHT = 0.06
_FEEDBACK_WEIGHT = 0.04


def keyword_overlap(query: str, content: str) -> float:
    """Return the keyword overlap ratio between query and content."""
    query_terms = set(_tokens(query))
    content_terms = set(_tokens(content))
    if not query_terms or not content_terms:
        return 0.0
    return len(query_terms & content_terms) / len(query_terms)


def score_record(query: str, record: MemoryRecord) -> float:
    """Score a record against the query."""
    overlap = keyword_overlap(query, record.content)
    recency = _recency_score(record.created_at)
    feedback = max(-1.0, min(1.0, record.feedback_score))
    normalized_feedback = (feedback + 1.0) / 2.0
    return (
        overlap * _OVERLAP_WEIGHT
        + record.importance * _IMPORTANCE_WEIGHT
        + record.confidence * _CONFIDENCE_WEIGHT
        + recency * _RECENCY_WEIGHT
        + normalized_feedback * _FEEDBACK_WEIGHT
    )


def _tokens(text: str) -> list[str]:
    return [match.group(0).lower() for match in TOKEN_RE.finditer(text)]


def _recency_score(created_at: datetime) -> float:
    now = datetime.now(timezone.utc)
    age_days = max(0.0, (now - created_at).total_seconds() / 86400)
    if age_days <= 1:
        return 1.0
    if age_days >= 30:
        return 0.0
    return 1.0 - (age_days / 30)
