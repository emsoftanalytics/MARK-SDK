# SPDX-License-Identifier: MIT
# Copyright (c) 2025 MARK Contributors
"""Memory gap report types."""
from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field


class GapSeverity(str, Enum):
    """
    Mirrors gap_detector.py severity levels.
    Used by the RetrievalPipeline and surfaced in RetrievalResult
    so the agent can decide how to react.
    """
    NONE     = "none"       # retrieval was confident
    LOW      = "low"        # slightly below threshold; agent may proceed
    MEDIUM   = "medium"     # notable gap; agent should note uncertainty
    HIGH     = "high"       # significant gap; agent should search
    CRITICAL = "critical"   # no results; agent MUST NOT answer from memory


class GapReport(BaseModel):
    """
    Structured gap analysis attached to a RetrievalResult.

    The pipeline fills this when top_score is below the policy threshold.
    The agent reads it to decide whether to trigger a web search, tell the
    user it doesn't know, or proceed with a low-confidence answer.

    Transparency rule: when gap_severity is HIGH or CRITICAL the agent MUST
    surface this to the user rather than hallucinating.
    """
    model_config = ConfigDict(frozen=True)

    severity:          GapSeverity  = GapSeverity.NONE
    reason:            str          = ""
    suggested_action:  str          = "none"     # "none"|"web_search"|"ask_user"|"refuse"
    missing_coverage:  float        = 0.0        # 0.0–1.0, fraction of query unaddressed
    search_query_hint: Optional[str] = None      # refined query for web search
    confidence:        float        = Field(default=1.0, ge=0.0, le=1.0)
    missing:           List[str]    = Field(default_factory=list)
    action_hint:       str          = "answer"

    @classmethod
    def from_score(
        cls,
        top_score:    float,
        result_count: int,
        threshold:    float,
    ) -> "GapReport":
        """Build a GapReport from retrieval scores. Mirrors detect_gap() logic."""
        threshold = max(0.0, float(threshold))
        top_score = max(0.0, float(top_score))

        if result_count == 0:
            return cls(
                severity=GapSeverity.CRITICAL,
                reason="No memory fragments found for this query.",
                suggested_action="web_search",
                missing_coverage=1.0,
                confidence=0.0,
            )
        if threshold == 0:
            return cls()
        if top_score < threshold * 0.5:
            return cls(
                severity=GapSeverity.HIGH,
                reason=f"Top score {top_score:.3f} is well below threshold {threshold:.3f}.",
                suggested_action="web_search",
                missing_coverage=round(1.0 - top_score / threshold, 3),
                confidence=top_score,
            )
        if top_score < threshold:
            severity = GapSeverity.MEDIUM if top_score >= threshold * 0.75 else GapSeverity.HIGH
            return cls(
                severity=severity,
                reason=f"Top score {top_score:.3f} below threshold {threshold:.3f}.",
                suggested_action="web_search" if severity == GapSeverity.HIGH else "none",
                missing_coverage=round(1.0 - top_score / threshold, 3),
                confidence=top_score,
            )
        return cls()

    @property
    def should_search(self) -> bool:
        """Return True when external lookup is advised."""
        return self.suggested_action == "web_search"

    @property
    def must_not_answer(self) -> bool:
        """True when the agent MUST NOT generate an answer from stale/empty memory."""
        return self.severity == GapSeverity.CRITICAL
