"""Answer evidence the score threshold cannot see, so a found answer is not reported as a gap.

The gap report compares the best calibrated relevance with the policy's
``min_confidence``. Two kinds of evidence say the answer *is* in memory even
when that number is low:

- **a fact names what was asked**: "Who is my boss?" selected
  ``user.boss = Priya Raman`` by its attribute. A fact chosen for the
  attribute the question names is an answer, whatever the wording of the
  memories;
- **one clear vector winner** (semantic embedders only): small static
  models such as potion-base-8M put correct paraphrase matches at cosine
  0.3-0.4, under the threshold, so the absolute value is not comparable
  across models. Calibrating against the query's own candidate pool is:
  a best match at least ``OUTLIER_RATIO`` times the runner-up and
  ``OUTLIER_MARGIN`` ahead of it stands out from everything else stored.
  Near-ties (where absent questions also land) keep their gap.

Neither rule applies when the question names an identifier or name that is
not in memory (missing anchors always report a gap).
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence

from mark.intelligence.fact_relevance import _attribute_text, _expand
from mark.intelligence.query_understanding import _content_terms

OUTLIER_RATIO = 2.0
OUTLIER_MARGIN = 0.15
# Below this cosine nothing counts as a winner, however far ahead.
OUTLIER_MIN = 0.25


def fact_answers(query: str, facts: Sequence) -> bool:
    """True when a returned current fact's attribute is named by the query (or a synonym of it)."""
    if not facts:
        return False
    terms = _expand(set(_content_terms(query)))
    for fact in facts:
        if getattr(fact, "is_current", True) and set(_content_terms(_attribute_text({"attribute": fact.attribute}))) & terms:
            return True
    return False


def vector_outlier(top_id: str | None, cosine: Mapping[str, float]) -> bool:
    """True when *top_id* is the query's nearest memory by a wide margin (see module docstring)."""
    if top_id is None or len(cosine) < 2:
        return False
    best_id = max(cosine, key=cosine.__getitem__)
    if best_id != top_id:
        return False
    best = cosine[best_id]
    second = max(value for key, value in cosine.items() if key != best_id)
    return best >= OUTLIER_MIN and best - second >= OUTLIER_MARGIN and best >= OUTLIER_RATIO * max(second, 0.0)
