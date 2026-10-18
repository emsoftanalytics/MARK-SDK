"""Turning ranked candidates into the shown list: one score, few chunks per document, no padding.

Three rules keep what a caller sees honest and small:

1. **Displayed score = order.** Each result's shown score is its
   calibrated relevance (the evidence the gap report judges) times the
   reranker's penalties (retired, contradicted, assistant text). Results
   are sorted by that number, the ranking score only breaking ties, so a
   ``[1.000]`` can no longer sit below a ``[0.750]``. Later stages that
   reorder on purpose (an update ahead of what it replaces, the newest of a
   state band first) show the evidence they displaced: :func:`display_scores`.
2. **Documents don't crowd.** At most ``MAX_PER_DOCUMENT`` chunks of one
   document reach the list; the next best memories take the freed slots.
3. **Relevance floor.** Results far below the top (under
   ``RELATIVE_FLOOR`` of it) or with almost no evidence (under
   ``ABSOLUTE_FLOOR``, ``TIME_SCOPED_FLOOR`` inside a time window) are
   dropped instead of padding the list, so a filter that leaves nothing
   relevant returns nothing. A pure listing question with a time window
   ("what happened since Monday?") keeps its in-window results. The best
   match always stays outside a time window (the gap report says how weak
   it is), and so do pinned memories (importance >= ``PINNED``).

Expects nothing from the host class beyond the methods defined here.
"""
from __future__ import annotations

from mark.intelligence.query_understanding import _content_terms
from mark.types import MemoryFragment

MAX_PER_DOCUMENT = 2
ABSOLUTE_FLOOR = 0.1
# Inside a since/until window the scope is already narrow: fuzzy near-misses are noise.
TIME_SCOPED_FLOOR = 0.2
# Memories stored as canon (importance >= this, e.g. a block's style guide) are never floored.
PINNED = 0.9
RELATIVE_FLOOR = 0.3
# Words of a "what happened / what's new" listing question: no topic to match.
_LISTING_TERMS = frozenset(_content_terms(
    "happen happened do did done new news change changes changed update updates updated discuss discussed "
    "talk talked work worked going on anything everything"))


def demoted_relevance(relevance: float, signals: dict | None) -> float:
    """Relevance times the reranker's penalty factor (1.0 when it reported none)."""
    return relevance * float((signals or {}).get("demotion", 1.0))


def materialise(ranked, fragments_by_id: dict, relevance: dict, fetch, retrievable: set, top_k: int):
    """Ranked candidates -> (fragments, scores, displayed relevances), display-sorted and diversified.

    State is re-checked here: a plugin scorer may return ids that were never
    fetched, or whose state changed since.
    """
    fragments: list[MemoryFragment] = []
    scores: list[float] = []
    relevances: list[float] = []
    for result in ranked:
        fragment = fragments_by_id.get(result.fragment_id) or fetch(result.fragment_id)
        if fragment is None or fragment.state not in retrievable:
            continue
        fragments.append(fragment)
        scores.append(result.score)
        relevances.append(demoted_relevance(relevance.get(result.fragment_id, 0.0), getattr(result, "signals", None)))
    return diversify(*sort_by_display(fragments, scores, relevances), top_k)


def sort_by_display(fragments: list[MemoryFragment], scores: list[float], relevances: list[float]):
    """Stable sort by displayed relevance; equal values keep the ranking order."""
    order = sorted(range(len(fragments)), key=lambda i: -relevances[i])
    return [fragments[i] for i in order], [scores[i] for i in order], [relevances[i] for i in order]


def diversify(fragments: list[MemoryFragment], scores: list[float], relevances: list[float], top_k: int,
              per_document: int = MAX_PER_DOCUMENT):
    """Keep at most *per_document* chunks of each document, then cut to *top_k*."""
    seen: dict[str, int] = {}
    keep: list[int] = []
    for i, fragment in enumerate(fragments):
        document = (fragment.metadata or {}).get("document_id")
        if document:
            if seen.get(document, 0) >= per_document:
                continue
            seen[document] = seen.get(document, 0) + 1
        keep.append(i)
        if len(keep) >= top_k:
            break
    return [fragments[i] for i in keep], [scores[i] for i in keep], [relevances[i] for i in keep]


def display_scores(relevances: list[float]) -> list[float]:
    """Non-increasing shown scores: each result shows the best evidence at or below its rank.

    Already ordered lists are unchanged. Only deliberate reorders (an update
    ahead of what it replaces, the newest of a state band first, temporal
    bands) break the order; a memory moved up then shows the evidence of the
    one it displaced, which keeps its own value, so the replaced statement
    stays visible as "outdated" instead of falling under the floor.
    """
    shown = list(relevances)
    for i in range(len(shown) - 2, -1, -1):
        shown[i] = max(shown[i], shown[i + 1])
    return shown


def apply_floor(query: str, fragments: list[MemoryFragment], scores: list[float], relevances: list[float],
                *, time_scoped: bool = False):
    """Drop the tail that is far below the top or carries almost no evidence (see module docstring)."""
    if not relevances:
        return fragments, scores, relevances
    if time_scoped and set(_content_terms(query)) <= _LISTING_TERMS:
        return fragments, scores, relevances
    floor = max(TIME_SCOPED_FLOOR if time_scoped else ABSOLUTE_FLOOR, RELATIVE_FLOOR * relevances[0])
    # The best match stays (the gap report says how weak it is) unless a time
    # window was asked for; pinned memories (importance >= PINNED) always stay.
    keep = [i for i, rel in enumerate(relevances)
            if rel >= floor or (i == 0 and not time_scoped) or fragments[i].importance >= PINNED]
    return [fragments[i] for i in keep], [scores[i] for i in keep], [relevances[i] for i in keep]
