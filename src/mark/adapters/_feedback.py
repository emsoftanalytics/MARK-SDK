"""Feedback payload helpers for :class:`mark.adapters.backend.LocalMarkBackend`.

Split out of ``backend.py`` to keep that module small: they turn a free-form
agent feedback payload into fragment ids and a score in ``[-1, 1]``.
"""
from __future__ import annotations

from typing import Any

#: Default feedback score per agent outcome when the caller omits ``score``.
_OUTCOME_SCORES: dict[str, float] = {
    "completed": 1.0,
    "success": 1.0,
    "accepted": 1.0,
    "partial": 0.5,
    "failed": -1.0,
    "rejected": -1.0,
}


def _feedback_fragment_ids(payload: dict[str, Any]) -> list[str]:
    ids: list[str] = []
    raw = payload.get("fragment_ids")
    if raw:
        ids.extend(str(item) for item in raw if item)
    for key in ("fragment_id", "record_id"):
        value = payload.get(key)
        if value:
            ids.append(str(value))
    return list(dict.fromkeys(ids))


def _feedback_score(payload: dict[str, Any]) -> float | None:
    score = payload.get("score")
    if score is None:
        score = _OUTCOME_SCORES.get(str(payload.get("outcome", "")).lower())
    if score is None:
        return None
    return max(-1.0, min(1.0, float(score)))
