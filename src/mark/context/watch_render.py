"""Rendering of a ContextWatch briefing from a state snapshot.

Two renderers share one layout (corrections, then one group per subject with
its current facts and records, then records related to the request, then the
"Not in memory" line):

* :func:`render_chars` — the original character-budget renderer, used when no
  token budget is set. Kept byte-for-byte compatible with earlier releases.
* :func:`briefing_items` + :func:`wrap_briefing` — the same lines as
  :class:`~mark.context.budget.BudgetItem` objects, for priority fill under a
  :class:`~mark.context.budget.ContextBudget`.

A *snapshot* is the dict ``ContextWatch.briefing`` takes under its lock:
``active`` anchors, ``unknown`` labels, per-anchor ``results``,
``task_result`` and ``corrections``.
"""
from __future__ import annotations

from typing import Any, Sequence

from mark.context.budget import BudgetItem, render_lines
from mark.context.watch_support import _anchor_norm, _contains_anchor, _one_line, _timeline_key

CORRECTIONS_HEADER = "Corrections (memory has newer values):"
RELATED_HEADER = "Related to the current request:"
OPEN, CLOSE = "<mark_briefing>", "</mark_briefing>"


def _unknown_line(unknown: list[str]) -> str:
    return "Not in memory: " + ", ".join(unknown) + " (say so rather than guessing)."


def _fact_line(fact: Any) -> str:
    return f"- {fact.attribute}: {fact.value} (since {fact.valid_from.date().isoformat()})"


def _anchor_records(result: Any, anchor: Any, seen: set[str], limit: int | None) -> list[Any]:
    """The anchor's records not shown yet, newest first (undated last)."""
    records = [f for f in result.fragments
               if f.id not in seen and _contains_anchor(_anchor_norm(f.content), anchor.key)]
    records = records[:limit] if limit is not None else records
    records.sort(key=_timeline_key, reverse=True)
    return records


def wrap_briefing(body: str) -> str:
    """The ``<mark_briefing>`` block around *body* ("" stays "")."""
    return f"{OPEN}\n{body}\n{CLOSE}" if body else ""


def render_briefing(items: Sequence[BudgetItem]) -> str:
    """Budget renderer for watch briefings: grouped lines inside the briefing tag."""
    return wrap_briefing(render_lines(items)) if items else ""


def render_chars(snapshot: dict[str, Any], *, budget_chars: int, per_anchor: int,
                 line_chars: int) -> tuple[str, list[str]]:
    """Original renderer: lines are appended until ``budget_chars`` is reached."""
    lines: list[str] = []
    used = 0
    shown: list[str] = []

    def add(line: str) -> bool:
        nonlocal used
        if used + len(line) + 1 > budget_chars:
            return False
        lines.append(line)
        used += len(line) + 1
        return True

    if snapshot["corrections"]:
        add(CORRECTIONS_HEADER)
        for text in snapshot["corrections"]:
            add(f"- {text}")
    seen_ids: set[str] = set()
    for anchor in snapshot["active"]:
        result = snapshot["results"].get(anchor.key)
        if result is None:
            continue
        facts = [f for f in getattr(result, "facts", []) if f.is_current]
        records = _anchor_records(result, anchor, seen_ids, per_anchor)
        if not facts and not records:
            continue
        if not add(f"{anchor.label}{' (linked)' if anchor.linked else ''}:"):
            break
        for fact in facts:
            add(_fact_line(fact))
        for fragment in records:
            if add(f"- {_one_line(fragment.content, line_chars)}"):
                seen_ids.add(fragment.id)
                shown.append(fragment.id)
    task_result = snapshot["task_result"]
    if task_result is not None:
        extra = [f for f in task_result.fragments if f.id not in seen_ids][:per_anchor]
        if extra:
            add(RELATED_HEADER)
            for fragment in extra:
                if add(f"- {_one_line(fragment.content, line_chars)}"):
                    seen_ids.add(fragment.id)
                    shown.append(fragment.id)
    if snapshot["unknown"]:
        add(_unknown_line(snapshot["unknown"]))
    return wrap_briefing("\n".join(lines)), shown


def briefing_items(snapshot: dict[str, Any], *, per_anchor: int | None,
                   line_chars: int) -> list[BudgetItem]:
    """Every candidate line of the briefing, in render order, tagged by section.

    ``per_anchor=None`` (budget ``fill=True``) lifts the per-subject record
    limit; the budget then decides how much is shown. A record is listed
    under the first subject that claims it only.
    """
    items = [BudgetItem("corrections", f"- {text}", CORRECTIONS_HEADER)
             for text in snapshot["corrections"]]
    seen: set[str] = set()
    for anchor in snapshot["active"]:
        result = snapshot["results"].get(anchor.key)
        if result is None:
            continue
        header = f"{anchor.label}{' (linked)' if anchor.linked else ''}:"
        facts = [f for f in getattr(result, "facts", []) if f.is_current]
        for fact in facts:
            items.append(BudgetItem("linked" if anchor.linked else "facts", _fact_line(fact), header))
        for fragment in _anchor_records(result, anchor, seen, per_anchor):
            seen.add(fragment.id)
            items.append(BudgetItem("linked" if anchor.linked else "records",
                                    f"- {_one_line(fragment.content, line_chars)}", header, fragment.id))
    task_result = snapshot["task_result"]
    if task_result is not None:
        extra = [f for f in task_result.fragments if f.id not in seen]
        for fragment in extra[:per_anchor] if per_anchor is not None else extra:
            seen.add(fragment.id)
            items.append(BudgetItem("related", f"- {_one_line(fragment.content, line_chars)}",
                                    RELATED_HEADER, fragment.id))
    if snapshot["unknown"]:
        items.append(BudgetItem("unknown", _unknown_line(snapshot["unknown"])))
    return items
