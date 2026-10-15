"""Rendering of a ContextWatch briefing from a state snapshot.

One flat layout, with headers a small model can read without instructions
(usage test 0.2.0a9: extra system-prompt rules *lowered* accuracy, so the
briefing has to explain itself)::

    <mark_briefing>
    Corrections:
    - atlas-core port is 4444 now (since 2026-03-30); 9239 is outdated.
    Known facts (current):
    - atlas-core port = 4444 (since 2026-03-30)
    Related notes:
    - User said: Biscuit's vet appointment moved to 21 October, same time.
    Earlier assistant replies (unverified):
    - Assistant replied: ...
    Not in memory: Zorbatron-9 (say you don't know rather than guessing).
    </mark_briefing>

Every fact is listed once (no repeated block under a linked subject), and
notes are de-duplicated and filtered by provenance (``watch_lines``).

Which lines survive a budget is decided by section priority
(``mark.context.budget``): corrections > unknown > relevant (what the
current request retrieved: its facts and best ``relevant_lines`` records) >
facts of subjects in play > their records > linked subjects > the rest of
the request's records > assistant replies. Rendering keeps the layout order.

* :func:`briefing_items` lists the candidate lines as budget items.
* :func:`render_briefing` renders a selection inside ``<mark_briefing>``.
* :func:`render_chars` is the character-budget path (no token budget): the
  same items, priority-filled under ``budget_chars``.

A *snapshot* is the dict ``ContextWatch.briefing`` takes under its lock:
``active`` anchors, ``unknown`` labels, per-anchor ``results``,
``task_result``, ``corrections`` and the optional ``task`` text,
``relevant_lines`` and ``assistant_lines``.
"""
from __future__ import annotations

from typing import Any, Sequence

from mark.context.budget import BudgetItem, ContextBudget, render_lines
from mark.context.provenance import is_assistant
from mark.context.watch_lines import (
    Deduper,
    current_facts,
    fact_key,
    fact_line,
    is_echo,
    trusted,
    words,
)
from mark.context.watch_support import _anchor_norm, _contains_anchor, _one_line, _timeline_key

CORRECTIONS_HEADER = "Corrections:"
FACTS_HEADER = "Known facts (current):"
RELATED_HEADER = "Related notes:"
ASSISTANT_HEADER = "Earlier assistant replies (unverified):"
OPEN, CLOSE = "<mark_briefing>", "</mark_briefing>"
#: Records of the current request that rank as "relevant" (the rest rank low).
RELEVANT_LINES = 4


def _unknown_line(unknown: list[str]) -> str:
    return "Not in memory: " + ", ".join(unknown) + " (say you don't know rather than guessing)."


def _anchor_records(result: Any, anchor: Any, limit: int | None) -> list[Any]:
    """The anchor's trusted records that name it, newest first (undated last)."""
    records = [f for f in result.fragments
               if trusted(f) and _contains_anchor(_anchor_norm(f.content), anchor.key)]
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
    """Character-budget path: the same candidate lines, priority-filled under ``budget_chars``."""
    items = briefing_items(snapshot, per_anchor=per_anchor, line_chars=line_chars)
    # The wrapper tags are not part of the character budget (as in earlier releases).
    overhead = len(wrap_briefing("x")) - 1
    fit = ContextBudget(max(0, budget_chars) + overhead, token_counter=len).fit(items, render_briefing)
    return fit.text, [item.ref for item in fit.items if item.ref]


def briefing_items(snapshot: dict[str, Any], *, per_anchor: int | None,
                   line_chars: int) -> list[BudgetItem]:
    """Every candidate line of the briefing, in render order, tagged by section.

    ``per_anchor=None`` (budget ``fill=True``) lifts the per-subject record
    limit; the budget then decides how much is shown.
    """
    task_result = snapshot.get("task_result")
    task_words = words(snapshot.get("task") or "")
    active = snapshot["active"]
    results = snapshot["results"]
    items = [BudgetItem("corrections", f"- {text}", CORRECTIONS_HEADER)
             for text in snapshot["corrections"]]
    items += _fact_items(task_result, active, results)

    dedupe = Deduper()
    seen: set[str] = set()

    def note(section: str, fragment: Any, header: str = RELATED_HEADER) -> BudgetItem | None:
        text = f"- {_one_line(fragment.content, line_chars)}"
        if fragment.id in seen or is_echo(fragment, task_words) or not dedupe.fresh(text):
            return None
        seen.add(fragment.id)
        return BudgetItem(section, text, header, fragment.id)

    task_records = [f for f in (task_result.fragments if task_result is not None else [])
                    if trusted(f)]
    relevant_count = snapshot.get("relevant_lines", RELEVANT_LINES)
    notes: list[BudgetItem | None] = [note("relevant", f) for f in task_records[:relevant_count]]
    for anchor in active:
        result = results.get(anchor.key)
        if result is not None:
            section = "linked" if anchor.linked else "records"
            notes += [note(section, f) for f in _anchor_records(result, anchor, per_anchor)]
    rest = task_records[relevant_count:]
    notes += [note("related", f) for f in (rest[:per_anchor] if per_anchor is not None else rest)]
    items += [item for item in notes if item is not None]

    limit = snapshot.get("assistant_lines", 0)
    if task_result is not None and limit:
        replies = [f for f in task_result.fragments if is_assistant(f)][:limit]
        items += [item for item in (note("assistant", f, ASSISTANT_HEADER) for f in replies) if item]
    if snapshot["unknown"]:
        items.append(BudgetItem("unknown", _unknown_line(snapshot["unknown"])))
    return items


def _fact_items(task_result: Any, active: list[Any], results: dict[str, Any]) -> list[BudgetItem]:
    """One line per current fact: the request's facts rank as relevant, then subjects in play."""
    items: list[BudgetItem] = []
    shown: set[tuple[str, str]] = set()
    sources = [("relevant", task_result)] if task_result is not None else []
    sources += [("linked" if a.linked else "facts", results.get(a.key)) for a in active]
    for section, result in sources:
        if result is None:
            continue
        for fact in current_facts([result]):
            if fact_key(fact) not in shown:
                shown.add(fact_key(fact))
                items.append(BudgetItem(section, fact_line(fact), FACTS_HEADER))
    return items
