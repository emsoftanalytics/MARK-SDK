"""Rendering of a ContextWatch briefing from a state snapshot.

One flat layout, with headers a small model can read without instructions
(usage test 0.2.0a9: extra system-prompt rules *lowered* accuracy, so the
briefing has to explain itself)::

    <mark_briefing>
    Corrections:
    - atlas-core port is 4444 now (since 2026-03-30); 9239 is outdated.
    Known facts (current):
    - atlas-core port = 4444 (was 9239; since 2026-03-30)
    Recent changes:
    - I moved from Porto to Braga last weekend.
    Related notes:
    - Biscuit's vet appointment moved to 21 October, same time.
    - (outdated) Biscuit has a vet appointment on 14 October at 10:30.
    Earlier assistant replies (unverified):
    - Assistant replied: ...
    Not in memory: Zorbatron-9 (say you don't know rather than guessing).
    </mark_briefing>

Every fact is listed once (no repeated block under a linked subject), and
notes are de-duplicated and filtered by provenance; greetings and sign-offs
are skipped (``watch_lines``).

Which lines survive a budget is decided by section priority
(``mark.context.budget``): corrections > unknown > relevant (what the
current request retrieved: its facts and best ``relevant_lines`` records) >
facts of subjects in play > their records > linked subjects > the rest of
the request's records > assistant replies. Rendering keeps the layout order;
the request's best lines come first; within each part notes are listed
newest first (declared event time, then storage time).

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

import re
from dataclasses import replace
from typing import Any, Sequence

from mark.context.budget import BudgetItem, ContextBudget, render_lines
from mark.context.outdated import outdated_ids
from mark.context.provenance import is_assistant
from mark.context.relevance import (
    above_floor,
    content_words,
    missing_relations,
    inside_longer_name,
    sequence_order,
    shares_words,
    stem,
)
from mark.context.watch_lines import (
    Deduper,
    current_facts,
    fact_key,
    fact_line,
    is_echo,
    note_text,
    small_talk,
    trusted,
    words,
)
from mark.context.watch_support import _anchor_norm, _contains_anchor, _timeline_key

CORRECTIONS_HEADER = "Corrections:"
FACTS_HEADER = "Known facts (current):"
RELATED_HEADER = "Related notes:"
CHANGES_HEADER = "Recent changes:"
ASSISTANT_HEADER = "Earlier assistant replies (unverified):"
OPEN, CLOSE = "<mark_briefing>", "</mark_briefing>"
#: Records of the current request that rank as "relevant" (the rest rank low).
RELEVANT_LINES = 4


def _unknown_line(unknown: list[str]) -> str:
    return "Not in memory: " + ", ".join(unknown) + " (say you don't know rather than guessing)."


def _anchor_records(result: Any, anchor: Any, limit: int | None) -> list[Any]:
    """The anchor's trusted records that name it, best match first."""
    records = [f for f in result.fragments
               if trusted(f) and _contains_anchor(_anchor_norm(f.content), anchor.key)
               and not inside_longer_name(f.content, anchor.label)]
    return records[:limit] if limit is not None else records


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
    limit; the budget then decides how much is shown. Optional snapshot keys:
    ``outdated`` ("mark" or "drop": lines a later statement replaced),
    ``floor`` (relevance floor ratio, 0 = off) and ``previous`` (old fact values).
    """
    task_result = snapshot.get("task_result")
    task = snapshot.get("task") or ""
    task_words = words(task)
    active = snapshot["active"]
    results = snapshot["results"]
    items = [BudgetItem("corrections", f"- {text}", CORRECTIONS_HEADER)
             for text in snapshot["corrections"]]
    # Outdated = retired by a revision, or replaced by a later line in this pool.
    pool = [f for f in _pool(snapshot) if trusted(f)]
    stale = outdated_ids(pool, _timeline_key)
    keep_stale = snapshot.get("outdated", "mark") != "drop" or bool(_PAST.search(task))
    facts = _fact_items(task_result, active, results, snapshot.get("previous") or {},
                        snapshot.get("fact_dates", True))
    # A fact read from a line that a later statement replaced is outdated too.
    items += [(replace(item, text=item.text.replace("- ", "- (outdated) ", 1)) if old else item)
              for item, old in ((item, item.ref in stale) for item in facts) if keep_stale or not old]

    dedupe = Deduper()
    seen: set[str] = set()

    def note(section: str, fragment: Any, header: str = RELATED_HEADER) -> BudgetItem | None:
        old = fragment.id in stale
        text = note_text(fragment, line_chars, outdated=old)
        if fragment.id in seen or is_echo(fragment, task_words) or small_talk(fragment) \
                or (old and not keep_stale) or not dedupe.fresh(text):
            return None
        seen.add(fragment.id)
        return BudgetItem(section, text, header, fragment.id)

    # Change statements for "now/before" questions: their own group, best match first.
    items += [item for item in (note("relevant", f, CHANGES_HEADER)
                                for f in snapshot.get("changes") or ()) if item]
    pairs = [(f, s) for f, s in zip(task_result.fragments, _scores(task_result)) if trusted(f)] \
        if task_result is not None else []
    task_records = above_floor(pairs, task, float(snapshot.get("floor", 0.0)))
    missing = missing_relations(task, [getattr(f, "content", "") for f in pool]) \
        if task and snapshot.get("relations", True) else []
    if missing:   # "my brother's name": only lines that share a word with the question
        asked = content_words(task) - {stem(m.split()[-1]) for m in missing} - {m.split()[-1] for m in missing}
        task_records = [f for f in task_records if shares_words(getattr(f, "content", ""), asked)]
    relevant_count = snapshot.get("relevant_lines", RELEVANT_LINES)
    picked: list[tuple[str, Any]] = [("relevant", f) for f in task_records[:relevant_count]]
    for anchor in active:
        result = results.get(anchor.key)
        if result is not None:
            section = "linked" if anchor.linked else "records"
            picked += [(section, f) for f in _anchor_records(result, anchor, per_anchor)]
    rest = task_records[relevant_count:]
    picked += [("related", f) for f in (rest[:per_anchor] if per_anchor is not None else rest)]
    # Best match first (pick order); a replaced line right after the line that
    # replaced it; the parts of a sequence (steps, days) together, oldest first.
    order = sequence_order(_after_replacement([f for _, f in picked], stale), _timeline_key)
    section_of: dict[str, str] = {}
    for section, fragment in picked:
        section_of.setdefault(fragment.id, section)
    notes = [note(section_of[f.id], f) for f in order]
    items += [item for item in notes if item is not None]

    limit = snapshot.get("assistant_lines", 0)
    if task_result is not None and limit:
        replies = [f for f in task_result.fragments if is_assistant(f)][:limit]
        items += [item for item in (note("assistant", f, ASSISTANT_HEADER) for f in replies) if item]
    unknown = list(snapshot["unknown"])
    unknown += [u for u in missing if u not in unknown and not _fact_mentions(items, u)]
    if unknown:
        items.append(BudgetItem("unknown", _unknown_line(unknown)))
    return items


#: Questions about an earlier value: outdated lines are the answer, so they stay.
_PAST = re.compile(r"\b(before|originally|original|old|previous(ly)?|used to|earlier|former(ly)?|"
                   r"chang\w*|history|first)\b", re.IGNORECASE)


def _fact_mentions(items: list[BudgetItem], missing: str) -> bool:
    """A fact line already answers it ("user brother = Tom")."""
    word = missing.split()[-1]
    return any(item.group == FACTS_HEADER and word in item.text.lower() for item in items)


def _pool(snapshot: dict[str, Any]) -> list[Any]:
    """Every fragment the snapshot holds (task, subjects, change statements), once."""
    out: dict[str, Any] = {}
    sources = [snapshot.get("task_result"), *snapshot["results"].values()]
    for result in sources:
        for fragment in getattr(result, "fragments", None) or ():
            out.setdefault(fragment.id, fragment)
    for fragment in [*(snapshot.get("changes") or ()), *(snapshot.get("change_pool") or ())]:
        out.setdefault(fragment.id, fragment)
    return list(out.values())


def _scores(result: Any) -> list[float]:
    scores = list(getattr(result, "scores", None) or [])
    return scores + [0.0] * (len(result.fragments) - len(scores))


def _after_replacement(fragments: list[Any], stale: dict[str, str | None]) -> list[Any]:
    """Move each replaced line right after the line that replaced it (newest first between them)."""
    ids = {f.id for f in fragments}
    moved = [f for f in fragments if stale.get(f.id) in ids]
    if not moved:
        return fragments
    out = [f for f in fragments if f not in moved]
    for fragment in sorted(moved, key=_timeline_key, reverse=True):
        anchor = next(i for i, f in enumerate(out) if f.id == stale[fragment.id])
        out.insert(anchor + 1, fragment)
    return out


def _fact_items(task_result: Any, active: list[Any], results: dict[str, Any],
                previous: dict[tuple[str, str], Any], dates: bool = True) -> list[BudgetItem]:
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
                # ref = the memory the fact was read from (for outdated checks).
                items.append(BudgetItem(section, fact_line(fact, previous.get(fact_key(fact)), since=dates),
                                        FACTS_HEADER, getattr(fact, "source_fragment_id", None)))
    return items
