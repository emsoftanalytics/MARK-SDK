"""One-call session brief: what an agent should know before it starts working.

A coding agent's first minute in a new session used to cost three or four
recall calls and ~1-1.5k tokens (usage test 3). :func:`brief` returns the
same orientation as one budgeted ``<mark_briefing>`` block::

    Known facts (current):
    - tasky test_command = pytest -q (since 2026-10-02)
    Open TODOs:
    - 2026-10-03 TODO: add pagination to /tasks (id 3f2a9c1b)
    Recent decisions:
    - 2026-10-02 DECISION: SQLite for storage, no ORM
    Recent bugs and fixes:
    - 2026-10-04 BUG: due dates shift by one day in UTC-5
    Recent changes:
    - tasky port: 8000 -> 8080 (since 2026-10-05)

Rules: current facts first, then open TODOs (a TODO is closed by a status
or tag, by a memory that ``resolves`` it, or by a later DONE/FIX note that
names it), decisions, bugs/fixes and changes, newest first, deduped, each
group capped and the whole block held to ``budget_tokens``. ``focus`` (a
query) puts the records retrieved for it first. Kinds are read from
``metadata["kind"]``, ``kind:<name>`` tags or a content prefix such as
"TODO:" / "DECISION:" / "BUG:" (see ``watch_kinds``). The model's own
replies (``mark:assistant``) are never part of a brief.
"""
from __future__ import annotations

import re
import time
from typing import Any

from mark.context.budget import BudgetItem, ContextBudget, as_budget
from mark.context.watch_kinds import closed_todos, kind_of, recent_first, when_of
from mark.context.watch_lines import Deduper, current_facts, fact_key, fact_line, trusted
from mark.context.watch_render import FACTS_HEADER, RELATED_HEADER, render_briefing
from mark.context.watch_support import Briefing, _one_line

TODO_HEADER = "Open TODOs:"
DECISION_HEADER = "Recent decisions:"
BUG_HEADER = "Recent bugs and fixes:"
CHANGE_HEADER = "Recent changes:"
#: Lines per group: the budget decides the rest, but no group may crowd out
#: the others when it is long (a backlog of 200 TODOs).
GROUP_CAPS = {"facts": 12, "todo": 6, "decision": 5, "bug": 5, "change": 5, "focus": 5}
_LEADING_DATE = re.compile(r"^\s*\[?\d{4}-\d{2}-\d{2}")
_GROUPS = (("todo", TODO_HEADER), ("decision", DECISION_HEADER), ("bug", BUG_HEADER),
           ("change", CHANGE_HEADER))


def brief(memory: Any, *, budget_tokens: int | ContextBudget = 400, focus: str | None = None,
          token_counter: Any = None, scan: int = 500, line_chars: int = 200) -> Briefing:
    """Build the session brief for *memory* (a ``MarkMemory``) under *budget_tokens*.

    ``scan`` bounds how many stored memories are read to find TODOs,
    decisions, bugs and changes (most important, then newest, first).
    """
    started = time.perf_counter()
    budget = as_budget(budget_tokens, token_counter)
    fragments = [f for f in _scan(memory, scan) if trusted(f)]
    facts = _facts(memory)
    focus_result = _retrieve(memory, focus) if focus else None
    dedupe = Deduper()
    items: list[BudgetItem] = []
    shown: set[str] = set()

    def add(section: str, text: str, header: str, ref: str | None = None, check: bool = True) -> bool:
        if ref in shown or (check and not dedupe.fresh(text)):   # fact changes repeat a fact on purpose
            return False
        if ref:
            shown.add(ref)
        items.append(BudgetItem(section, text, header, ref))
        return True

    # Facts block: facts about the focus first (highest priority), then the rest.
    focus_facts = current_facts([focus_result]) if focus_result is not None else []
    focus_keys = {fact_key(f) for f in focus_facts}
    for fact in focus_facts:
        add("relevant", fact_line(fact), FACTS_HEADER)
    for fact in [f for f in facts if fact_key(f) not in focus_keys][:GROUP_CAPS["facts"]]:
        add("facts", fact_line(fact), FACTS_HEADER)
    if focus_result is not None:
        _add_lines(add, "relevant", RELATED_HEADER, GROUP_CAPS["focus"],
                   [f for f in focus_result.fragments if trusted(f)], line_chars)
    closed = closed_todos(fragments)
    by_kind: dict[str, list[Any]] = {name: [] for name, _ in _GROUPS}
    for fragment in recent_first(fragments):
        kind = "bug" if kind_of(fragment) == "fix" else kind_of(fragment)   # bugs and fixes together
        if kind in by_kind and not (kind == "todo" and fragment.id in closed):
            by_kind[kind].append(fragment)
    for name, header in _GROUPS:
        extra = _fact_changes(memory, facts) if name == "change" else []
        _add_lines(add, "records", header, GROUP_CAPS[name], by_kind[name], line_chars,
                   with_id=name == "todo", first=extra)
    fit = budget.fit(items, render_briefing)
    return Briefing(
        text=fit.text,
        memory_ids=[item.ref for item in fit.items if item.ref],
        latency_ms=round((time.perf_counter() - started) * 1000, 2),
        usage=fit.usage,
        items=tuple(items),
    )


def _scan(memory: Any, limit: int) -> list[Any]:
    """Stored memories to classify (empty when the memory cannot list)."""
    try:
        return list(memory.list(limit=limit))
    except Exception:
        return []


def _retrieve(memory: Any, query: str) -> Any:
    try:
        return memory.retrieve_sync(query)
    except Exception:  # a brief without its focus is still a brief
        return None


def _facts(memory: Any) -> list[Any]:
    """Current facts, the user's first, then the most recently changed."""
    try:
        facts = [f for f in memory.facts.current() if f.is_current]
    except Exception:  # stores without facts
        return []
    facts.sort(key=lambda f: (str(f.subject).lower() != "user", -f.valid_from.timestamp()))
    return facts


def _add_lines(add: Any, section: str, header: str, cap: int, fragments: list[Any],
               line_chars: int, *, with_id: bool = False, first: list[str] | None = None) -> None:
    """Up to *cap* new lines under *header*: the *first* text lines, then the records."""
    count = 0
    for line in first or []:
        if count < cap and add(section, line, header, check=False):
            count += 1
    for fragment in fragments:
        if count >= cap:
            return
        if add(section, _record_line(fragment, line_chars, with_id=with_id), header, fragment.id):
            count += 1


def _fact_changes(memory: Any, facts: list[Any]) -> list[str]:
    """"subject attribute: old -> new (since date)" for current facts that changed."""
    lines: list[tuple[float, str]] = []
    for fact in facts:
        try:
            history = memory.facts.history(fact.subject, fact.attribute)
        except Exception:
            return []
        if len(history) > 1 and history[-2].value != fact.value:
            since = fact.valid_from.date().isoformat()
            lines.append((fact.valid_from.timestamp(),
                          f"- {fact.subject} {fact.attribute}: {history[-2].value} -> {fact.value} (since {since})"))
    return [line for _, line in sorted(lines, reverse=True)]


def _record_line(fragment: Any, line_chars: int, *, with_id: bool = False) -> str:
    """``- 2026-10-03 content`` (plus a short id for TODOs, so they can be resolved)."""
    when = when_of(fragment)
    content = getattr(fragment, "content", "") or ""
    dated = _LEADING_DATE.match(content) is not None      # the text already starts with its date
    date = f"{when.date().isoformat()} " if when is not None and not dated else ""
    suffix = f" (id {fragment.id[:8]})" if with_id and fragment.id else ""
    return f"- {date}{_one_line(fragment.content, line_chars)}{suffix}"
