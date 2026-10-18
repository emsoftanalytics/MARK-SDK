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
group capped and the whole block held to ``budget_tokens``. The first
fact, the top open TODOs and open bugs are reserved (``RESERVED``): they
outrank the other lines and are shortened at small budgets, so a 100-token brief is not all
facts; a budget too small for the ``<mark_briefing>`` tags still returns
the most important line, cut to fit, without them. ``focus`` (a
query) puts the records retrieved for it first and filters every group to
records and facts that share a word with it. Retired records (superseded,
resolved) are never shown as current; a bug and its fix share one line; a
resolved bug is marked "(resolved)" (``brief_lines``); untrusted memories
are labelled. An empty store briefs as "(memory is empty)". Kinds are read from
``metadata["kind"]``, ``kind:<name>`` tags or a content prefix such as
"TODO:" / "DECISION:" / "BUG:" (see ``watch_kinds``). The model's own
replies (``mark:assistant``) are never part of a brief.
"""
from __future__ import annotations

import re
import time
from typing import Any

from mark.context.brief_lines import current_records, fix_pairs, record_line
from mark.context.outdated import resolved
from mark.context.budget import BudgetItem, ContextBudget, as_budget, render_lines, warn_if_empty
from mark.context.relevance import content_words
from mark.context.watch_kinds import closed_todos, kind_of, recent_first
from mark.context.watch_lines import Deduper, current_facts, fact_key, fact_line, trusted
from mark.context.watch_render import FACTS_HEADER, RELATED_HEADER, render_briefing
from mark.security.boundary import defuse_boundaries
from mark.context.watch_support import Briefing

TODO_HEADER = "Open TODOs:"
DECISION_HEADER = "Recent decisions:"
BUG_HEADER = "Recent bugs and fixes:"
CHANGE_HEADER = "Recent changes:"
#: Lines per group: the budget decides the rest, but no group may crowd out
#: the others when it is long (a backlog of 200 TODOs).
GROUP_CAPS = {"facts": 12, "todo": 6, "decision": 5, "bug": 5, "change": 5, "focus": 5}
#: What an empty store briefs as (an empty string looked like a failure).
EMPTY = "(memory is empty)"
#: Open TODOs / open bugs that are always shown (a small budget used to fill
#: with facts and drop every TODO).
#: The first fact is reserved too, so a small brief is not all TODOs; below
#: ``_FACT_RESERVE_MIN`` tokens there is room for one line only: a TODO.
RESERVED = {"facts": 1, "todo": 3, "bug": 2}
_FACT_RESERVE_MIN = 60
_MIN_RESERVED_CHARS = 40   # shorter than this a TODO line says too little
_DATE = re.compile(r"^(- (?:\([a-z]+\) )*)\d{4}-\d{2}-\d{2} ")
_GROUPS = (("todo", TODO_HEADER), ("decision", DECISION_HEADER), ("bug", BUG_HEADER),
           ("change", CHANGE_HEADER))


def brief(memory: Any, *, budget_tokens: int | ContextBudget = 400, focus: str | None = None,
          token_counter: Any = None, scan: int = 500, line_chars: int = 200) -> Briefing:
    """Build the session brief for *memory* (a ``MarkMemory``) under *budget_tokens*.

    ``scan`` bounds how many stored memories are read to find TODOs,
    decisions, bugs and changes (most important, then newest, first).
    """
    started = time.perf_counter()
    warn_if_empty(budget_tokens, "brief")
    budget = as_budget(budget_tokens, token_counter)
    scanned = _scan(memory, scan)
    facts = _facts(memory)
    if not scanned and not facts:   # say so: an empty answer looked like a failure (agent N8)
        return Briefing(text=EMPTY if budget.max_tokens else "",
                        latency_ms=round((time.perf_counter() - started) * 1000, 2))
    fragments = current_records([f for f in scanned if trusted(f)])
    focus_result = _retrieve(memory, focus) if focus else None
    # focus= is a filter: only facts and records that share a word with it.
    focus_words = content_words(focus or "")

    def on_focus(text: str, ref: str | None = None) -> bool:
        del ref   # retrieval ranks everything; only shared words make a record "about" the focus
        return not focus_words or bool(content_words(text) & focus_words)

    dedupe = Deduper()
    items: list[BudgetItem] = []
    shown: set[str] = set()

    def add(section: str, text: str, header: str, ref: str | None = None, check: bool = True) -> bool:
        if ref in shown or (check and not dedupe.fresh(text)):   # fact changes repeat a fact on purpose
            return False
        if ref:
            shown.add(ref)
        # Every line (records, facts, fact changes) has boundary-tag lookalikes defused.
        items.append(BudgetItem(section, defuse_boundaries(text), header, ref))
        return True

    # Facts block: facts about the focus first (highest priority), then the rest.
    focus_facts = current_facts([focus_result]) if focus_result is not None else []
    focus_keys = {fact_key(f) for f in focus_facts}
    for fact in focus_facts:
        add("relevant", fact_line(fact), FACTS_HEADER)
    for rank, fact in enumerate([f for f in facts if fact_key(f) not in focus_keys
                                 and on_focus(f"{f.subject} {f.attribute} {f.value}")][:GROUP_CAPS["facts"]]):
        top = rank < RESERVED["facts"] and budget.max_tokens >= _FACT_RESERVE_MIN
        add("relevant" if top else "facts", fact_line(fact), FACTS_HEADER)
    if focus_result is not None:
        _add_lines(add, "relevant", RELATED_HEADER, GROUP_CAPS["focus"],
                   [f for f in current_records([f for f in focus_result.fragments if trusted(f)])
                    if on_focus(f.content)], line_chars)
    closed = closed_todos(fragments)
    by_kind: dict[str, list[Any]] = {name: [] for name, _ in _GROUPS}
    for fragment in recent_first(fragments):
        kind = "bug" if kind_of(fragment) == "fix" else kind_of(fragment)   # bugs and fixes together
        if kind in by_kind and not (kind == "todo" and fragment.id in closed) \
                and on_focus(fragment.content, fragment.id):
            by_kind[kind].append(fragment)
    # A bug and the fix that closed it share one line; the fix is not repeated.
    bugs = [f for f in by_kind["bug"] if kind_of(f) == "bug"]
    pairs = fix_pairs(bugs, [f for f in by_kind["bug"] if kind_of(f) == "fix"])
    paired = {fix.id for fix in pairs.values()}
    by_kind["bug"] = [f for f in by_kind["bug"] if f.id not in paired]
    reserved = {"todo": by_kind["todo"][:RESERVED["todo"]],
                "bug": [f for f in by_kind["bug"] if kind_of(f) == "bug" and f.id not in pairs
                        and not resolved(f)][:RESERVED["bug"]]}
    # The budget's characters (~4 per token) left after tags, headers and the
    # first fact (~140), shared by the reserved lines (~17 of each is "- " and the id).
    count = max(1, sum(len(v) for v in reserved.values()))
    short = max(_MIN_RESERVED_CHARS, min(line_chars, (budget.max_tokens * 4 - 140) // count - 17))
    for name, header in _GROUPS:
        extra = [line for line in _fact_changes(memory, facts) if on_focus(line)] if name == "change" else []
        _add_lines(add, "records", header, GROUP_CAPS[name], by_kind[name], line_chars,
                   with_id=name == "todo", first=extra, pairs=pairs,
                   reserved={f.id for f in reserved.get(name, ())}, short=short)
    fit = budget.fit(items, render_briefing)
    if not fit.text and items and budget.max_tokens:
        # Too small for the tags and a header: the single most important line, cut to fit.
        plain = [BudgetItem(i.section, i.text, None, i.ref) for i in items]
        fit = ContextBudget(budget.max_tokens, token_counter=budget.token_counter).fit(plain, render_lines)
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
               line_chars: int, *, with_id: bool = False, first: list[str] | None = None,
               pairs: dict[str, Any] | None = None, reserved: set[str] = frozenset(),
               short: int = 0) -> None:
    """Up to *cap* new lines under *header*: the *first* text lines, then the records.

    *reserved* records are added to the "relevant" section (above facts) and
    shortened to *short* characters.
    """
    count = 0
    for line in first or []:
        if count < cap and add(section, line, header, check=False):
            count += 1
    for fragment in fragments:
        if count >= cap:
            return
        keep = fragment.id in reserved
        line = record_line(fragment, short if keep else line_chars, with_id=with_id,
                           fix=(pairs or {}).get(fragment.id))
        if keep and short < line_chars:   # shortened: the date is the first thing to go
            line = _DATE.sub(r"\1", line, count=1)
        if add("relevant" if keep else section, line, header, fragment.id):
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
