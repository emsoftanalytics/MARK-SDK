"""Token budget for everything MARK puts into a model's context.

A :class:`ContextBudget` is a hard cap, in tokens, on one injected block
(a ContextWatch briefing, or the middleware's whole per-call injection).
Candidate lines are :class:`BudgetItem` objects tagged with a *section*; the
budget includes them by section priority until the cap is reached:

    corrections > unknown ("Not in memory") > relevant > facts > records >
    linked > related > assistant

``relevant`` holds what the current request retrieved (its facts and best
records): the line that answers the question beats a record that merely
names a subject in play (usage test 0.2.0a9: at 200 tokens 11 of 25 answers
were cut while subject records were kept). ``assistant`` is the model's own
earlier output (low trust, see ``mark.context.provenance``): kept only when
everything else fits.

Rules (all deterministic — same items, same counter, same result):

* **Hard cap.** The final rendered text is measured with the token counter
  and items are removed (lowest priority first) until it fits, so wrapper
  tags and headers count too. An empty selection renders as "" (0 tokens).
* **Whole items only.** A line is either included or dropped. The single
  exception: when the *first* item of the highest-priority non-empty section
  does not fit on its own, it is cut to fit and ends with "…" (a lone
  correction is better half-shown than lost). Every other item that does not
  fit is skipped and smaller lower-priority items may still use the space.
* **Order is preserved.** Priority decides what is kept, not where it goes:
  kept items are rendered in the order they were given.

Token counting: the default counter is ``ceil(len(text) / 4)`` — free and
dependency-free, but only an estimate (English prose is ~4 chars/token for
GPT/Llama-style BPE tokenizers; code, numbers, non-Latin scripts and
identifiers often use more tokens per char). Pass ``token_counter`` (any
``str -> int`` callable, e.g. ``lambda s: len(enc.encode(s))`` for tiktoken
or a Hugging Face tokenizer) for an exact cap; MARK never imports one itself.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Sequence

TokenCounter = Callable[[str], int]

#: Section priority, highest first.
SECTIONS: tuple[str, ...] = ("corrections", "unknown", "relevant", "facts", "records", "linked",
                             "related", "assistant")
_RANK = {name: rank for rank, name in enumerate(SECTIONS)}
_ELLIPSIS = "…"
_MIN_TRUNCATED_CHARS = 8   # a cut item shorter than this carries no information


def estimate_tokens(text: str) -> int:
    """The default counter: ``ceil(len(text) / 4)`` (an estimate; see module doc)."""
    return math.ceil(len(text) / 4) if text else 0


@dataclass(frozen=True)
class BudgetItem:
    """One candidate line. ``group`` is a header line rendered before the first kept
    item of the group (charged once); ``ref`` is an opaque id (e.g. a fragment id)."""

    section: str
    text: str
    group: str | None = None
    ref: str | None = None

    def __post_init__(self) -> None:
        if self.section not in _RANK:
            raise ValueError(f"unknown budget section {self.section!r}; expected one of {SECTIONS}")


@dataclass(frozen=True)
class SectionUsage:
    """Items of one section that were included / dropped, and their token cost."""

    included: int = 0
    dropped: int = 0
    tokens: int = 0


@dataclass(frozen=True)
class BudgetUsage:
    """What a budgeted block used: ``used_tokens`` of ``max_tokens``, per section."""

    max_tokens: int
    used_tokens: int
    sections: dict[str, SectionUsage] = field(default_factory=dict)
    truncated: bool = False

    @property
    def included(self) -> int:
        return sum(s.included for s in self.sections.values())

    @property
    def dropped(self) -> int:
        return sum(s.dropped for s in self.sections.values())

    @property
    def remaining(self) -> int:
        return max(0, self.max_tokens - self.used_tokens)

    def as_dict(self) -> dict[str, Any]:
        """JSON-ready form (for diagnostics and logs)."""
        return {
            "max_tokens": self.max_tokens, "used_tokens": self.used_tokens,
            "truncated": self.truncated,
            "sections": {name: vars(s).copy() for name, s in self.sections.items()},
        }


@dataclass(frozen=True)
class BudgetFit:
    """Result of :meth:`ContextBudget.fit`: kept items (input order), rendered text, usage."""

    items: tuple[BudgetItem, ...]
    text: str
    usage: BudgetUsage


def render_lines(items: Sequence[BudgetItem]) -> str:
    """Plain rendering: group headers, then item lines, newline-joined."""
    lines: list[str] = []
    group: str | None = None
    for item in items:
        if item.group and item.group != group:
            lines.append(item.group)
        group = item.group
        lines.append(item.text)
    return "\n".join(lines)


class ContextBudget:
    """A hard token cap with priority fill.

    max_tokens     the cap (0 means "inject nothing")
    fill           False: the budget only caps what the producer would inject
                   anyway; True: producers lift their own count limits (records
                   per subject, subjects per briefing) and fill up to the cap
    token_counter  ``str -> int``; default :func:`estimate_tokens`
    """

    def __init__(self, max_tokens: int, *, fill: bool = False,
                 token_counter: TokenCounter | None = None) -> None:
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens < 0:
            raise ValueError("max_tokens must be a non-negative int")
        self.max_tokens = max_tokens
        self.fill = bool(fill)
        self.token_counter = token_counter

    def __repr__(self) -> str:
        return f"ContextBudget(max_tokens={self.max_tokens}, fill={self.fill})"

    def count(self, text: str) -> int:
        """Tokens in *text* under this budget's counter ("" is always 0)."""
        if not text:
            return 0
        counter = self.token_counter or estimate_tokens
        return max(0, int(counter(text)))

    def fit(self, items: Sequence[BudgetItem],
            render: Callable[[Sequence[BudgetItem]], str] = render_lines) -> BudgetFit:
        """Select items by priority so that ``render(selected)`` fits the cap.

        *render* must return "" for an empty selection; it is called on the
        final selection, so whatever it adds (tags, headers) is counted.
        """
        items = list(items)
        # Priority order: section rank, then input position (stable => deterministic).
        order = sorted(range(len(items)), key=lambda i: (_RANK[items[i].section], i))
        # Fixed overhead of the renderer (wrapper tags), measured on an empty line.
        overhead = self.count(render([BudgetItem("records", "")])) if items else 0
        remaining = self.max_tokens - overhead
        chosen: dict[int, BudgetItem] = {}
        groups: set[str] = set()
        truncated = False
        for rank, index in enumerate(order):
            item = items[index]
            cost = self.count(item.text) + 1
            if item.group and item.group not in groups:
                cost += self.count(item.group) + 1
            if cost <= remaining:
                chosen[index] = item
            elif rank == 0:
                cut = self._truncate(item, render)
                if cut is None:
                    continue
                chosen[index], truncated = cut, True
                cost = self.count(render([cut]))
            else:
                continue
            remaining -= cost
            if item.group:
                groups.add(item.group)
        # Verification: the estimate above is per line; the cap is on the real text.
        keep = sorted(chosen)
        text = render([chosen[i] for i in keep]) if keep else ""
        while keep and self.count(text) > self.max_tokens:
            worst = max(keep, key=lambda i: (_RANK[items[i].section], i))
            keep.remove(worst)
            truncated = truncated and worst != order[0]
            text = render([chosen[i] for i in keep]) if keep else ""
        return BudgetFit(tuple(chosen[i] for i in keep), text,
                         self._usage(items, chosen, keep, text, truncated))

    def _truncate(self, item: BudgetItem,
                  render: Callable[[Sequence[BudgetItem]], str]) -> BudgetItem | None:
        """Longest prefix of *item* (plus "…") whose rendering fits, or None."""
        low, high, best = _MIN_TRUNCATED_CHARS, len(item.text) - 1, None
        while low <= high:   # binary search on the character length
            middle = (low + high) // 2
            candidate = replace(item, text=item.text[:middle].rstrip() + _ELLIPSIS)
            if self.count(render([candidate])) <= self.max_tokens:
                best, low = candidate, middle + 1
            else:
                high = middle - 1
        return best

    def _usage(self, items: list[BudgetItem], chosen: dict[int, BudgetItem],
               keep: list[int], text: str, truncated: bool) -> BudgetUsage:
        kept = set(keep)
        sections: dict[str, SectionUsage] = {}
        for index, item in enumerate(items):
            old = sections.get(item.section, SectionUsage())
            if index in kept:
                sections[item.section] = SectionUsage(
                    old.included + 1, old.dropped, old.tokens + self.count(chosen[index].text))
            else:
                sections[item.section] = SectionUsage(old.included, old.dropped + 1, old.tokens)
        ordered = {name: sections[name] for name in SECTIONS if name in sections}
        return BudgetUsage(self.max_tokens, self.count(text), ordered, truncated)


def warn_if_empty(value: Any, where: str) -> None:
    """Warn when a caller asks for a 0-token budget.

    ``budget_tokens=0`` used to return "" silently (MCP test m2: a client
    passing 0 for "no limit" got no memory and no hint why). It still means
    "inject nothing", but now says so; negative values raise in ContextBudget.
    """
    if isinstance(value, int) and not isinstance(value, bool) and value == 0:
        import warnings

        warnings.warn(f"{where}: budget_tokens=0 injects no memory at all; pass None for the "
                      "default budget", UserWarning, stacklevel=3)


def as_budget(value: Any, token_counter: TokenCounter | None = None,
              fill: bool = False) -> ContextBudget | None:
    """Coerce ``None`` / an int / a :class:`ContextBudget` into a budget (or None)."""
    if value is None or isinstance(value, ContextBudget):
        return value
    return ContextBudget(value, fill=fill, token_counter=token_counter)
