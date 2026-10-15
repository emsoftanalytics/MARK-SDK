"""Context watch: anticipate what an agent needs and have it ready before it asks.

A :class:`ContextWatch` follows the agent's stream — user messages, its own
outputs, tool calls and tool results — and, without the agent issuing any
search, keeps a small working set of relevant facts and memories warm:

1. **Observe.** Every event is scanned for anchors (service names, ticket ids,
   people, known fact subjects). New anchors trigger a background prefetch,
   so retrieval happens while the agent is still thinking or a tool is
   running, not on the critical path of the next model call.
2. **Anticipate.** The working set is ranked by how recently and how often
   each anchor came up; anchors that fall out of the conversation decay away.
3. **Brief.** :meth:`ContextWatch.briefing` returns a compact, LLM-friendly
   block grouped by subject — current facts first, then the most relevant
   records — within a character budget, or a hard token budget
   (``budget_tokens``, priority-filled; see ``mark.context.budget``). It
   never blocks on search unless the caller asks it to wait.
4. **Correct.** When the agent states a value that memory knows was
   superseded ("atlas-core listens on 9239" after it moved to 4444), the next
   briefing leads with a correction. Only values found in a fact's history
   trigger this, so the agent is never "corrected" on a guess.

Unknown anchors are reported as "Not in memory", so the agent can say it
doesn't know instead of improvising.

    watch = ContextWatch(mark.runtime.memory("__mark__"))
    watch.observe("Can you check why atlas-core is timing out?", role="user")
    ...
    system_prompt += watch.briefing().text   # before every model call
    watch.observe(model_output, role="assistant")

Plain agent loops with OpenAI-style message lists use
``watch.observe_messages(messages)`` and :func:`inject` (``watch_messages``);
``ContextWatch.for_small_models(memory)`` and ``ContextWatch.for_tiny_models``
are measured presets for small models (``mark.context.presets``).

Provenance: the model's own replies stored in memory (``mark:assistant``)
never confirm a subject, never become notes about the user and are shown, if
at all, only as "unverified" (``mark.context.provenance``).

The background machinery lives in ``watch_prefetch`` and the data types and
text helpers in ``watch_support``; both are re-exported here.
"""
from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

from mark.context.budget import BudgetItem, ContextBudget, as_budget
from mark.context.watch_messages import _WatchMessagesMixin, inject
from mark.context.watch_render import briefing_items, render_briefing, render_chars
from mark.context.watch_prefetch import _WatchPrefetchMixin
# Some of these are not used here; they stay importable from this module.
from mark.context.watch_support import (
    _ROLES,
    _TOKEN_RE,
    Briefing,
    _Anchor,
    _anchor_norm,
    _contains_anchor,
    _label_for,
    _mentions_value,
    _one_line,
    _timeline_key,
    _wait,
)

logger = logging.getLogger("mark.context.watch")


class ContextWatch(_WatchMessagesMixin, _WatchPrefetchMixin):
    """Follows an agent's stream and keeps anticipated context ready.

    memory          a ``MarkMemory`` (``runtime.memory(agent_id)``)
    budget_chars    maximum briefing size
    max_anchors     how many active subjects a briefing covers
    per_anchor      memories shown per subject
    ttl_turns       turns an anchor stays active without being mentioned
    line_chars      maximum length of one record bullet
    background      prefetch on a worker thread (False: inline, deterministic)
    budget_tokens   hard token cap (int or ContextBudget); replaces budget_chars
    token_counter   ``str -> int`` for budget_tokens (default: ceil(chars / 4))
    budget_fill     True: lift max_anchors/per_anchor and fill up to budget_tokens
    relevant_lines  records retrieved for the current request that rank as
                    "relevant" (above subject records) under a budget
    assistant_lines earlier assistant replies (low trust, labelled
                    "unverified") the briefing may show (default 0: none)
    """

    def __init__(
        self,
        memory: Any,
        *,
        budget_chars: int = 1800,
        max_anchors: int = 6,
        per_anchor: int = 3,
        ttl_turns: int = 6,
        background: bool = True,
        line_chars: int = 240,
        budget_tokens: int | ContextBudget | None = None,
        token_counter: Any = None,
        budget_fill: bool = False,
        relevant_lines: int = 4,
        assistant_lines: int = 0,
    ) -> None:
        self._memory = memory
        self.budget_chars = budget_chars
        self.max_anchors = max_anchors
        self.per_anchor = per_anchor
        self.ttl_turns = ttl_turns
        self.line_chars = line_chars
        self.relevant_lines = max(0, int(relevant_lines))
        self.assistant_lines = max(0, int(assistant_lines))
        self.budget = as_budget(budget_tokens, token_counter, budget_fill)
        self._lock = threading.RLock()
        self._anchors: dict[str, _Anchor] = {}
        self._results: dict[str, Any] = {}          # anchor key -> RetrievalResult
        self._task_results: dict[str, Any] = {}     # task text -> RetrievalResult
        self._task: str = ""
        self._corrections: dict[str, str] = {}      # correction key -> text
        self._turn = 0
        self._seq = 0
        self.follow_links = 3   # linked subjects prefetched per new anchor (0 disables)
        self._pending: set[Future] = set()
        # One worker: prefetches run in order and never compete with each other.
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mark-watch") if background else None
        self._subjects_cache: tuple[float, set[str]] = (0.0, set())

    # ── stream input ──────────────────────────────────────────────────────────

    def observe(self, text: str, *, role: str = "user") -> list[str]:
        """Feed one event from the agent's stream. Returns newly seen anchors."""
        if role not in _ROLES:
            raise ValueError(f"role must be one of {sorted(_ROLES)}")
        text = (text or "").strip()
        if not text:
            return []
        found = self._extract_anchors(text)
        with self._lock:
            if role == "user":
                self._turn += 1
                self._task = text
            turn = self._turn
            self._seq += 1
            new: list[str] = []
            for key, label in found:
                anchor = self._anchors.get(key)
                if anchor is None:
                    self._anchors[key] = _Anchor(key, label, turn, turn, last_seq=self._seq,
                                                 assistant_only=role == "assistant")
                    new.append(key)
                else:
                    if role != "assistant":
                        anchor.assistant_only = False
                    if anchor.linked:  # an anticipated subject is now actually discussed
                        anchor.linked = False
                        anchor.known = None if anchor.known is False else anchor.known
                    anchor.last_turn = turn
                    anchor.last_seq = self._seq
                    anchor.mentions += 1
            self._expire(turn)
            # Only *new* anchors are prefetched; a user message also refreshes
            # the results for the request as a whole.
            queries = [self._anchors[key].label for key in new]
            task = self._task if role == "user" else None
        if role == "assistant":
            self._check_claims(text, [key for key, _ in found])
        if queries or task:
            self._schedule(queries, task)
        return new

    def observe_tool_call(self, name: str, arguments: Any = None) -> list[str]:
        """Feed a tool call: its arguments often name exactly what is needed next.

        Only the arguments are scanned; the tool's own name is not a subject.
        """
        del name
        if arguments is None:
            return []
        if isinstance(arguments, dict):
            text = " ".join(str(value) for value in arguments.values())
        else:
            text = str(arguments)
        return self.observe(text, role="tool")

    # ── output ────────────────────────────────────────────────────────────────

    def briefing(self, *, wait: float = 0.0, budget_tokens: Any = None) -> Briefing:
        """Assemble the current briefing. ``wait`` > 0 lets running prefetches finish first.

        ``budget_tokens`` (an int or a :class:`ContextBudget`) overrides the
        watch's own token budget for this call. With a token budget the
        briefing is priority-filled under a hard cap and ``Briefing.usage``
        reports what was kept; without one the ``budget_chars`` renderer runs.
        """
        started = time.perf_counter()
        if wait > 0:
            self.flush(timeout=wait)
        budget = as_budget(budget_tokens) if budget_tokens is not None else self.budget
        lifted = budget is not None and budget.fill   # fill: the budget, not the counts, limits
        # Snapshot shared state under the lock, then render without holding it,
        # so a running prefetch is never blocked by (or blocks) a briefing.
        with self._lock:
            # Mentioned subjects first (most recent first), anticipated links after.
            # Subjects only the model has named ("nectar-rich", "Olá Ana") are
            # shown only once memory confirms them, and never as "Not in memory".
            active = sorted(
                (a for a in self._anchors.values()
                 if a.known is not False and not (a.assistant_only and a.known is not True)),
                key=lambda a: (not a.linked, a.last_seq, a.mentions), reverse=True,
            )[: None if lifted else self.max_anchors]
            snapshot = {
                "active": active,
                "unknown": [a.label for a in self._anchors.values()
                            if a.known is False and not a.linked and not a.assistant_only
                            and self._turn - a.last_turn <= 1],
                "results": {a.key: self._results.get(a.key) for a in active},
                "task_result": self._task_results.get(self._task),
                "corrections": list(self._corrections.values()),
                "task": self._task,
                "relevant_lines": self.relevant_lines,
                "assistant_lines": self.assistant_lines,
            }
            pending = len(self._pending)

        usage = None
        items: tuple[BudgetItem, ...] = ()
        if budget is None:
            text, shown = render_chars(snapshot, budget_chars=self.budget_chars,
                                       per_anchor=self.per_anchor, line_chars=self.line_chars)
        else:
            items = tuple(briefing_items(snapshot, per_anchor=None if lifted else self.per_anchor,
                                         line_chars=self.line_chars))
            fit = budget.fit(items, render_briefing)
            text, usage = fit.text, fit.usage
            shown = [item.ref for item in fit.items if item.ref]
        return Briefing(
            text=text,
            anchors=[a.label for a in active],
            corrections=snapshot["corrections"],
            unknown=snapshot["unknown"],
            memory_ids=shown,
            pending=pending,
            latency_ms=round((time.perf_counter() - started) * 1000, 2),
            usage=usage,
            items=items,
        )

    def flush(self, timeout: float | None = None) -> bool:
        """Wait for running prefetches. Returns True when none are left."""
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            with self._lock:
                pending = list(self._pending)
            if not pending:
                return True
            remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
            if remaining == 0.0:
                return False
            _wait(pending[0], remaining)

    def reset(self) -> None:
        """Forget the conversation (keep memory)."""
        self.flush(timeout=5)
        with self._lock:
            self._anchors.clear()
            self._results.clear()
            self._task_results.clear()
            self._corrections.clear()
            self._task = ""
            self._turn = 0

    def close(self) -> None:
        """Stop the background worker."""
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None

    def __enter__(self) -> "ContextWatch":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()
