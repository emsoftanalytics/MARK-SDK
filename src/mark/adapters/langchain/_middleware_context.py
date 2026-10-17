"""Token budget and history replacement for :class:`MarkAgentMiddleware`.

**Budget** (``budget_tokens``). Everything MARK injects into one model call —
the ContextWatch briefing, retrieved memories and an optional thinking-model
plan — becomes one list of :class:`~mark.context.budget.BudgetItem` lines,
fitted under ONE hard cap (``ContextBudget.fit``) and rendered, wrapper
included, as the ``[MARK project memory]`` block. The packaged skill text
(static instructions; ``include_skill=False`` removes it) is not counted.

**History replacement** (``history="mark"``). Before each model call the
request's messages are cut to the last ``keep_last`` (see
:func:`mark.context.history.split_history` for the tool-call and current-turn
guarantees). Dropped messages are written to MARK first — in
``before_agent``/``before_model`` so retrieval and the briefing can already
see them, and again (if anything is missing) in ``wrap_model_call``. A
message is only dropped once its write succeeded; if writing fails, the full
history is sent for that call instead. Graph state is never modified: the
checkpointer keeps the full transcript, only the model's view is windowed.
"""
from __future__ import annotations

import threading
from collections import OrderedDict
from typing import Any

from mark.adapters.langchain._middleware_support import _RunState, _run_sync, _state_messages
from mark.context.budget import BudgetItem, ContextBudget, as_budget, render_lines
from mark.context.history import message_key, split_history, turn_text
from mark.context.watch_render import render_briefing
from mark.context.watch_support import _one_line

RETRIEVED_HEADER = "Retrieved memory:"
REASONING_HEADER = "[MARK reasoning]"
_OUTSIDE_BRIEFING = (RETRIEVED_HEADER, REASONING_HEADER)
_MAX_ARCHIVED = 8192      # message keys remembered as stored (bounded LRU)
_RECORD_CHARS = 400       # one retrieved record line in budget mode


class _ContextControlMixin:
    """Budget + history mixin. Expects ``_backend_call``, ``_record``, ``_memory_block``,
    ``_agent_id`` and ``_thinking_model`` on the host class."""

    def _init_context_controls(self, history: str, keep_last: int,
                               budget_tokens: int | ContextBudget | None,
                               token_counter: Any) -> None:
        if history not in ("full", "mark"):
            raise ValueError("history must be 'full' or 'mark'")
        if keep_last < 1:
            raise ValueError("keep_last must be >= 1")
        self._history = history
        self._keep_last = keep_last
        self._budget = as_budget(budget_tokens, token_counter)
        self._archived: OrderedDict[str, None] = OrderedDict()
        self._archived_lock = threading.Lock()

    # ── budget ────────────────────────────────────────────────────────────

    @staticmethod
    def _retrieval_items(value: Any) -> list[BudgetItem]:
        """Budget items for one retrieval result: current facts, then records."""
        items: list[BudgetItem] = []
        for fact in getattr(value, "facts", None) or []:
            if getattr(fact, "is_current", True):
                text = fact.as_text() if hasattr(fact, "as_text") else str(fact)
                items.append(BudgetItem("facts", f"- {_one_line(text, _RECORD_CHARS)}", RETRIEVED_HEADER))
        for fragment in getattr(value, "fragments", None) or []:
            content = str(getattr(fragment, "content", "") or "")
            if content.strip():
                items.append(BudgetItem("records", f"- {_one_line(content, _RECORD_CHARS)}",
                                        RETRIEVED_HEADER, str(getattr(fragment, "id", "") or "") or None))
        return items

    def _render_budgeted(self, items: Any) -> str:
        """Briefing lines inside ``<mark_briefing>``, retrieved memory and plan after it."""
        inside = [i for i in items if i.group not in _OUTSIDE_BRIEFING]
        outside = [i for i in items if i.group in _OUTSIDE_BRIEFING]
        return "\n\n".join(part for part in (render_briefing(inside), render_lines(outside)) if part)

    def _budgeted_context(self, run: _RunState) -> str:
        """This step's watch, retrieval and plan lines fitted under the one budget."""
        items = list(run.watch_items)
        seen = {(i.ref or i.text) for i in items}
        for query in run.step_queries:
            for item in run.recall_items.get(query, ()):
                if (item.ref or item.text) not in seen:
                    seen.add(item.ref or item.text)
                    items.append(item)
        if run.thoughts:
            items.append(BudgetItem("related", _one_line(run.thoughts, 4000), REASONING_HEADER))

        def render(selected: Any) -> str:   # measured exactly as it will be injected
            return self._memory_block(self._render_budgeted(selected)) if selected else ""

        fit = self._budget.fit(items, render)
        run.last_usage = fit.usage
        self._record("budget", max_tokens=fit.usage.max_tokens, used_tokens=fit.usage.used_tokens,
                     included=fit.usage.included, dropped=fit.usage.dropped,
                     truncated=fit.usage.truncated)
        return self._render_budgeted(fit.items)

    # ── history replacement ───────────────────────────────────────────────

    def _unarchived(self, messages: Any) -> list[Any]:
        """Messages the window would drop that MARK does not hold yet."""
        if getattr(self, "_history", "full") != "mark":
            return []
        dropped = split_history(list(messages or []), self._keep_last).dropped
        with self._archived_lock:
            return [m for m in dropped if message_key(m) not in self._archived]

    async def _archive(self, messages: Any) -> int:
        """Store every not-yet-stored message the window drops; return how many were written."""
        written = 0
        for message in self._unarchived(messages):
            text = turn_text(message)
            if text:
                result = await self._backend_call("observe", text, agent_id=self._agent_id)
                if not result.ok:
                    continue          # stays unarchived: it will not be dropped
                written += 1
            with self._archived_lock:
                self._archived[message_key(message)] = None
                while len(self._archived) > _MAX_ARCHIVED:
                    self._archived.popitem(last=False)
        if written:
            self._record("history_archive", count=written)
        return written

    async def _archive_state(self, state: Any) -> None:
        """Hook helper: archive what the window drops from the graph state's messages."""
        if getattr(self, "_history", "full") == "mark":
            await self._archive(_state_messages(state))

    def _archive_request_sync(self, request: Any) -> None:
        """wrap_model_call safety net (sync path)."""
        try:
            if self._unarchived(getattr(request, "messages", None)):
                _run_sync(self._archive(getattr(request, "messages", None)))
        except Exception as exc:
            self._hook_failed("history_archive", exc)

    async def _aarchive_request(self, request: Any) -> None:
        """wrap_model_call safety net (async path)."""
        try:
            if self._unarchived(getattr(request, "messages", None)):
                await self._archive(getattr(request, "messages", None))
        except Exception as exc:
            self._hook_failed("history_archive", exc)

    def _windowed_request(self, request: Any) -> Any:
        """*request* with its messages cut to the window (``history="mark"`` only)."""
        if getattr(self, "_history", "full") != "mark":
            return request
        messages = list(getattr(request, "messages", None) or [])
        split = split_history(messages, self._keep_last)
        if not split.dropped:
            return request
        if self._unarchived(messages):
            self._record("history_full", reason="unarchived", dropped=len(split.dropped))
            return request            # never drop what MARK could not store
        self._record("history_window", kept=len(split.kept), dropped=len(split.dropped))
        return request.override(messages=split.kept)
