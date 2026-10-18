"""Token budget and history replacement for :class:`MarkAgentMiddleware`.

**Budget** (``budget_tokens``). Everything MARK injects into one model call —
the ContextWatch briefing, retrieved memories and an optional thinking-model
plan — becomes one list of :class:`~mark.context.budget.BudgetItem` lines,
fitted under ONE hard cap (``ContextBudget.fit``) and rendered, wrapper
and block separator included, as the ``[MARK memory]`` block (so the text MARK
adds to the system prompt never exceeds the cap; a budget too small for the
wrapper plus one line injects nothing). The packaged skill the
middleware injects is charged to the same budget (``split_budget``): with
``budget_tokens=300`` and a 170-token skill, memory gets 130 tokens, so the
budget is the whole MARK footprint of the system prompt (usage test M7: a
3 KB skill on top of a "300-token" budget). Small budgets and the small/tiny
presets therefore get no skill unless one is named (see ``_skill_name``).

Retrieved records are filtered like briefing notes: the model's own stored
replies are kept apart under an "unverified" header (lowest priority), bare
questions, small talk and a stored copy of the current message are dropped,
and near-duplicates of briefing lines are shown once.

**History replacement** (``history="mark"``). Before each model call the
request's messages are cut to the last ``keep_last`` (see
:func:`mark.context.history.split_history` for the tool-call and current-turn
guarantees). Dropped messages are written to MARK first, with provenance
(:func:`~mark.adapters.langchain._middleware_provenance.store_options`:
user statements ``mark:user``, questions/requests ``mark:question`` /
``mark:request``; assistant replies low-trust ``mark:assistant``, or not
stored at all with ``store_assistant=False``; tool results ``mark:tool``,
failures also ``mark:error``; tool-call-only assistant messages store nothing) — in
``before_agent``/``before_model`` so retrieval and the briefing can already
see them, and again (if anything is missing) in ``wrap_model_call``. A
message is only dropped once its write succeeded; if writing fails, the full
history is sent for that call instead. Graph state is never modified: the
checkpointer keeps the full transcript, only the model's view is windowed.
Every message is stored at most once across all paths: the run-end reply and
the tool wrapper register the same :func:`message_key` as archival does.
"""
from __future__ import annotations

import threading
from collections import OrderedDict
from typing import Any

from mark.adapters.langchain._middleware_support import _RunState, _run_sync, _state_messages
from mark.context.budget import BudgetItem, ContextBudget, as_budget, render_lines
from mark.adapters.langchain._middleware_provenance import BLOCK_JOINT, store_options
from mark.context.history import message_key, message_role, split_history, turn_text
from mark.context.presets import split_budget
from mark.context.provenance import is_assistant
from mark.context.watch_lines import Deduper, is_echo, note_text, small_talk, trusted, words
from mark.context.watch_render import ASSISTANT_HEADER, render_briefing
from mark.context.watch_support import _one_line

RETRIEVED_HEADER = "Retrieved memory:"
REASONING_HEADER = "[MARK reasoning]"
_OUTSIDE_BRIEFING = (RETRIEVED_HEADER, REASONING_HEADER, ASSISTANT_HEADER)
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
    def _retrieval_items(value: Any, query: str = "", assistant_lines: int = 0) -> list[BudgetItem]:
        """Budget items for one retrieval result: current facts, then trusted records.

        Questions, small talk and the stored current message (*query*) are
        dropped. Assistant output is left out unless ``assistant_lines`` > 0
        (then it goes last, under its "unverified" header): in the M7 re-run
        llama3.2:1b copied an unverified reply that quoted its own system
        prompt straight back into its answer.
        """
        items: list[BudgetItem] = []
        for fact in getattr(value, "facts", None) or []:
            if getattr(fact, "is_current", True):
                text = fact.as_text() if hasattr(fact, "as_text") else str(fact)
                items.append(BudgetItem("facts", f"- {_one_line(text, _RECORD_CHARS)}", RETRIEVED_HEADER))
        query_words = words(query)
        replies: list[BudgetItem] = []
        for fragment in getattr(value, "fragments", None) or []:
            content = str(getattr(fragment, "content", "") or "")
            if not content.strip() or is_echo(fragment, query_words) or small_talk(fragment):
                continue
            ref = str(getattr(fragment, "id", "") or "") or None
            line = note_text(fragment, _RECORD_CHARS)
            if is_assistant(fragment):
                replies.append(BudgetItem("assistant", line, ASSISTANT_HEADER, ref))
            elif trusted(fragment):
                items.append(BudgetItem("records", line, RETRIEVED_HEADER, ref))
        return items + replies[:assistant_lines]

    def _render_budgeted(self, items: Any) -> str:
        """Briefing lines inside ``<mark_briefing>``, retrieved memory and plan after it."""
        inside = [i for i in items if i.group not in _OUTSIDE_BRIEFING]
        outside = [i for i in items if i.group in _OUTSIDE_BRIEFING]
        return "\n\n".join(part for part in (render_briefing(inside), render_lines(outside)) if part)

    def _budgeted_context(self, run: _RunState) -> str:
        """This step's watch, retrieval and plan lines fitted under the one budget."""
        items = list(run.watch_items)
        seen = {(i.ref or i.text) for i in items}
        dedupe = Deduper()
        for item in items:
            dedupe.fresh(item.text)
        for query in run.step_queries:
            for item in run.recall_items.get(query, ()):
                if (item.ref or item.text) not in seen and dedupe.fresh(item.text):
                    seen.add(item.ref or item.text)
                    items.append(item)
        if run.thoughts:
            items.append(BudgetItem("related", _one_line(run.thoughts, 4000), REASONING_HEADER))

        def render(selected: Any) -> str:   # measured exactly as it will be added
            return BLOCK_JOINT + self._memory_block(self._render_budgeted(selected)) if selected else ""

        fit = self._memory_budget().fit(items, render)
        run.last_usage = fit.usage
        self._record("budget", max_tokens=fit.usage.max_tokens, used_tokens=fit.usage.used_tokens,
                     included=fit.usage.included, dropped=fit.usage.dropped,
                     truncated=fit.usage.truncated)
        return self._render_budgeted(fit.items)

    def _memory_budget(self) -> ContextBudget:
        """The budget left for memory once the injected skill is charged to it."""
        skill = self._skill_block() if getattr(self, "_include_skill", False) else ""
        if not skill:
            return self._budget
        left = split_budget(self._budget.max_tokens, BLOCK_JOINT + skill, self._budget.token_counter)
        return ContextBudget(left, fill=self._budget.fill, token_counter=self._budget.token_counter)

    # ── history replacement ───────────────────────────────────────────────

    def _is_archived(self, key: str) -> bool:
        with self._archived_lock:
            return key in self._archived

    def _mark_archived(self, key: str) -> None:
        """Record *key* as stored (bounded LRU), so no other path stores it again."""
        with self._archived_lock:
            self._archived[key] = None
            self._archived.move_to_end(key)
            while len(self._archived) > _MAX_ARCHIVED:
                self._archived.popitem(last=False)

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
            role = message_role(message)
            text = turn_text(message)
            if role == "assistant" and not getattr(self, "_store_assistant", True):
                text = ""             # opted out: dropped without being stored
            if text:
                result = await self._backend_call("observe", text, agent_id=self._agent_id,
                                                  **store_options(message))
                if not result.ok:
                    continue          # stays unarchived: it will not be dropped
                written += 1
            self._mark_archived(message_key(message))
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
