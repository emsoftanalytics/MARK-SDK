"""Model call wrapper mixin for :class:`MarkAgentMiddleware`.

``(a)wrap_model_call`` is a pure transform: it appends the packaged
middleware skill (once) and the ``[MARK memory]`` block to the system
message, with no backend I/O, and releases the run's state when the model
call raises. The tool wrapper lives in ``_middleware_tools`` (mixed in here).
"""

from __future__ import annotations

from typing import Any, Callable

from mark.adapters.langchain._middleware_provenance import memory_block, wrapper_chars
from mark.adapters.langchain._middleware_support import _SystemMessage, content_text
from mark.adapters.langchain._middleware_tools import _ToolWrapperMixin


_LONG_SKILL = "mark-memory-middleware"
_COMPACT_SKILL = "mark-memory-compact"


class _CallWrappersMixin(_ToolWrapperMixin):
    """System-message injection (and, via the base, tool-result observation)."""

    def _system_text(self, system_message: Any) -> str:
        if system_message is None:
            return ""
        content = getattr(system_message, "content", "")
        if isinstance(content, str):
            return content
        blocks = getattr(system_message, "content_blocks", None)
        if blocks is None:
            blocks = content if isinstance(content, list) else []
        return content_text(list(blocks or []))

    def _score_skill(self, system_text: str) -> dict[str, Any]:
        lowered = system_text.lower()
        checks = {
            "frontmatter": "---" in system_text and "name:" in lowered and "description:" in lowered,
            "mark_named": "mark" in lowered and "middleware" in lowered,
            "recall": "recall" in lowered or "retrieve" in lowered,
            "ground": "ground" in lowered or "current evidence" in lowered,
            "act": "act" in lowered or "use live" in lowered,
            "remember": "remember" in lowered or "observe" in lowered,
            "safety": "do not store secrets" in lowered or "no secrets" in lowered,
        }
        score = round(sum(checks.values()) / len(checks), 2)
        return {"present": checks["mark_named"], "score": score, "checks": checks}

    # ── (a)wrap_model_call: inject MARK context into system message ───────

    def _memory_block(self, context: str) -> str:
        """The injected block (wrapper included); the budget measures this text."""
        return memory_block(context)

    def _context_for_request(self, request: Any) -> str:
        state = getattr(request, "state", None) or {}
        if isinstance(state, dict):
            if "mark_context" in state:
                return state.get("mark_context") or ""
        elif hasattr(state, "mark_context"):
            return getattr(state, "mark_context") or ""
        # Fallback for requests built outside the agent graph: the run's
        # latest context, or the run's cached recall for the human query.
        with self._runs_lock:
            run = self._runs.get(self._run_id(state))
        if run is None:
            return ""
        if run.last_context:
            return run.last_context
        query = self._get_human_query_from_messages(getattr(request, "messages", []))
        return run.recall_cache.get(query, "")

    def _prepare_model_request(self, request: Any) -> Any:
        """Pure transform shared by ``wrap_model_call`` and ``awrap_model_call``.

        Returns *request* overridden with MARK's system-message additions, or
        the original request when there is nothing to add (or on any error).
        """
        try:
            request = self._windowed_request(request)   # history="mark" only
            context = self._context_for_request(request)
            system_message = getattr(request, "system_message", None)
            system_text = self._system_text(system_message)
            additions: list[dict[str, str]] = []

            # Auto-inject the packaged middleware skill once, so agents know how
            # to treat MARK-injected memory without manual prompt wiring.
            skill_block = self._skill_block() if self._include_skill else ""
            marker = "mark-memory-middleware" if self._skill_name() == _LONG_SKILL else skill_block
            if skill_block and marker not in system_text:
                additions.append({"type": "text", "text": skill_block})

            if context:
                skill_hash = self._hash(system_text)
                if skill_hash != self._last_skill_hash:
                    skill = self._score_skill(system_text + skill_block)
                    self._record("skill", skill_hash=skill_hash, **skill)
                    self._last_skill_hash = skill_hash
                if self._budget is None:   # a budgeted context is already capped
                    context = self._truncate_context(context, reserve=wrapper_chars())
                context_hash = self._hash(context)
                block = self._memory_block(context)
                # Inject on every model request; skip only if this exact block
                # is already part of the system prompt.
                if block in system_text:
                    self._record("inject_skip", chars=len(context), context_hash=context_hash)
                else:
                    self._record("inject", chars=len(context), context_hash=context_hash)
                    additions.append({"type": "text", "text": block})

            if not additions:
                return request
            new_content = self._system_content_blocks(system_message) + additions
            return request.override(system_message=_SystemMessage(content=new_content))
        except Exception as exc:
            self._hook_failed("wrap_model_call", exc)
            return request

    def wrap_model_call(
        self,
        request: Any,
        handler: Callable[[Any], Any],
    ) -> Any:
        """Sync: inject MARK context into the system message, then call the model.

        Appends a ``[MARK memory]`` block to ``request.system_message``
        via ``request.override()``.  No backend calls, except that with
        ``history="mark"`` messages not yet stored are written before being dropped.
        """
        self._archive_request_sync(request)
        prepared = self._prepare_model_request(request)
        try:
            return handler(prepared)
        except BaseException as exc:
            self._on_run_error(getattr(request, "state", None), exc)
            raise

    async def awrap_model_call(
        self,
        request: Any,
        handler: Callable[[Any], Any],
    ) -> Any:
        """Async: inject MARK context into the system message, then call the model."""
        await self._aarchive_request(request)
        prepared = self._prepare_model_request(request)
        try:
            return await handler(prepared)
        except BaseException as exc:
            self._on_run_error(getattr(request, "state", None), exc)
            raise

    def _system_content_blocks(self, system_message: Any) -> list[dict[str, Any]]:
        """Return LangChain-compatible system content blocks.

        LangChain model requests may carry no system message, a string content
        system message, or an already-blocked message.  `wrap_model_call` should
        be a pure transform and must not assume a specific provider shape.
        """
        if system_message is None:
            return []
        blocks = getattr(system_message, "content_blocks", None)
        if blocks is not None:
            return list(blocks)
        content = getattr(system_message, "content", "")
        if isinstance(content, list):
            return list(content)
        if content:
            return [{"type": "text", "text": str(content)}]
        return []

    def _packaged_skill(self) -> str:
        """Lazy-load the packaged mark-memory-middleware skill text (the long skill)."""
        if self._skill_text is None:
            try:
                from mark.middlewares.skills import load_skill_text
                self._skill_text = load_skill_text(_LONG_SKILL)
            except Exception:
                self._skill_text = ""
        return self._skill_text

    def _skill_name(self) -> str | None:
        """Which packaged skill to inject: an explicit name, the long skill, or none.

        ``include_skill=True`` keeps the long skill for large models and
        injects *no* skill for the small/tiny presets or when the long skill
        would take more than half of ``budget_tokens`` (it is charged to the
        budget, see ``_middleware_context``). Measured on llama3.2:1b (a10
        context report): the compact skill cost 8 of 25 answers even with the
        full memory budget beside it, so small models get a self-explanatory
        briefing instead; pass ``include_skill="mark-memory-compact"`` to opt in.
        """
        choice = self._include_skill
        if not choice:
            return None
        if isinstance(choice, str):
            return choice
        if getattr(self, "_preset", None) in ("small", "tiny"):
            return None
        budget = getattr(self, "_budget", None)
        if budget is not None and budget.count(self._packaged_skill()) > budget.max_tokens // 2:
            return None
        return _LONG_SKILL

    def _skill_block(self) -> str:
        """The skill text exactly as injected ("" when none)."""
        name = self._skill_name()
        if name is None:
            return ""
        if name == _LONG_SKILL:
            return self._packaged_skill()       # unchanged for large models
        cache = self.__dict__.setdefault("_skill_cache", {})
        if name not in cache:
            from mark.middlewares.skills.loader import load_skill_text, prompt_text
            # Without frontmatter: small models read "name: ..." as a tool to call (M7).
            cache[name] = prompt_text(load_skill_text(name))
        return cache[name]
