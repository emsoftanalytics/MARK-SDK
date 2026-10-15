"""LangChain middleware for MARK — real-time context window management.

MARK operates as a transparent context window manager inside the agent lifecycle.
It watches the message history that LangChain passes toward the LLM, retrieves
only relevant durable memory before the model call, and archives useful evidence
that would otherwise fall out of the active context window.

Two surfaces are provided:

1. :class:`MarkAgentMiddleware` — a full LangChain ``AgentMiddleware`` that
   intercepts six lifecycle points:

   **RETRIEVE (inject context — synchronous with the agent):**

   * ``abefore_agent``   — captures the user query and warms retrieval.
   * ``abefore_model``   — per-step retrieval (user goal + reasoning/tool triggers).
   * ``(a)wrap_model_call`` — injects the ``[MARK project memory]`` block (no I/O).

   **OBSERVE (archive context — background, non-blocking):**

   * ``aafter_model`` / ``(a)wrap_tool_call`` — queue reasoning (low trust)
     and tool results for archival, with provenance.
   * ``aafter_agent``    — flushes queued observations, emits feedback and
     releases the per-run state.

2. :class:`MarkMemoryMiddleware` — a lightweight dataclass helper for
   augmenting plain message lists (without a full LangChain agent graph).
   Useful for scripts, notebooks, or framework-agnostic code.

Both back onto the same :class:`~mark.adapters.backend.MarkBackend`
protocol, so they work identically against compatible runtimes.

Robustness contract
───────────────────
* **Per-run isolation.** Everything that belongs to one agent run (recall
  cache, queued observations, retrieved fragment ids, background tasks) is kept
  in a per-run record keyed by a run id stored in graph state
  (``mark_run_id``, created in ``before_agent``).  One middleware instance can
  safely serve concurrent ``ainvoke()`` calls.  The record is released in
  ``after_agent`` and when a model/tool call raises; a bounded LRU guards
  against runs that never finish.
* **Never break the agent.** Every hook degrades gracefully: MARK failures are
  recorded in :attr:`MarkAgentMiddleware.diagnostics` and logged on the
  ``mark.adapters.langchain`` logger, and the agent continues without MARK.

Implementation layout
─────────────────────
This module defines :class:`MarkAgentMiddleware` (constructor and
configuration) and re-exports the public helpers. Behaviour is composed from
private mixins: ``_middleware_runs`` (per-run state, background tasks,
diagnostics), ``_middleware_retrieval`` (queries and recall),
``_middleware_calls`` (model/tool call wrappers), ``_middleware_hooks``
(agent/model lifecycle hooks); shared plumbing is in ``_middleware_support``
and :class:`MarkMemoryMiddleware` in ``_middleware_memory``.
"""

from __future__ import annotations

import asyncio
import threading
from collections import OrderedDict, deque
from typing import Any

from mark.adapters.backend import MarkBackend
from mark.adapters.langchain._middleware_calls import _CallWrappersMixin
from mark.adapters.langchain._middleware_context import _ContextControlMixin
from mark.adapters.langchain._middleware_hooks import _LifecycleHooksMixin

# Re-exported: this module is the public/historical import path for the
# middleware classes and helpers that now live in the private submodules.
from mark.adapters.langchain._middleware_memory import MarkMemoryMiddleware
from mark.adapters.langchain._middleware_retrieval import _RetrievalMixin
from mark.adapters.langchain._middleware_runs import _RunsMixin
from mark.adapters.langchain._middleware_watch import _AnticipationMixin
from mark.adapters.langchain._middleware_support import (
    _BG_DRAIN_TIMEOUT_S,
    _DEFAULT_RUN_ID,
    _IN_SYNC_BRIDGE,
    _LANGCHAIN_AVAILABLE,
    _MAX_DIAGNOSTICS,
    _MAX_FRAGMENT_IDS,
    _MAX_RECALL_CACHE,
    _MAX_RUNS,
    _MAX_SEEN_OBSERVATIONS,
    AgentState,
    NotRequired,
    PrivateStateAttr,
    _AgentMiddlewareBase,
    _bridged,
    _GraphBubbleUp,
    _MarkAgentState,
    _message_content,
    _message_role,
    _run_sync,
    _RunState,
    _state_get,
    _state_messages,
    _SystemMessage,
    content_text,
    logger,
)

# ---------------------------------------------------------------------------
# 1. MarkAgentMiddleware — full lifecycle context window manager
# ---------------------------------------------------------------------------

# The mixins precede the LangChain base so their hook methods override the
# base's defaults; LangChain detects implemented hooks by comparing
# ``type(mw).<hook>`` with ``AgentMiddleware.<hook>``, which follows the MRO.
class MarkAgentMiddleware(  # type: ignore[misc]
    _AnticipationMixin,
    _ContextControlMixin,
    _LifecycleHooksMixin,
    _CallWrappersMixin,
    _RetrievalMixin,
    _RunsMixin,
    _AgentMiddlewareBase,
):
    """
    LangChain ``AgentMiddleware`` that manages the agent's context window via MARK.

    MARK intercepts every lifecycle event — capturing the user query the instant
    it enters, injecting relevant archived knowledge before each model call, and
    archiving AI reasoning and tool results — so the agent's context window
    always contains what it needs, and nothing is ever lost.

    Pass it to ``create_agent(model, tools, middleware=[MarkAgentMiddleware(...)])``.
    Both ``invoke()``/``stream()`` and ``ainvoke()``/``astream()`` are supported,
    and one instance may serve concurrent runs.

    Lifecycle intercepts
    ────────────────────
    **Retrieve (inject context, synchronous with agent):**

    * ``abefore_agent``   — assigns a run id, captures the user query at entry
      and warms retrieval so context is ready before the first model call.
    * ``abefore_model``   — triggered retrieval per model call: user goal query
      (cached per run) + planning/tool triggers.  Always writes the
      ``mark_context`` state field (empty when nothing relevant was found).
    * ``(a)wrap_model_call`` — appends ``[MARK project memory]`` to the system
      message of every model request via ``request.override()``.  No I/O.

    **Observe (archive context):**

    * ``aafter_model``    — queues each substantive AI reasoning step.
    * ``(a)wrap_tool_call`` — absorbs tool results into MARK.
    * ``aafter_agent``    — flushes queued observations, emits a feedback
      signal for the retrieved fragments, clears ``mark_context``.

    Args:
        backend:         A :class:`~mark.adapters.backend.MarkBackend`.
        agent_id:        Memory namespace (default ``"__mark__"`` — the same
                         namespace ``Mark.local().memory`` writes to).
        max_context_chars: Truncate injected context at this character count (default 4000).
        write_outcomes:  Archive AI reasoning + tool results into MARK (default ``True``).
        auto_heal:       Call ``heal_gap`` when retrieval finds nothing or the gap
                         severity is HIGH/CRITICAL (default ``False``).  With the
                         local backend this runs an explicit *web lookup*, so it is
                         opt-in to keep MARK local-first.
        compress:        Run LLM contextual compression on retrieved fragments
                         (default ``False`` — vector search + reranker only; fast).

    Example::

        from mark.runtime import Mark
        from mark.adapters.backend import LocalMarkBackend
        from mark.adapters.langchain.middleware import MarkAgentMiddleware
        from langchain.agents import create_agent

        mark  = Mark.local(".")
        mw    = MarkAgentMiddleware(backend=LocalMarkBackend(mark))
        agent = create_agent(llm, tools, middleware=[mw])
        result = agent.invoke({"messages": [...]})
    """

    # Expose the custom state fields to the LangGraph agent graph.
    state_schema = _MarkAgentState

    #: Keywords that activate thinking mode when found in the user query or AI reasoning.
    THINKING_TRIGGERS: tuple[str, ...] = (
        "plan", "design", "architect", "approach", "strategy",
        "think", "reason", "analyze", "how should", "best way",
        "structure", "implement", "build", "create", "refactor",
    )

    def __init__(
        self,
        backend: MarkBackend,
        *,
        agent_id: str = "__mark__",
        max_context_chars: int = 4000,
        write_outcomes: bool = True,
        auto_heal: bool = False,
        compress: bool = False,
        observe_tool_results: bool = True,
        immediate_tool_observe: bool = False,
        retrieval_timeout_ms: int = 2500,
        include_skill: "bool | str" = True,
        thinking_model: "Any | None" = None,
        thinking_budget_tokens: int = 2000,
        anticipate: bool = True,
        budget_tokens: "int | Any | None" = None,
        token_counter: "Any | None" = None,
        history: str = "full",
        keep_last: int = 6,
        preset: "str | None" = None,
        store_assistant: bool = True,
    ) -> None:
        """
        Args:
            include_skill: Automatically append the packaged
                ``mark-memory-middleware`` agent skill to the system message
                when it is not already present, so agents know how to work
                with MARK-injected context without manual prompt wiring.
                Default ``True``.
            thinking_model: Advanced, opt-in helper model for experiments where
                MARK should compile a compact planning note before the main model
                call.  The primary LangChain model remains the agent's reasoner;
                MARK's default role is memory retrieval and observation.  Accepts
                a model identifier string (e.g., ``"ollama:qwen3:4b"``) or a
                ``BaseChatModel`` instance.  Default ``None`` - disabled.
            thinking_budget_tokens: Maximum tokens in the thinking output (default 2000).
                Only used when ``thinking_model`` is set.
            anticipate: Follow the conversation with a ContextWatch and inject
                its briefing (facts about subjects in play, corrections of
                outdated values the agent states, "not in memory" notices).
                Needs a backend with ``memory_for``; default ``True``.
            budget_tokens: Hard token cap (int or ``ContextBudget``) on all
                MARK context injected per model call — briefing, retrieved
                memories and thinking plan under ONE priority-filled cap
                (``mark.context.budget``). Replaces ``max_context_chars``.
            token_counter: ``str -> int`` for the budget (default ceil(chars/4)).
            history: ``"full"`` (default) sends the whole conversation;
                ``"mark"`` keeps system + the last ``keep_last`` messages and
                lets MARK stand in for the rest (stored before dropping).
            keep_last: Messages kept in ``history="mark"`` (default 6).
            preset: Model-size preset for the briefing: ``"small"`` (~1-3B)
                or ``"tiny"`` (<=1B) also select the compact skill
                (``mark.context.presets``); default: none.
            store_assistant: Store the model's replies when they leave the
                window (low trust, ``mark:assistant``; never anchors a
                briefing). ``False`` drops them unstored.

        ``include_skill`` may also name a packaged skill. ``True`` picks the
        compact skill for small presets or when the long one would take over
        half of ``budget_tokens``; the injected skill counts inside the budget.
        """
        if not _LANGCHAIN_AVAILABLE:
            raise ImportError(
                "LangChain is required for MarkAgentMiddleware. "
                'Install with: pip install "mark-sdk[langchain]"'
            )
        super().__init__()
        self._backend = backend
        self._agent_id = agent_id
        self._max_chars = max_context_chars
        self._write_outcomes = write_outcomes
        self._auto_heal = auto_heal
        self._compress = compress
        self._observe_tool_results = observe_tool_results
        self._immediate_tool_observe = immediate_tool_observe
        self._retrieval_timeout_s = max(retrieval_timeout_ms, 0) / 1000
        self._include_skill = include_skill
        if preset is not None:
            from mark.context.presets import preset_settings
            preset_settings(preset)          # validates the name
        self._preset = preset
        self._store_assistant = store_assistant
        self._skill_text: str | None = None  # lazy-loaded packaged skill
        self._thinking_budget = thinking_budget_tokens
        self._anticipate = anticipate
        self._init_context_controls(history, keep_last, budget_tokens, token_counter)

        # Per-run state, keyed by the ``mark_run_id`` stored in graph state.
        self._runs: OrderedDict[str, _RunState] = OrderedDict()
        self._runs_lock = threading.Lock()
        # Cross-run, bounded: de-duplicates identical observations.
        self._seen_observation_hashes: OrderedDict[str, None] = OrderedDict()
        self._seen_lock = threading.Lock()
        # Strong references to in-flight background tasks (all runs).
        self._bg_tasks: set[asyncio.Task[Any]] = set()
        self._last_skill_hash: str = ""
        self._diagnostics: deque[dict[str, Any]] = deque(maxlen=_MAX_DIAGNOSTICS)

        # Thinking model — initialised lazily from string if needed
        if thinking_model is None:
            self._thinking_model: Any = None
        elif isinstance(thinking_model, str):
            try:
                from langchain.chat_models import init_chat_model  # type: ignore[import]
                self._thinking_model = init_chat_model(thinking_model)
            except Exception as exc:
                raise ImportError(
                    f"Could not initialise thinking_model from string {thinking_model!r}: {exc}"
                ) from exc
        else:
            self._thinking_model = thinking_model
