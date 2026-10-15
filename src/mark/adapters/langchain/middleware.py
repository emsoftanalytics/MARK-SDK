"""LangChain middleware for MARK — real-time context window management.

MARK operates as a transparent context window manager inside the agent lifecycle.
It watches the message history that LangChain passes toward the LLM, retrieves
only relevant durable memory before the model call, and archives useful evidence
that would otherwise fall out of the active context window.

Two surfaces are provided:

1. :class:`MarkAgentMiddleware` — a full LangChain ``AgentMiddleware`` that
   intercepts six lifecycle points:

   **RETRIEVE (inject context — synchronous with the agent):**

   * ``abefore_agent``   — captures the user query the instant it enters the
     agent; fires retrieval immediately so context is ready before the first
     model call.  Like a real-time search that pre-loads results as the query
     arrives.
   * ``abefore_model``   — per-step multi-query parallel retrieval: user goal
     (cached) + current AI reasoning excerpt (always fresh).  Context adapts
     as the agent's plan evolves, step by step.
   * ``(a)wrap_model_call`` — injects the retrieved ``[MARK project memory]``
     block into the system message of *every* model request.  Pure read +
     transform; no I/O, no latency.

   **OBSERVE (archive context — background, non-blocking):**

   * ``aafter_model``    — queues each substantive AI reasoning step so older
     turns remain retrievable even after they scroll out of the active
     context window.
   * ``(a)wrap_tool_call`` — absorbs every tool result that enters the agent's
     context window into MARK.  What the agent sees, MARK archives.
   * ``aafter_agent``    — flushes queued observations, emits a
     retrieval-quality feedback signal and releases the per-run state.

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
"""

from __future__ import annotations

import asyncio
import contextvars
import hashlib
import logging
import threading
import uuid
from collections import OrderedDict, deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from time import perf_counter
from typing import Annotated, Any, Callable

from mark.adapters.backend import BackendResult, MarkBackend, call_backend_off_loop, run_off_loop

try:
    from typing_extensions import NotRequired
except ImportError:
    from typing import Optional as NotRequired  # type: ignore[assignment]

try:
    from langchain.agents.middleware import AgentMiddleware as _AgentMiddlewareBase, AgentState
    from langchain.agents.middleware.types import PrivateStateAttr
    from langchain.messages import SystemMessage as _SystemMessage
    _LANGCHAIN_AVAILABLE = True
except ImportError:  # pragma: no cover
    _AgentMiddlewareBase = object  # type: ignore[assignment,misc]
    AgentState = dict  # type: ignore[assignment,misc]
    PrivateStateAttr = None  # type: ignore[assignment]
    _SystemMessage = None  # type: ignore[assignment]
    _LANGCHAIN_AVAILABLE = False

try:
    from langgraph.errors import GraphBubbleUp as _GraphBubbleUp
except ImportError:  # pragma: no cover
    class _GraphBubbleUp(Exception):  # type: ignore[no-redef]
        """Placeholder when LangGraph is unavailable."""


logger = logging.getLogger("mark.adapters.langchain")

#: Fallback run key for direct hook calls made outside a LangGraph run
#: (no ``mark_run_id`` in state).
_DEFAULT_RUN_ID = "__default__"
#: Upper bound on concurrently tracked runs (LRU; protects against runs that
#: never reach ``after_agent``).
_MAX_RUNS = 256
#: Upper bound on cached recall queries per run.
_MAX_RECALL_CACHE = 64
#: Upper bound on fragment ids remembered per run for feedback.
_MAX_FRAGMENT_IDS = 256
#: Upper bound on observation hashes remembered for de-duplication.
_MAX_SEEN_OBSERVATIONS = 4096
#: Upper bound on retained diagnostics events.
_MAX_DIAGNOSTICS = 1000
#: How long ``aafter_agent`` waits for the run's background tasks.
_BG_DRAIN_TIMEOUT_S = 5.0

#: True while a coroutine runs inside :func:`_run_sync` (sync hook path).
#: Background work is then executed inline, because the temporary event loop
#: is closed — cancelling any pending tasks — as soon as the hook returns.
_IN_SYNC_BRIDGE: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "mark_in_sync_bridge", default=False
)


# ---------------------------------------------------------------------------
# Custom agent state — carries MARK context between abefore_model and
# wrap_model_call without any blocking I/O in the wrap layer.
# ---------------------------------------------------------------------------

class _MarkAgentState(AgentState):  # type: ignore[misc,valid-type]
    """Extends AgentState with MARK's retrieved context and run identity."""
    mark_context: NotRequired[str]
    mark_context_meta: NotRequired[dict[str, Any]]
    mark_run_id: NotRequired[Annotated[str, PrivateStateAttr]]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_SYNC_EXECUTOR: ThreadPoolExecutor | None = None
_SYNC_EXECUTOR_LOCK = threading.Lock()


async def _bridged(coro: Any) -> Any:
    _IN_SYNC_BRIDGE.set(True)
    return await coro


def _run_sync(coro: Any) -> Any:
    """Run an async coroutine from synchronous code.

    Outside an event loop this is plain ``asyncio.run()``. When a loop is
    already running in the current thread — Jupyter kernels, GUI frameworks,
    or servers driving the sync agent API — the coroutine executes on a
    dedicated worker thread with its own loop instead: the caller blocks for
    the result and the outer loop is never touched. MARK's middleware
    coroutines only talk to the (thread-safe) local backend, so they do not
    depend on the caller's loop.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(_bridged(coro))
    global _SYNC_EXECUTOR
    if _SYNC_EXECUTOR is None:
        with _SYNC_EXECUTOR_LOCK:
            if _SYNC_EXECUTOR is None:
                _SYNC_EXECUTOR = ThreadPoolExecutor(
                    max_workers=1, thread_name_prefix="mark-sync-bridge"
                )
    return _SYNC_EXECUTOR.submit(asyncio.run, _bridged(coro)).result()


def content_text(content: Any) -> str:
    """Return the plain text of a LangChain message ``content`` value.

    Handles string content and list content made of text blocks (``{"type":
    "text", "text": ...}``), objects with a ``text`` attribute, or bare
    strings.  Non-text blocks (images, tool-use payloads) are skipped instead
    of being rendered as a Python ``repr``.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, (list, tuple)):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                text: Any = block
            elif isinstance(block, dict):
                if block.get("type") not in (None, "text", "text_delta", "output_text"):
                    continue
                text = block.get("text") or ""
            else:
                text = getattr(block, "text", None) or ""
            if text:
                parts.append(str(text))
        return "\n".join(parts)
    return str(content)


def _message_role(msg: Any) -> str:
    if isinstance(msg, dict):
        return str(msg.get("type") or msg.get("role") or "")
    return str(getattr(msg, "type", None) or getattr(msg, "role", "") or "")


def _message_content(msg: Any) -> Any:
    if isinstance(msg, dict):
        return msg.get("content", "")
    return getattr(msg, "content", "")


def _state_get(state: Any, key: str, default: Any = None) -> Any:
    if isinstance(state, dict):
        return state.get(key, default)
    return getattr(state, key, default)


def _state_messages(state: Any) -> list[Any]:
    messages = _state_get(state, "messages", None)
    return list(messages or [])


@dataclass
class _RunState:
    """Everything MARK tracks for one agent run."""

    recall_cache: OrderedDict[str, str] = field(default_factory=OrderedDict)
    pending_observations: list[str] = field(default_factory=list)
    fragment_ids: OrderedDict[str, None] = field(default_factory=OrderedDict)
    last_injected_key: str = ""
    last_context: str = ""
    last_tool_trigger_hash: str = ""
    thoughts: str | None = None
    tasks: set[asyncio.Task[Any]] = field(default_factory=set)
    prewarm: asyncio.Task[Any] | None = None


# ---------------------------------------------------------------------------
# 1. MarkAgentMiddleware — full lifecycle context window manager
# ---------------------------------------------------------------------------

class MarkAgentMiddleware(_AgentMiddlewareBase):  # type: ignore[misc]
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
        include_skill: bool = True,
        thinking_model: "Any | None" = None,
        thinking_budget_tokens: int = 2000,
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
        self._skill_text: str | None = None  # lazy-loaded packaged skill
        self._thinking_budget = thinking_budget_tokens

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

    # ── diagnostics / error handling ──────────────────────────────────────

    @property
    def diagnostics(self) -> list[dict[str, Any]]:
        """Return lightweight middleware telemetry for tracing and benchmarks.

        Bounded to the most recent ``1000`` events.
        """
        return list(self._diagnostics)

    def _record(self, kind: str, **data: Any) -> None:
        self._diagnostics.append({"kind": kind, **data})

    def _hook_failed(self, hook: str, exc: BaseException) -> None:
        """Record and log a MARK failure; the agent continues without MARK."""
        self._record("hook_error", hook=hook, error=repr(exc)[:200])
        logger.warning(
            "MARK %s failed; continuing without MARK memory: %r",
            hook, exc, exc_info=(type(exc), exc, exc.__traceback__),
        )

    # ── per-run state ─────────────────────────────────────────────────────

    def _run_id(self, state: Any) -> str:
        return str(_state_get(state, "mark_run_id", None) or _DEFAULT_RUN_ID)

    def _run(self, state: Any) -> _RunState:
        """Return (creating if needed) the per-run record for *state*."""
        return self._run_by_id(self._run_id(state))

    def _run_by_id(self, run_id: str) -> _RunState:
        with self._runs_lock:
            run = self._runs.get(run_id)
            if run is None:
                run = _RunState()
                self._runs[run_id] = run
                while len(self._runs) > _MAX_RUNS:
                    evicted, _ = self._runs.popitem(last=False)
                    self._record("run_evicted", run_id=evicted)
            else:
                self._runs.move_to_end(run_id)
            return run

    def _pop_run(self, run_id: str) -> _RunState | None:
        with self._runs_lock:
            return self._runs.pop(run_id, None)

    def _on_run_error(self, state: Any, exc: BaseException) -> None:
        """Release per-run state when a model/tool call raises.

        LangGraph interrupts (human-in-the-loop) are control flow, not errors:
        the run resumes later, so its state is kept.
        """
        if isinstance(exc, _GraphBubbleUp):
            return
        run_id = self._run_id(state)
        if self._pop_run(run_id) is not None:
            self._record("run_aborted", run_id=run_id, error=repr(exc)[:200])

    @property
    def active_runs(self) -> int:
        """Number of runs currently holding per-run state."""
        with self._runs_lock:
            return len(self._runs)

    # ── background tasks ──────────────────────────────────────────────────

    async def _spawn(self, run: _RunState | None, coro: Any) -> asyncio.Task[Any] | None:
        """Run *coro* in the background, tracked and with errors logged.

        On the sync hook path (inside :func:`_run_sync`) the temporary event
        loop is closed when the hook returns, which would cancel pending
        tasks — the coroutine is therefore awaited inline instead.
        """
        if _IN_SYNC_BRIDGE.get():
            try:
                await coro
            except Exception as exc:
                self._hook_failed("background", exc)
            return None
        task = asyncio.get_running_loop().create_task(coro)
        self._bg_tasks.add(task)
        if run is not None:
            run.tasks.add(task)
        task.add_done_callback(lambda t: self._task_done(run, t))
        return task

    def _task_done(self, run: _RunState | None, task: asyncio.Task[Any]) -> None:
        self._bg_tasks.discard(task)
        if run is not None:
            run.tasks.discard(task)
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            self._hook_failed("background", exc)

    async def _drain(self, run: _RunState) -> None:
        """Wait (bounded) for the run's background tasks on this loop."""
        loop = asyncio.get_running_loop()
        pending = [t for t in run.tasks if not t.done() and t.get_loop() is loop]
        if not pending:
            return
        _, still_pending = await asyncio.wait(pending, timeout=_BG_DRAIN_TIMEOUT_S)
        if still_pending:
            self._record("bg_drain_timeout", pending=len(still_pending))

    # ── backend access ────────────────────────────────────────────────────

    async def _backend_call(self, endpoint: str, *args: Any, **kwargs: Any) -> BackendResult:
        """Call a backend endpoint off-loop; never raises (except cancellation)."""
        started = perf_counter()
        try:
            result = await call_backend_off_loop(self._backend, endpoint, *args, **kwargs)
        except Exception as exc:
            self._record(
                "endpoint",
                endpoint=endpoint,
                ok=False,
                latency_ms=round((perf_counter() - started) * 1000, 1),
                error=str(exc)[:200],
            )
            logger.warning("MARK backend %s failed: %r", endpoint, exc)
            return BackendResult(False, error=str(exc))
        self._record(
            "endpoint",
            endpoint=endpoint,
            ok=bool(getattr(result, "ok", False)),
            latency_ms=round((perf_counter() - started) * 1000, 1),
            error=getattr(result, "error", None),
        )
        return result

    def _hash(self, text: str) -> str:
        return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()[:12]

    def _message_text(self, msg: Any) -> str:
        return content_text(_message_content(msg))

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

    def _tool_name(self, msg: Any) -> str:
        return str(
            getattr(msg, "name", "")
            or getattr(msg, "tool_name", "")
            or (getattr(msg, "additional_kwargs", None) or {}).get("name", "")
        )

    def _latest_tool_trigger(self, state: Any) -> tuple[str, str]:
        """Return a retrieval query from the latest useful tool/subagent result.

        Tool output is valuable context, but write/delete results often reflect
        unfinished intermediate work. Those are queued for observation at session
        end instead of being made immediately retrievable during the same run.
        Ordinary filesystem reads are also already present in the live context,
        so their right MARK action is observe-later, not retrieve-now.
        """
        for msg in reversed(_state_messages(state)):
            role = _message_role(msg)
            if role in ("human", "user"):
                # Tool results from earlier turns (checkpointed history) are
                # not triggers for the current request.
                return "", ""
            if role not in ("tool", "function"):
                continue
            name = self._tool_name(msg)
            lowered = name.lower()
            if any(skip in lowered for skip in ("write", "delete", "remove", "mark_")):
                return "", ""
            if not any(
                trigger in lowered
                for trigger in ("subagent", "search", "grep", "retrieve", "lookup", "web")
            ):
                return "", ""
            content = self._message_text(msg).strip()
            if len(content) < 40:
                return "", ""
            query = f"Context from {name or 'tool'} result:\n{content[:700]}"
            return query, self._hash(f"{name}:{content[:700]}")
        return "", ""

    def _truncate_context(self, text: str) -> str:
        if len(text) <= self._max_chars:
            return text
        keep = max(self._max_chars - 80, 0)
        return text[:keep].rstrip() + "\n\n[MARK context truncated]"

    def _queue_observation(self, run: _RunState, content: str) -> None:
        if not content:
            return
        normalized = " ".join(content.split())
        if len(normalized) < 40:
            return
        key = self._hash(normalized)
        with self._seen_lock:
            if key in self._seen_observation_hashes:
                self._seen_observation_hashes.move_to_end(key)
                return
            self._seen_observation_hashes[key] = None
            while len(self._seen_observation_hashes) > _MAX_SEEN_OBSERVATIONS:
                self._seen_observation_hashes.popitem(last=False)
        run.pending_observations.append(normalized[:1200])

    async def _flush_observations_async(self, run: _RunState) -> int:
        pending = run.pending_observations
        run.pending_observations = []
        count = 0
        for content in pending:
            result = await self._backend_call("observe", content, agent_id=self._agent_id)
            if result.ok:
                count += 1
        if count:
            self._record("observe_flush", count=count)
        return count

    # ── query extraction helpers ──────────────────────────────────────────

    def _build_queries(self, state: Any) -> tuple[str, str]:
        """Extract retrieval queries from agent state.

        Returns ``(human_query, ai_query)`` where:

        * ``human_query`` — the user's original goal (stable across all model
          calls in one run; cached so only one backend round-trip per run).
        * ``ai_query``    — the latest AI reasoning excerpt (first 300 chars of
          the most recent non-tool-call AI message).
        """
        human_query = ""
        ai_query = ""
        for msg in reversed(_state_messages(state)):
            role = _message_role(msg)
            if not human_query and role in ("human", "user"):
                human_query = self._message_text(msg).strip()
            if not ai_query and role in ("ai", "assistant"):
                if getattr(msg, "tool_calls", None):
                    continue
                content = self._message_text(msg).strip()
                if len(content) > 20:
                    ai_query = content[:300]
            if human_query and ai_query:
                break
        return human_query, ai_query

    def _get_human_query_from_messages(self, messages: Any) -> str:
        """Extract last human message text from a ModelRequest message list."""
        for msg in reversed(messages or []):
            if _message_role(msg) in ("human", "user"):
                return self._message_text(msg).strip()
        return ""

    # ── core retrieval ────────────────────────────────────────────────────

    @staticmethod
    def _render(value: Any) -> tuple[str, list[str]]:
        """Render a retrieval value as ``(context_text, fragment_ids)``.

        Uses the structured result: a result whose ``fragments`` list is empty
        has no memory, so ``""`` is returned (nothing gets injected).
        """
        fragments = getattr(value, "fragments", None)
        if fragments is not None and not fragments:
            return "", []
        text = (value.as_context() if hasattr(value, "as_context")
                else value.as_text() if hasattr(value, "as_text")
                else str(value))
        ids = [str(getattr(f, "id", "") or "") for f in (fragments or [])]
        return (text or "").strip(), [i for i in ids if i]

    @staticmethod
    def _needs_heal(value: Any, has_memory: bool) -> bool:
        if not has_memory:
            return True
        from mark.types.gap import GapSeverity

        severity = getattr(getattr(value, "gap_report", None), "severity", None)
        try:
            return GapSeverity(severity) in (GapSeverity.HIGH, GapSeverity.CRITICAL)
        except ValueError:
            return False

    async def _timed_backend(self, endpoint: str, query: str) -> BackendResult | None:
        """Call a retrieval endpoint under ``retrieval_timeout_ms``.

        Returns ``None`` on timeout.
        """
        call = self._backend_call(endpoint, query, agent_id=self._agent_id, compress=self._compress)
        if not self._retrieval_timeout_s:
            return await call
        try:
            return await asyncio.wait_for(call, timeout=self._retrieval_timeout_s)
        except (TimeoutError, asyncio.TimeoutError):
            self._record(
                "endpoint",
                endpoint=endpoint,
                ok=False,
                latency_ms=round(self._retrieval_timeout_s * 1000, 1),
                error="timeout",
            )
            return None

    async def _recall(self, query: str, run: _RunState, *, cache: bool = True) -> str:
        """Retrieve relevant MARK memory for *query* (bounded, ``""`` if none).

        Args:
            query: Retrieval query string.
            run:   Per-run record owning the cache.
            cache: When ``True`` (default), result is read from / written to the
                   run's recall cache.  Pass ``cache=False`` for step-specific
                   queries (e.g., AI reasoning excerpt) that change each model call.
        """
        if not query:
            return ""
        if cache and query in run.recall_cache:
            run.recall_cache.move_to_end(query)
            return run.recall_cache[query]

        text, ids = "", []
        result = await self._timed_backend("retrieve", query)
        if result is not None and result.ok:
            text, ids = self._render(result.value)
            if self._auto_heal and self._needs_heal(result.value, bool(text)):
                healed = await self._timed_backend("heal_gap", query)
                if healed is not None and healed.ok:
                    healed_text, healed_ids = self._render(healed.value)
                    if healed_text:
                        text, ids = healed_text, healed_ids
                        self._record("heal", chars=len(healed_text))

        text = self._truncate_context(text) if text else ""
        for fragment_id in ids:
            run.fragment_ids[fragment_id] = None
        while len(run.fragment_ids) > _MAX_FRAGMENT_IDS:
            run.fragment_ids.popitem(last=False)
        if cache and result is not None:
            run.recall_cache[query] = text
            while len(run.recall_cache) > _MAX_RECALL_CACHE:
                run.recall_cache.popitem(last=False)
        return text

    # ── thinking engine ───────────────────────────────────────────────────

    def _should_think(self, query: str) -> bool:
        """Return True when *query* contains planning/reasoning trigger keywords."""
        lowered = query.lower()
        return any(t in lowered for t in self.THINKING_TRIGGERS)

    async def _think(self, query: str, context: str) -> str:
        """Call the thinking model to generate a structured reasoning plan.

        On the sync hook path the model's synchronous ``invoke`` is used (in a
        worker thread), so providers that cache event-loop-bound async clients
        are never driven from a throwaway loop.  On the async path ``ainvoke``
        is awaited.  Failures are logged and recorded; ``""`` is returned.
        """
        model = self._thinking_model
        if model is None:
            return ""

        thinking_prompt = (
            "You are a focused reasoning assistant. Given the user's request and "
            "relevant project context, think through the best approach step by step.\n\n"
            f"User request: {query}\n\n"
            f"Project context:\n{context}\n\n"
            "Think through:\n"
            "## SESSION INTENT\nWhat exactly needs to be done?\n\n"
            "## APPROACH\nWhat is the best approach given the project context?\n\n"
            "## STEPS\nWhat concrete steps should be taken? List them.\n\n"
            "## RISKS\nWhat could go wrong? What should be avoided?\n\n"
            "Be concise and actionable. Focus on what matters for this specific request."
        )
        try:
            if (_IN_SYNC_BRIDGE.get() or not hasattr(model, "ainvoke")) and hasattr(model, "invoke"):
                response = await run_off_loop(model.invoke, thinking_prompt)
            else:
                response = await model.ainvoke(thinking_prompt)
            thought = content_text(getattr(response, "content", response)).strip()
            return thought[:self._thinking_budget * 4]  # approx chars to tokens ratio
        except Exception as exc:
            self._record("thinking", ok=False, error=repr(exc)[:200])
            logger.warning("MARK thinking_model call failed; continuing without it: %r", exc)
            return ""

    async def _retrieve_and_combine(
        self,
        run: _RunState,
        human_query: str,
        ai_query: str = "",
        tool_query: str = "",
    ) -> str:
        """Run triggered retrieval queries and return a bounded combined block.

        Human query is cached (stable). AI/tool queries are only used when they
        represent a meaningful trigger point, such as a planning response or a
        newly observed read/search/subagent result.

        Returns empty string when no relevant memory is found.
        """
        tasks: list[Any] = [self._recall(human_query, run, cache=True)]
        if ai_query:
            tasks.append(self._recall(ai_query, run, cache=False))
        if tool_query:
            tasks.append(self._recall(tool_query, run, cache=False))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        blocks: list[str] = []
        for r in results:
            if isinstance(r, BaseException):
                if not isinstance(r, Exception):
                    raise r
                self._hook_failed("retrieve", r)
                continue
            if r and r not in blocks:
                blocks.append(r)

        if not blocks:
            return ""

        combined = self._truncate_context("\n\n---\n\n".join(blocks))
        self._record(
            "retrieve",
            queries=1 + bool(ai_query) + bool(tool_query),
            chars=len(combined),
            context_hash=self._hash(combined),
        )
        return combined

    # ── abefore_agent: run entry intercept ────────────────────────────────

    async def abefore_agent(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        """Async: start a run — assign its id, capture the query, warm retrieval.

        Returns ``{"mark_run_id": ..., "mark_context": ""}`` so per-run data is
        isolated and context checkpointed by a previous turn is never reused.
        """
        try:
            run_id = uuid.uuid4().hex
            run = self._run_by_id(run_id)
            human_query, _ = self._build_queries(state)
            if human_query:
                run.prewarm = await self._spawn(run, self._recall(human_query, run, cache=True))
            return {"mark_run_id": run_id, "mark_context": "", "mark_context_meta": {}}
        except Exception as exc:
            self._hook_failed("before_agent", exc)
            return None

    def before_agent(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        """Sync: start a run (see :meth:`abefore_agent`)."""
        try:
            return _run_sync(self.abefore_agent(state, runtime))
        except Exception as exc:
            self._hook_failed("before_agent", exc)
            return None

    # ── abefore_model: per-step multi-query retrieval ─────────────────────

    async def abefore_model(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        """Async: triggered retrieval before every model call.

        Always returns a ``mark_context`` update — the combined retrieved
        memory, or ``""`` when nothing relevant was found — so stale context
        from earlier turns or steps never leaks into this model call.
        """
        try:
            run = self._run(state)
            human_query, ai_query = self._build_queries(state)
            if not human_query:
                return {"mark_context": "", "mark_context_meta": {"chars": 0, "source": "none"}}

            prewarm = run.prewarm
            if prewarm is not None:
                run.prewarm = None
                if not prewarm.done() and prewarm.get_loop() is asyncio.get_running_loop():
                    await asyncio.wait({prewarm}, timeout=self._retrieval_timeout_s or None)

            tool_query, tool_hash = self._latest_tool_trigger(state)

            # Skip re-retrieval when neither the user goal nor the triggers have
            # changed since the last model call of this run.
            ai_trigger = ai_query if self._should_think(ai_query) else ""
            inject_key = f"{human_query}||{ai_trigger}||{tool_hash}"
            if inject_key == run.last_injected_key:
                cached = run.last_context
                return {
                    "mark_context": cached,
                    "mark_context_meta": {
                        "chars": len(cached),
                        "context_hash": self._hash(cached),
                        "source": "cache",
                    },
                }

            refresh_tool = bool(tool_hash) and tool_hash != run.last_tool_trigger_hash
            combined = await self._retrieve_and_combine(
                run,
                human_query,
                ai_trigger,
                tool_query if refresh_tool else "",
            )
            if refresh_tool:
                run.last_tool_trigger_hash = tool_hash

            # Thinking mode — secondary LLM when the query has planning triggers.
            # Computed once per run, injected alongside MARK memory and archived.
            if self._thinking_model and self._should_think(human_query):
                if run.thoughts is None:
                    run.thoughts = await self._think(human_query, combined or "")
                    if run.thoughts:
                        await self._spawn(run, self._backend_call(
                            "observe",
                            f"MARK reasoning plan for query '{human_query[:100]}':\n"
                            f"{run.thoughts[:400]}",
                            agent_id=self._agent_id,
                        ))
                if run.thoughts:
                    thought_block = f"[MARK reasoning]\n{run.thoughts}\n[/MARK reasoning]"
                    combined = (combined + "\n\n---\n\n" + thought_block) if combined else thought_block

            run.last_injected_key = inject_key
            run.last_context = combined
            return {
                "mark_context": combined,
                "mark_context_meta": {
                    "chars": len(combined),
                    "context_hash": self._hash(combined),
                    "source": "retrieve",
                    "tool_trigger": refresh_tool,
                },
            }
        except Exception as exc:
            self._hook_failed("before_model", exc)
            return {"mark_context": "", "mark_context_meta": {"chars": 0, "source": "error"}}

    def before_model(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        """Sync: triggered MARK retrieval before every model call."""
        try:
            return _run_sync(self.abefore_model(state, runtime))
        except Exception as exc:
            self._hook_failed("before_model", exc)
            return {"mark_context": "", "mark_context_meta": {"chars": 0, "source": "error"}}

    # ── (a)wrap_model_call: inject MARK context into system message ───────

    def _memory_block(self, context: str) -> str:
        return (
            "[MARK project memory]\n"
            "Use this as durable project memory. Prefer live tool results for "
            "current-session file contents.\n"
            f"{context}\n"
            "[/MARK project memory]"
        )

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
            context = self._context_for_request(request)
            system_message = getattr(request, "system_message", None)
            system_text = self._system_text(system_message)
            additions: list[dict[str, str]] = []

            # Auto-inject the packaged middleware skill once, so agents know how
            # to treat MARK-injected memory without manual prompt wiring.
            skill_block = self._packaged_skill() if self._include_skill else ""
            if skill_block and "mark-memory-middleware" not in system_text:
                additions.append({"type": "text", "text": skill_block})

            if context:
                skill_hash = self._hash(system_text)
                if skill_hash != self._last_skill_hash:
                    skill = self._score_skill(system_text + skill_block)
                    self._record("skill", skill_hash=skill_hash, **skill)
                    self._last_skill_hash = skill_hash
                context = self._truncate_context(context)
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

        Appends a ``[MARK project memory]`` block to ``request.system_message``
        via ``request.override()``.  Pure read + transform — no backend calls.
        """
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
        """Lazy-load the packaged mark-memory-middleware skill text."""
        if self._skill_text is None:
            try:
                from mark.middlewares.skills import load_skill_text
                self._skill_text = load_skill_text("mark-memory-middleware")
            except Exception:
                self._skill_text = ""
        return self._skill_text

    # ── (a)wrap_tool_call: absorb tool results into MARK ──────────────────

    def _observe_tool_result(self, request: Any, result: Any) -> Any:
        """Queue (or schedule) observation of a tool result.

        Returns a coroutine for immediate observation, else ``None``.
        """
        if not (self._write_outcomes and self._observe_tool_results):
            return None
        tool_call = getattr(request, "tool_call", None) or {}
        tool_name = str(tool_call.get("name", "unknown_tool"))
        lowered = tool_name.lower()
        content = content_text(getattr(result, "content", "") or "")[:900]
        if not content or len(content) < 20:
            return None
        observation = f"Tool [{tool_name}] returned: {content}"
        run = self._run(getattr(request, "state", None))
        if self._immediate_tool_observe and not any(
            skip in lowered for skip in ("write", "delete", "remove")
        ):
            self._record("observe_immediate", tool=tool_name, chars=len(content))
            return self._backend_call("observe", observation, agent_id=self._agent_id)
        self._queue_observation(run, observation)
        return None

    def wrap_tool_call(self, request: Any, handler: Callable[[Any], Any]) -> Any:
        """Sync: execute the tool and queue its result for run-end archival."""
        try:
            result = handler(request)
        except BaseException as exc:
            self._on_run_error(getattr(request, "state", None), exc)
            raise
        try:
            pending = self._observe_tool_result(request, result)
            if pending is not None:
                _run_sync(pending)
        except Exception as exc:
            self._hook_failed("wrap_tool_call", exc)
        return result

    async def awrap_tool_call(self, request: Any, handler: Callable[[Any], Any]) -> Any:
        """Async: execute the tool and archive its result into MARK.

        The result is returned unchanged — MARK is completely transparent.
        With ``immediate_tool_observe=True`` the observation runs as a tracked
        background task; otherwise it is queued until the run ends.
        """
        try:
            result = await handler(request)
        except BaseException as exc:
            self._on_run_error(getattr(request, "state", None), exc)
            raise
        try:
            pending = self._observe_tool_result(request, result)
            if pending is not None:
                await self._spawn(self._run(getattr(request, "state", None)), pending)
        except Exception as exc:
            self._hook_failed("wrap_tool_call", exc)
        return result

    # ── aafter_model: queue AI reasoning step ─────────────────────────────

    def _queue_latest_reasoning(self, state: Any) -> None:
        for msg in reversed(_state_messages(state)):
            if _message_role(msg) in ("ai", "assistant"):
                content = self._message_text(msg).strip()
                has_tool_calls = bool(getattr(msg, "tool_calls", None))
                if content and len(content) > 100 and not has_tool_calls:
                    self._queue_observation(self._run(state), f"Agent reasoning: {content[:600]}")
                break

    async def aafter_model(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        """Async: queue each substantive AI reasoning step for archival.

        Only substantive turns (>100 chars, no tool-dispatch payloads) are
        queued, to avoid noise.  Returns ``None`` (no state update needed).
        """
        if not self._write_outcomes:
            return None
        try:
            self._queue_latest_reasoning(state)
        except Exception as exc:
            self._hook_failed("after_model", exc)
        return None

    def after_model(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        """Sync: queue substantive AI reasoning for run-end archival."""
        if not self._write_outcomes:
            return None
        try:
            self._queue_latest_reasoning(state)
        except Exception as exc:
            self._hook_failed("after_model", exc)
        return None

    # ── aafter_agent: run end — flush, feedback, cleanup ──────────────────

    async def aafter_agent(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        """Async: flush observations, emit feedback, release the run.

        Fires once when the agent graph reaches its terminal node.  Queued
        observations are written, a ``feedback()`` signal is sent for the
        fragments retrieved during the run, pending background tasks are
        drained (bounded wait), and ``mark_context`` is cleared so the next
        turn on the same thread never sees stale memory.
        """
        run_id = self._run_id(state)
        run = self._pop_run(run_id) or _RunState()
        try:
            if self._write_outcomes:
                for msg in reversed(_state_messages(state)):
                    if _message_role(msg) in ("ai", "assistant"):
                        content = self._message_text(msg)[:200]
                        self._queue_observation(run, f"Agent completed task. Response: {content}")
                        await self._spawn(run, self._backend_call(
                            "feedback",
                            agent_id=self._agent_id,
                            outcome="completed",
                            summary=content,
                            fragment_ids=list(run.fragment_ids),
                        ))
                        break
                await self._flush_observations_async(run)
        except Exception as exc:
            self._hook_failed("after_agent", exc)
        finally:
            try:
                await self._drain(run)
            except Exception as exc:
                self._hook_failed("after_agent", exc)
        return {"mark_context": "", "mark_context_meta": {}}

    def after_agent(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        """Sync: flush queued observations and release the run."""
        try:
            return _run_sync(self.aafter_agent(state, runtime))
        except Exception as exc:
            self._hook_failed("after_agent", exc)
            self._pop_run(self._run_id(state))
            return {"mark_context": "", "mark_context_meta": {}}


# ---------------------------------------------------------------------------
# 2. MarkMemoryMiddleware — standalone dataclass helper
# ---------------------------------------------------------------------------

@dataclass
class MarkMemoryMiddleware:
    """
    Lightweight MARK memory helper for message-list augmentation.

    Use when you are *not* using a full LangChain agent graph — e.g. when
    calling a chat model directly.  For agent-graph integration use
    :class:`MarkAgentMiddleware` instead.

    Example::

        mw = MarkMemoryMiddleware(backend=backend, agent_id="__mark__")
        messages = [{"role": "user", "content": "Add a health check."}]
        augmented = mw.augment_messages(messages, "health check")
        response  = llm.invoke(augmented)
        mw.remember_sync(f"Response: {response.content}")
    """

    backend: MarkBackend
    agent_id: str
    session_id: str | None = None
    max_context_chars: int = 6000
    write_outcomes: bool = False

    # ── async ────────────────────────────────────────────────────────────

    async def recall(self, query: str) -> str:
        """Retrieve and format memory context for the query (``""`` if none)."""
        result = await call_backend_off_loop(
            self.backend, "retrieve", query,
            agent_id=self.agent_id, session_id=self.session_id,
        )
        if not result.ok:
            return ""
        text, _ = MarkAgentMiddleware._render(result.value)
        return text[: self.max_context_chars]

    async def remember(self, content: str, **meta: Any) -> None:
        """Store content into MARK memory."""
        if not self.write_outcomes:
            return
        await call_backend_off_loop(
            self.backend, "observe", content, agent_id=self.agent_id, metadata=meta
        )

    # ── sync ─────────────────────────────────────────────────────────────

    def recall_sync(self, query: str) -> str:
        """Synchronous variant of recall()."""
        return _run_sync(self.recall(query))

    def remember_sync(self, content: str, **meta: Any) -> None:
        """Synchronous variant of remember()."""
        _run_sync(self.remember(content, **meta))

    # ── message augmentation ─────────────────────────────────────────────

    def augment_messages(
        self,
        messages: list[Any],
        query: str,
        *,
        inject_as: str = "system",
    ) -> list[Any]:
        """Return a new message list with MARK context prepended.

        Retrieves memory for *query* and injects it as a system message
        (default) or into the first human message (``inject_as="human"``).
        If memory is empty the original list is returned unchanged.
        """
        context = self.recall_sync(query)
        if not context:
            return list(messages)

        block = f"[MARK project memory]\n{context}\n[/MARK project memory]"
        out: list[Any] = []
        injected = False

        for msg in messages:
            role = (msg.get("role") if isinstance(msg, dict)
                    else getattr(msg, "type", getattr(msg, "role", None)))
            if role in ("system", "SystemMessage") and not injected:
                existing = (msg.get("content") if isinstance(msg, dict)
                            else getattr(msg, "content", "")) or ""
                new_content = f"{existing}\n\n{block}".lstrip()
                out.append({**msg, "content": new_content} if isinstance(msg, dict)
                           else type(msg)(content=new_content))
                injected = True
            else:
                out.append(msg)

        if not injected:
            if inject_as == "system":
                out.insert(0, {"role": "system", "content": block})
            else:
                for i, msg in enumerate(out):
                    role = (msg.get("role") if isinstance(msg, dict)
                            else getattr(msg, "type", getattr(msg, "role", None)))
                    if role in ("user", "human", "HumanMessage"):
                        existing = (msg.get("content") if isinstance(msg, dict)
                                    else getattr(msg, "content", "")) or ""
                        new_content = f"{block}\n\n{existing}"
                        out[i] = ({**msg, "content": new_content} if isinstance(msg, dict)
                                  else type(msg)(content=new_content))
                        break

        return out
