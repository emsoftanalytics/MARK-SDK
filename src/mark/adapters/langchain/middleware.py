"""LangChain middleware for MARK — real-time context window management.

MARK operates as a transparent context window manager inside the agent lifecycle.
It intercepts every piece of information that enters the agent's context and every
piece of reasoning the agent produces — archiving losslessly, retrieving precisely.

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
   * ``wrap_model_call`` — injects the retrieved ``[MARK project memory]`` block
     into the system message.  Pure read + transform; no I/O, no latency.

   **OBSERVE (archive context — background, non-blocking):**

   * ``aafter_model``    — archives each AI reasoning step as it happens so
     older turns remain retrievable even after they scroll out of the active
     context window.  Fire-and-forget.
   * ``awrap_tool_call`` — absorbs every tool result that enters the agent's
     context window into MARK.  What the agent sees, MARK archives.  Fire-and-forget.
   * ``aafter_agent``    — emits a retrieval-quality feedback signal and clears
     the per-session cache.  Fire-and-forget.

   All observe/feedback calls run as background ``asyncio`` tasks — MARK never
   blocks the agent's execution path.

2. :class:`MarkMemoryMiddleware` — a lightweight dataclass helper for
   augmenting plain message lists (without a full LangChain agent graph).
   Useful for scripts, notebooks, or framework-agnostic code.

Both back onto the same :class:`~mark.adapters.backend.MarkBackend`
protocol, so they work identically against local and cloud runtimes.
"""

from __future__ import annotations

import asyncio
import hashlib
from time import perf_counter
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

from mark.adapters.backend import MarkBackend

try:
    from typing_extensions import NotRequired
except ImportError:
    from typing import Optional as NotRequired  # type: ignore[assignment]

try:
    from langchain.agents.middleware import AgentMiddleware as _AgentMiddlewareBase, AgentState
    from langchain.messages import SystemMessage as _SystemMessage
    _LANGCHAIN_AVAILABLE = True
except ImportError:  # pragma: no cover
    _AgentMiddlewareBase = object  # type: ignore[assignment,misc]
    AgentState = dict  # type: ignore[assignment,misc]
    _SystemMessage = None  # type: ignore[assignment]
    _LANGCHAIN_AVAILABLE = False


# ---------------------------------------------------------------------------
# Custom agent state — carries MARK context between abefore_model and
# wrap_model_call without any blocking I/O in the wrap layer.
# ---------------------------------------------------------------------------

class _MarkAgentState(AgentState):  # type: ignore[misc,valid-type]
    """Extends AgentState with a slot for MARK's retrieved context."""
    mark_context: NotRequired[str]
    mark_context_meta: NotRequired[dict[str, Any]]


# ---------------------------------------------------------------------------
# Internal helper (kept for MarkMemoryMiddleware sync surface only)
# ---------------------------------------------------------------------------

def _run_sync(coro: Any) -> Any:
    """Run an async coroutine synchronously — only safe outside a running loop."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    raise RuntimeError(
        "MARK sync helpers cannot be called inside a running event loop. "
        "Use the async methods (recall / remember) instead."
    )


# ---------------------------------------------------------------------------
# 1. MarkAgentMiddleware — full lifecycle context window manager
# ---------------------------------------------------------------------------

class MarkAgentMiddleware(_AgentMiddlewareBase):  # type: ignore[misc]
    """
    LangChain ``AgentMiddleware`` that manages the agent's context window via MARK.

    MARK intercepts every lifecycle event — capturing the user query the instant
    it enters, injecting relevant archived knowledge before each model call, and
    archiving all AI reasoning and tool results in the background — so the agent's
    context window always contains what it needs, and nothing is ever lost.

    Pass it to ``create_agent(model, tools, middleware=[MarkAgentMiddleware(...)])``.

    Lifecycle intercepts
    ────────────────────
    **Retrieve (inject context, synchronous with agent):**

    * ``abefore_agent``   — captures the user query at entry; fires background
      retrieval immediately so context is warm before the first model call.
    * ``abefore_model``   — parallel multi-query retrieval per model call:
      user goal query (cached) + latest AI reasoning excerpt (always fresh).
      Injects combined result into the ``mark_context`` state field.
    * ``wrap_model_call`` — appends ``[MARK project memory]`` to the system
      message via ``request.override()``.  No I/O here.

    **Observe (archive context, background / non-blocking):**

    * ``aafter_model``    — archives each substantive AI reasoning step.
    * ``awrap_tool_call`` — absorbs every tool result into MARK as it arrives.
    * ``aafter_agent``    — records final outcome and emits feedback signal.

    Args:
        backend:         A :class:`~mark.adapters.backend.MarkBackend`.
        agent_id:        Memory namespace (default ``"__mark__"``).
        max_context_chars: Truncate injected context at this character count (default 4000).
        write_outcomes:  Archive AI reasoning + tool results into MARK (default ``True``).
        auto_heal:       Call ``heal_gap`` when retrieval gap is HIGH/CRITICAL (default ``True``).
        compress:        Run LLM contextual compression on retrieved fragments
                         (default ``False`` — vector search + reranker only; fast).

    Example::

        from mark.runtime import Mark
        from mark.adapters.backend import LocalMarkBackend
        from mark.adapters.langchain.middleware import MarkAgentMiddleware
        from langchain.agents import create_agent

        mark  = Mark.local(".")
        mw    = MarkAgentMiddleware(backend=LocalMarkBackend(mark), agent_id="coder")
        agent = create_agent(llm, tools, middleware=[mw])
        result = agent.invoke({"messages": [...]})
    """

    # Expose the custom state field to the LangGraph agent graph.
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
        auto_heal: bool = True,
        compress: bool = False,
        observe_tool_results: bool = True,
        immediate_tool_observe: bool = False,
        retrieval_timeout_ms: int = 2500,
        thinking_model: "Any | None" = None,
        thinking_budget_tokens: int = 2000,
    ) -> None:
        """
        Args:
            thinking_model: Optional secondary LLM to use as a reasoning layer for
                non-thinking models.  When set and planning/reasoning patterns are
                detected in the agent's context, MARK calls this model first to
                generate a structured thought plan.  The thoughts + MARK memory are
                both injected into the main agent's system prompt and archived for
                future retrieval.  Accepts a model identifier string (e.g.,
                ``"ollama:qwen3:4b"``) or a ``BaseChatModel`` instance.
                Default ``None`` — thinking mode disabled.
            thinking_budget_tokens: Maximum tokens in the thinking output (default 2000).
                Only used when ``thinking_model`` is set.
        """
        if not _LANGCHAIN_AVAILABLE:
            raise ImportError(
                "LangChain is required for MarkAgentMiddleware. "
                'Install with: pip install "mark-sdk[langchain]" langchain'
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
        self._thinking_budget = thinking_budget_tokens
        # Per-session cache: human query → retrieved context.
        # Cleared by aafter_agent so the next session starts fresh.
        self._recall_cache: dict[str, str] = {}
        # Track the last injected context to skip redundant re-injections on
        # back-to-back model calls where neither the query nor the AI reasoning
        # changed (e.g. short tool-dispatch turns with no meaningful new context).
        self._last_injected_key: str = ""
        self._last_context_hash: str = ""
        self._last_prompt_context_hash: str = ""
        self._last_skill_hash: str = ""
        self._last_tool_trigger_hash: str = ""
        self._pending_observations: list[str] = []
        self._seen_observation_hashes: set[str] = set()
        self._diagnostics: list[dict[str, Any]] = []

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

    # ── background task helper ────────────────────────────────────────────

    def _bg(self, coro: Any) -> None:
        """Schedule *coro* as a background asyncio task — non-blocking.

        Used for all observe/feedback calls so MARK never adds latency to the
        agent's critical execution path.  The coroutine runs concurrently on the
        same event loop that LangGraph is already using.
        """
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(coro)
        except RuntimeError:
            pass  # No running loop (sync context); observation skipped

    @property
    def diagnostics(self) -> list[dict[str, Any]]:
        """Return lightweight middleware telemetry for tracing and benchmarks."""
        return list(self._diagnostics)

    def _record(self, kind: str, **data: Any) -> None:
        self._diagnostics.append({"kind": kind, **data})

    async def _backend_call(self, endpoint: str, *args: Any, **kwargs: Any) -> Any:
        started = perf_counter()
        call = getattr(self._backend, endpoint)
        try:
            result = await call(*args, **kwargs)
        except Exception as exc:
            self._record(
                "endpoint",
                endpoint=endpoint,
                ok=False,
                latency_ms=round((perf_counter() - started) * 1000, 1),
                error=str(exc)[:200],
            )
            raise
        self._record(
            "endpoint",
            endpoint=endpoint,
            ok=bool(getattr(result, "ok", False)),
            latency_ms=round((perf_counter() - started) * 1000, 1),
            error=getattr(result, "error", None),
        )
        return result

    def _backend_call_sync(self, endpoint: str, *args: Any, **kwargs: Any) -> Any:
        started = perf_counter()
        result = _run_sync(getattr(self._backend, endpoint)(*args, **kwargs))
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
        content = getattr(msg, "content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for block in content:
                if isinstance(block, dict):
                    text = block.get("text") or block.get("content") or ""
                    if text:
                        parts.append(str(text))
                else:
                    text = getattr(block, "text", None) or getattr(block, "content", None)
                    if text:
                        parts.append(str(text))
            return "\n".join(parts)
        return str(content or "")

    def _system_text(self, system_message: Any) -> str:
        if system_message is None:
            return ""
        content = getattr(system_message, "content", "")
        if isinstance(content, str):
            return content
        blocks = getattr(system_message, "content_blocks", None)
        if blocks is None:
            blocks = content if isinstance(content, list) else []
        parts: list[str] = []
        for block in blocks or []:
            if isinstance(block, dict):
                text = block.get("text") or block.get("content") or ""
                if text:
                    parts.append(str(text))
            else:
                text = getattr(block, "text", None) or getattr(block, "content", None)
                if text:
                    parts.append(str(text))
        return "\n".join(parts)

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
            or getattr(msg, "additional_kwargs", {}).get("name", "")
        )

    def _latest_tool_trigger(self, state: Any) -> tuple[str, str]:
        """Return a retrieval query from the latest useful tool/subagent result.

        Tool output is valuable context, but write/delete results often reflect
        unfinished intermediate work. Those are queued for observation at session
        end instead of being made immediately retrievable during the same run.
        Ordinary filesystem reads are also already present in the live context,
        so their right MARK action is observe-later, not retrieve-now.
        """
        messages = (state or {}).get("messages", []) if isinstance(state, dict) else []
        for msg in reversed(messages):
            role = getattr(msg, "type", None) or getattr(msg, "role", "")
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

    def _queue_observation(self, content: str) -> None:
        if not content:
            return
        normalized = " ".join(content.split())
        if len(normalized) < 40:
            return
        key = self._hash(normalized)
        if key in self._seen_observation_hashes:
            return
        self._seen_observation_hashes.add(key)
        self._pending_observations.append(normalized[:1200])

    async def _flush_observations_async(self) -> int:
        pending = self._pending_observations
        self._pending_observations = []
        count = 0
        for content in pending:
            result = await self._backend_call("observe", content, agent_id=self._agent_id)
            if result.ok:
                count += 1
        if count:
            self._record("observe_flush", count=count)
        return count

    def _flush_observations_sync(self) -> int:
        pending = self._pending_observations
        self._pending_observations = []
        count = 0
        for content in pending:
            result = self._backend_call_sync("observe", content, agent_id=self._agent_id)
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
          calls in one session; cached so only one backend round-trip per session).
        * ``ai_query``    — the latest AI reasoning excerpt (first 300 chars of
          the most recent AI message; changes each step; always retrieved fresh
          so context adapts as the agent's plan evolves).
        """
        messages = (state or {}).get("messages", []) if isinstance(state, dict) else []
        human_query = ""
        ai_query = ""
        for msg in reversed(messages):
            role = getattr(msg, "type", None) or getattr(msg, "role", "")
            if not human_query and role in ("human", "user"):
                human_query = str(getattr(msg, "content", "")).strip()
            if not ai_query and role == "ai":
                if getattr(msg, "tool_calls", None):
                    continue
                content = str(getattr(msg, "content", "")).strip()
                if len(content) > 20:
                    ai_query = content[:300]
            if human_query and ai_query:
                break
        return human_query, ai_query

    def _get_human_query_from_messages(self, messages: Any) -> str:
        """Extract last human message from a ModelRequest message list (fallback path)."""
        for msg in reversed(messages or []):
            role = getattr(msg, "type", None) or getattr(msg, "role", "")
            if role in ("human", "user"):
                return str(getattr(msg, "content", ""))
        return ""

    # ── core retrieval ────────────────────────────────────────────────────

    async def _recall(self, query: str, *, cache: bool = True) -> str:
        """Retrieve relevant MARK memory for *query*.

        Args:
            query: Retrieval query string.
            cache: When ``True`` (default), result is read from / written to
                   ``self._recall_cache``.  Pass ``cache=False`` for step-specific
                   queries (e.g., AI reasoning excerpt) that change each model call.
        """
        if cache and query in self._recall_cache:
            return self._recall_cache[query]

        retrieve_coro = self._backend_call(
            "retrieve", query, agent_id=self._agent_id, compress=self._compress
        )
        try:
            result = (
                await asyncio.wait_for(retrieve_coro, timeout=self._retrieval_timeout_s)
                if self._retrieval_timeout_s else await retrieve_coro
            )
        except TimeoutError:
            self._record(
                "endpoint",
                endpoint="retrieve",
                ok=False,
                latency_ms=round(self._retrieval_timeout_s * 1000, 1),
                error="timeout",
            )
            if cache:
                self._recall_cache[query] = ""
            return ""
        if not result.ok:
            return ""
        v = result.value
        text = (v.as_context() if hasattr(v, "as_context")
                else v.as_text() if hasattr(v, "as_text")
                else str(v))

        if self._auto_heal:
            gap_report = getattr(v, "gap_report", None)
            gap_sev = (
                getattr(getattr(gap_report, "severity", None), "value", "")
                if gap_report else ""
            )
            if not text or gap_sev in ("HIGH", "CRITICAL"):
                healed = await self._backend_call(
                    "heal_gap", query, agent_id=self._agent_id, compress=self._compress
                )
                if healed.ok:
                    hv = healed.value
                    ht = (hv.as_context() if hasattr(hv, "as_context")
                          else hv.as_text() if hasattr(hv, "as_text")
                          else str(hv))
                    if ht and "no relevant" not in ht.lower():
                        if cache:
                            self._recall_cache[query] = ht
                        return ht

        if cache:
            self._recall_cache[query] = text
        return self._truncate_context(text)

    # ── thinking engine ───────────────────────────────────────────────────

    def _should_think(self, query: str) -> bool:
        """Return True when *query* contains planning/reasoning trigger keywords.

        Checks both the user query and any additional text for the presence of
        keywords that signal the agent is entering a planning or design phase.
        Called only when ``thinking_model`` is configured.
        """
        lowered = query.lower()
        return any(t in lowered for t in self.THINKING_TRIGGERS)

    async def _think(self, query: str, context: str) -> str:
        """Call the thinking model to generate a structured reasoning plan.

        Invoked when ``thinking_model`` is set and ``_should_think`` returns True.
        Calls the secondary LLM with a focused prompt that asks it to reason
        through the task given the user query and retrieved MARK context.

        The output is a structured reasoning trace (SESSION INTENT, APPROACH,
        STEPS, RISKS) that is injected alongside the MARK memory block so the
        main agent sees both retrieved knowledge and pre-computed reasoning.

        The thinking output is also archived to MARK via a background observe()
        so future agents can retrieve the reasoning trace.

        Returns the thinking text (up to ``thinking_budget_tokens`` approximate
        chars), or empty string on error.
        """
        if self._thinking_model is None:
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
            if hasattr(self._thinking_model, "ainvoke"):
                response = await self._thinking_model.ainvoke(thinking_prompt)
            else:
                response = self._thinking_model.invoke(thinking_prompt)
            thought = str(getattr(response, "content", response)).strip()
            return thought[:self._thinking_budget * 4]  # approx chars to tokens ratio
        except Exception:
            return ""

    async def _retrieve_and_combine(
        self,
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
        tasks: list[Any] = [self._recall(human_query, cache=True)]
        if ai_query:
            tasks.append(self._recall(ai_query, cache=False))
        if tool_query:
            tasks.append(self._recall(tool_query, cache=False))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        blocks: list[str] = []
        seen: set[str] = set()
        for r in results:
            if isinstance(r, Exception) or not r:
                continue
            if "no relevant local memory found" in r:
                continue
            if r not in seen:
                seen.add(r)
                blocks.append(r)

        if not blocks:
            return ""

        combined = self._truncate_context("\n\n---\n\n".join(blocks))
        # Store under human_query key so wrap_model_call fallback path works
        self._recall_cache[human_query] = combined
        self._last_context_hash = self._hash(combined)
        self._record(
            "retrieve",
            queries=1 + bool(ai_query) + bool(tool_query),
            chars=len(combined),
            context_hash=self._last_context_hash,
        )
        return combined

    # ── abefore_agent: session entry intercept ────────────────────────────
    # Fires ONCE when the user query enters the agent.  Captures the query
    # immediately and warms the retrieval cache in background so context is
    # ready before the first model call — like a real-time search that
    # pre-loads results as the query arrives.

    async def abefore_agent(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        """Async: capture user query at session entry and warm the retrieval cache.

        Fires once when the agent starts.  Extracts the user's query and fires
        an initial retrieval so the cache is pre-populated before the first
        model call in ``abefore_model``.

        This is the entry intercept: MARK captures the user's intent the instant
        it enters the agent's context, matching it against accumulated project
        knowledge in real time.

        Returns ``None`` (state update will come from ``abefore_model``).
        """
        human_query, _ = self._build_queries(state)
        if human_query and human_query not in self._recall_cache:
            # Pre-warm cache in background — abefore_model will await it
            self._bg(self._recall(human_query, cache=True))
        return None

    def before_agent(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        """Sync: capture user query at session entry and warm retrieval cache."""
        human_query, _ = self._build_queries(state)
        if human_query and human_query not in self._recall_cache:
            _run_sync(self._recall(human_query, cache=True))
        return None

    # ── abefore_model: per-step multi-query retrieval ─────────────────────
    # Runs BEFORE every model call.  Fires two parallel retrieval pipelines:
    #
    #   Pipeline 1 — human goal (cached):
    #     Retrieves stable project knowledge (conventions, architecture, prior
    #     solutions) relevant to the overall task.
    #
    #   Pipeline 2 — AI reasoning excerpt (always fresh):
    #     Retrieves context matched to the agent's CURRENT thinking.  Changes
    #     each step so MARK adapts as the agent narrows its plan.  This is the
    #     per-step equivalent of LangChain's SummarizationMiddleware — but
    #     instead of lossy LLM compression, MARK injects precisely what the
    #     current step needs from its lossless archive.
    #
    # Results stored in self._recall_cache AND {"mark_context": ...} state
    # so wrap_model_call can read via either path (dual-path protection).

    async def abefore_model(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        """Async: multi-query parallel retrieval before every model call.

        Fires two retrieval pipelines simultaneously:

        * **Human goal** (cached) — stable project conventions and architecture.
        * **AI reasoning** (fresh) — context matched to what the agent is
          currently thinking; adapts step by step.

        Both pipelines run in parallel via ``asyncio.gather``.  Combined result
        is stored in ``self._recall_cache`` (for ``wrap_model_call`` fallback)
        and returned as ``{"mark_context": ...}`` for LangGraph state propagation.

        Returns ``None`` when no relevant memory is found.
        """
        human_query, ai_query = self._build_queries(state)
        if not human_query:
            return None
        tool_query, tool_hash = self._latest_tool_trigger(state)

        # Skip re-retrieval when neither the user goal nor the AI reasoning has
        # changed since the last injection.  This avoids burning extra latency and
        # graph-node visits on back-to-back model calls that follow short tool-
        # dispatch turns (the most common case after the first planning step).
        ai_trigger = ai_query if self._should_think(ai_query) else ""
        inject_key = f"{human_query}||{ai_trigger}||{tool_hash}"
        if inject_key == self._last_injected_key and human_query in self._recall_cache:
            cached = self._recall_cache[human_query]
            if cached:
                return {
                    "mark_context": cached,
                    "mark_context_meta": {
                        "chars": len(cached),
                        "context_hash": self._hash(cached),
                        "source": "cache",
                    },
                }
            return None

        refresh_tool = tool_hash and tool_hash != self._last_tool_trigger_hash
        combined = await self._retrieve_and_combine(
            human_query,
            ai_trigger,
            tool_query if refresh_tool else "",
        )
        if refresh_tool:
            self._last_tool_trigger_hash = tool_hash

        # Thinking mode — call secondary LLM when query contains planning triggers.
        # Transparent to the user: thoughts are injected alongside MARK memory and
        # archived for future retrieval.  Only active when thinking_model is set.
        if self._thinking_model and self._should_think(human_query):
            thoughts = await self._think(human_query, combined or "")
            if thoughts:
                thought_block = f"[MARK reasoning]\n{thoughts}\n[/MARK reasoning]"
                combined = (combined + "\n\n---\n\n" + thought_block) if combined else thought_block
                self._recall_cache[human_query] = combined
                # Archive the reasoning plan in background
                self._bg(self._backend_call(
                    "observe",
                    f"MARK reasoning plan for query '{human_query[:100]}':\n{thoughts[:400]}",
                    agent_id=self._agent_id,
                ))

        if not combined:
            return None

        self._last_injected_key = inject_key
        return {
            "mark_context": combined,
            "mark_context_meta": {
                "chars": len(combined),
                "context_hash": self._hash(combined),
                "source": "retrieve",
                "tool_trigger": bool(refresh_tool),
            },
        }

    def before_model(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        """Sync: triggered MARK retrieval before every model call."""
        return _run_sync(self.abefore_model(state, runtime))

    # ── wrap_model_call: inject MARK context into system message ──────────
    # Synchronous wrap hook — no I/O, reads from state or instance cache.
    #
    # Dual-path read protects against LangGraph building ModelRequest before
    # applying node-hook state updates:
    #
    #   Path 1: request.state["mark_context"]  — set by abefore_model if
    #           LangGraph propagates hook updates before building the request.
    #   Path 2: self._recall_cache[human_query] — always populated by
    #           abefore_model via _retrieve_and_combine; reliable fallback.

    def wrap_model_call(
        self,
        request: Any,
        handler: Callable[[Any], Any],
    ) -> Any:
        """Sync: inject cached MARK context into the system message.

        Appends a ``[MARK project memory]`` block to ``request.system_message``
        via ``request.override()`` before forwarding to the model.  Reads from
        two paths to handle different LangGraph scheduling behaviours:

        * **Primary**: ``request.state["mark_context"]`` — from ``abefore_model``
          state update.
        * **Fallback**: ``self._recall_cache`` keyed by last human message —
          always populated by ``abefore_model``.

        Pure read + transform — no backend calls, no event-loop interaction.
        """
        # Path 1: state-level (works when LangGraph propagates hook updates first)
        state = getattr(request, "state", None) or {}
        context = (
            state.get("mark_context", "")
            if isinstance(state, dict)
            else getattr(state, "mark_context", "")
        )

        # Path 2: instance cache fallback (always populated by abefore_model)
        if not context:
            query = self._get_human_query_from_messages(getattr(request, "messages", []))
            context = self._recall_cache.get(query, "")

        if not context or "no relevant local memory found" in context:
            return handler(request)

        system_text = self._system_text(getattr(request, "system_message", None))
        skill_hash = self._hash(system_text)
        if skill_hash != self._last_skill_hash:
            skill = self._score_skill(system_text)
            self._record("skill", skill_hash=skill_hash, **skill)
            self._last_skill_hash = skill_hash
        context = self._truncate_context(context)
        context_hash = self._hash(context)
        if context_hash == self._last_prompt_context_hash:
            self._record("inject_skip", chars=len(context), context_hash=context_hash)
            return handler(request)
        self._last_prompt_context_hash = context_hash
        self._record("inject", chars=len(context), context_hash=context_hash)
        mark_block = (
            "[MARK project memory]\n"
            "Use this as durable project memory. Prefer live tool results for "
            "current-session file contents.\n"
            f"{context}\n"
            "[/MARK project memory]"
        )
        new_content = list(request.system_message.content_blocks) + [
            {"type": "text", "text": mark_block}
        ]
        return handler(request.override(system_message=_SystemMessage(content=new_content)))

    def wrap_tool_call(self, request: Any, handler: Callable[[Any], Any]) -> Any:
        """Sync: execute tool and queue useful result for session-end archival."""
        result = handler(request)
        if self._write_outcomes and self._observe_tool_results:
            self._observe_tool_result(request, result, immediate=False)
        return result

    # ── aafter_model: archive AI reasoning step (background) ─────────────
    # Fires AFTER every model call.  Archives the latest AI reasoning turn
    # immediately — incremental archival so older turns remain retrievable
    # even after they scroll out of the active context window.
    # Fire-and-forget: does not block the agent.

    async def aafter_model(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        """Async: archive each AI reasoning step into MARK (background).

        Runs after every model call.  Extracts the latest AI message and archives
        it via ``observe()`` as a background task.  Only substantive turns
        (>100 chars, no tool-dispatch payloads) are archived to avoid noise.

        This is the incremental push side of MARK's context-window management:
        as the agent's reasoning accumulates, each step is archived so nothing
        is lost even when the active window fills up.

        Returns ``None`` (no state update needed).
        """
        if not self._write_outcomes:
            return None
        messages = (state or {}).get("messages", []) if isinstance(state, dict) else []
        for msg in reversed(messages):
            if getattr(msg, "type", None) == "ai":
                content = str(getattr(msg, "content", "")).strip()
                has_tool_calls = bool(getattr(msg, "tool_calls", None))
                if content and len(content) > 100 and not has_tool_calls:
                    self._queue_observation(f"Agent reasoning: {content[:600]}")
                break
        return None

    def after_model(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        """Sync: queue substantive AI reasoning for session-end archival."""
        if not self._write_outcomes:
            return None
        messages = (state or {}).get("messages", []) if isinstance(state, dict) else []
        for msg in reversed(messages):
            if getattr(msg, "type", None) == "ai":
                content = self._message_text(msg).strip()
                has_tool_calls = bool(getattr(msg, "tool_calls", None))
                if content and len(content) > 100 and not has_tool_calls:
                    self._queue_observation(f"Agent reasoning: {content[:600]}")
                break
        return None

    def _observe_tool_result(self, request: Any, result: Any, *, immediate: bool) -> None:
        tool_name = (getattr(request, "tool_call", None) or {}).get("name", "unknown_tool")
        lowered = tool_name.lower()
        content = str(getattr(result, "content", "") or "")[:900]
        if not content or len(content) < 20:
            return
        observation = f"Tool [{tool_name}] returned: {content}"
        if immediate and not any(skip in lowered for skip in ("write", "delete", "remove")):
            self._bg(self._backend_call("observe", observation, agent_id=self._agent_id))
            self._record("observe_immediate", tool=tool_name, chars=len(content))
        else:
            self._queue_observation(observation)

    # ── awrap_tool_call: absorb tool results into MARK (background) ───────
    # Wraps EVERY tool call.  After the tool executes, the result is absorbed
    # into MARK as an observation — background, non-blocking.
    #
    # "What the agent sees, MARK archives."  Tool results are a major source
    # of context for the agent; capturing them lets MARK retrieve tool output
    # fragments in future sessions without re-running the tool.

    async def awrap_tool_call(self, request: Any, handler: Callable[[Any], Any]) -> Any:
        """Async: execute tool and archive result into MARK (background).

        Runs the tool via ``handler(request)``, then fires a background observe
        of the tool result so it becomes part of MARK's retrievable knowledge.
        The result is returned unchanged — MARK is completely transparent.

        Tool results absorbed here include file reads, search results, API
        responses — everything the agent loads into its context window.  MARK
        captures it all so future retrieval can surface relevant fragments
        without re-executing the tool.

        Returns the tool result unchanged.
        """
        result = await handler(request)
        if self._write_outcomes and self._observe_tool_results:
            self._observe_tool_result(request, result, immediate=self._immediate_tool_observe)
        return result

    # ── aafter_agent: session end — feedback + cache cleanup ─────────────
    # Runs ONCE after the agent finishes.  Emits a retrieval-quality feedback
    # signal so MARK can score how useful the injected context was, then clears
    # the per-session cache.  All backend calls are fire-and-forget.

    async def aafter_agent(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        """Async: emit feedback signal and clear session cache.

        Fires once when the agent graph reaches its terminal node.  Finds the
        last AI message, fires a background ``feedback()`` call, and clears
        ``self._recall_cache`` so the next ``agent.invoke()`` starts fresh.

        Returns ``None`` (no state update required).
        """
        messages = (state or {}).get("messages", []) if isinstance(state, dict) else []
        if self._write_outcomes:
            for msg in reversed(messages):
                if getattr(msg, "type", None) == "ai":
                    content = str(getattr(msg, "content", ""))[:200]
                    self._queue_observation(f"Agent completed task. Response: {content}")
                    self._bg(self._backend_call(
                        "feedback",
                        agent_id=self._agent_id,
                        outcome="completed",
                        summary=content,
                    ))
                    break
            await self._flush_observations_async()
        self._recall_cache.clear()
        self._last_injected_key = ""
        self._last_context_hash = ""
        self._last_prompt_context_hash = ""
        self._last_tool_trigger_hash = ""
        return None

    def after_agent(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        """Sync: flush queued observations and clear per-session cache."""
        messages = (state or {}).get("messages", []) if isinstance(state, dict) else []
        if self._write_outcomes:
            for msg in reversed(messages):
                if getattr(msg, "type", None) == "ai":
                    content = self._message_text(msg)[:200]
                    self._queue_observation(f"Agent completed task. Response: {content}")
                    self._backend_call_sync(
                        "feedback",
                        agent_id=self._agent_id,
                        outcome="completed",
                        summary=content,
                    )
                    break
            self._flush_observations_sync()
        self._recall_cache.clear()
        self._last_injected_key = ""
        self._last_context_hash = ""
        self._last_prompt_context_hash = ""
        self._last_tool_trigger_hash = ""
        return None


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

        mw = MarkMemoryMiddleware(backend=backend, agent_id="coder")
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
        """Retrieve and format memory context for the query."""
        result = await self.backend.retrieve(query, agent_id=self.agent_id,
                                             session_id=self.session_id)
        if not result.ok:
            return ""
        v = result.value
        text = (v.as_context() if hasattr(v, "as_context")
                else v.as_text() if hasattr(v, "as_text")
                else str(v))
        return text[: self.max_context_chars]

    async def remember(self, content: str, **meta: Any) -> None:
        """Store content into MARK memory."""
        if not self.write_outcomes:
            return
        await self.backend.observe(content, agent_id=self.agent_id, metadata=meta)

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
        if not context or "no relevant local memory found" in context:
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
