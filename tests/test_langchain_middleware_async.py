"""End-to-end and robustness tests for MarkAgentMiddleware inside create_agent.

All tests use a small in-process fake chat model (with ``bind_tools`` support)
and fake/local backends — no network, no API keys.
"""
from __future__ import annotations

import asyncio
import logging
import time
from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("langchain")
pytest.importorskip("langgraph")

from langchain.agents import create_agent
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import Field

from mark.adapters.backend import BackendResult, LocalMarkBackend
from mark.adapters.langchain.middleware import MarkAgentMiddleware, content_text
from mark.types.gap import GapReport, GapSeverity

MEMORY_TAG = "[MARK project memory]"


# ── fakes ──────────────────────────────────────────────────────────────────


class ToolCallingFakeModel(BaseChatModel):
    """Deterministic fake: first call per turn requests ``lookup``, then answers.

    Every request's system text is recorded in ``calls`` as
    ``(human_query, system_text)``.  Responses depend only on the request, so
    concurrent runs on one model never interfere.
    """

    calls: list[tuple[str, str]] = Field(default_factory=list)
    fail: bool = False

    @property
    def _llm_type(self) -> str:
        return "tool-calling-fake"

    def bind_tools(self, tools: Any, **kwargs: Any) -> "ToolCallingFakeModel":
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        if self.fail:
            raise RuntimeError("model exploded")
        system = "\n".join(
            content_text(m.content) for m in messages if isinstance(m, SystemMessage)
        )
        humans = [m for m in messages if isinstance(m, HumanMessage)]
        human = content_text(humans[-1].content) if humans else ""
        self.calls.append((human, system))
        last = messages[-1]
        if isinstance(last, ToolMessage):
            message = AIMessage(content=f"Final answer for {human}: {content_text(last.content)}")
        else:
            message = AIMessage(
                content="",
                tool_calls=[{
                    "name": "lookup",
                    "args": {"topic": human},
                    "id": f"call-{len(self.calls)}",
                    "type": "tool_call",
                }],
            )
        return ChatResult(generations=[ChatGeneration(message=message)])


@tool
def lookup(topic: str) -> str:
    """Look up project facts about a topic."""
    return f"lookup result for {topic}: the service exposes a health endpoint."


class _Fragment:
    def __init__(self, fid: str, content: str) -> None:
        self.id = fid
        self.content = content


class _Result:
    """Mimics RetrievalResult: structured fragments + gap report + as_context()."""

    def __init__(self, query: str, fragments: list[_Fragment], severity: Any = GapSeverity.NONE) -> None:
        self.query = query
        self.fragments = fragments
        self.gap_report = GapReport(severity=severity)

    def as_context(self) -> str:
        lines = [f"<memory_context query={self.query!r}>"]
        if not self.fragments:
            lines.append("  [No relevant memory found]")
        lines.extend(f"  {f.content}" for f in self.fragments)
        lines.append("</memory_context>")
        return "\n".join(lines)


class KeywordBackend:
    """Async fake backend: returns memory whose keyword appears in the query."""

    def __init__(
        self,
        memories: dict[str, str] | None = None,
        *,
        delay: float = 0.0,
        healed: str = "",
        severity: Any = GapSeverity.NONE,
    ) -> None:
        self.memories = memories or {}
        self.delay = delay
        self.healed = healed
        self.severity = severity
        self.retrieve_calls: list[str] = []
        self.heal_calls: list[str] = []
        self.observed: list[str] = []
        self.feedback_payloads: list[dict[str, Any]] = []
        self.in_flight = 0
        self.max_in_flight = 0

    async def retrieve(self, query: str, **options: Any) -> BackendResult:
        self.retrieve_calls.append(query)
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            if self.delay:
                await asyncio.sleep(self.delay)
        finally:
            self.in_flight -= 1
        fragments = [
            _Fragment(f"frag-{key}", text)
            for key, text in self.memories.items()
            if key in query.lower()
        ]
        return BackendResult(True, _Result(query, fragments, self.severity))

    async def heal_gap(self, query: str, **options: Any) -> BackendResult:
        self.heal_calls.append(query)
        fragments = [_Fragment("frag-healed", self.healed)] if self.healed else []
        return BackendResult(True, _Result(query, fragments))

    async def write(self, content: str, **options: Any) -> BackendResult:
        return BackendResult(True, content)

    async def observe(self, content: str, **options: Any) -> BackendResult:
        self.observed.append(content)
        return BackendResult(True, "fid", {"fragment_id": "fid"})

    async def compress(self, query: str, candidates: list[Any], **options: Any) -> BackendResult:
        return BackendResult(True, candidates)

    async def list_sessions(self, agent_id: str, **options: Any) -> BackendResult:
        return BackendResult(True, [])

    async def feedback(self, **payload: Any) -> BackendResult:
        self.feedback_payloads.append(payload)
        return BackendResult(True, payload)


class ExplodingBackend(KeywordBackend):
    async def retrieve(self, query: str, **options: Any) -> BackendResult:
        raise RuntimeError("retrieve exploded")

    async def observe(self, content: str, **options: Any) -> BackendResult:
        raise RuntimeError("observe exploded")

    async def feedback(self, **payload: Any) -> BackendResult:
        raise RuntimeError("feedback exploded")


def _agent(mw: MarkAgentMiddleware, model: ToolCallingFakeModel | None = None, **kwargs: Any) -> Any:
    model = model or ToolCallingFakeModel()
    return model, create_agent(model=model, tools=[lookup], middleware=[mw], **kwargs)


def _invoke(agent: Any, text: str, mode: str, **kwargs: Any) -> dict[str, Any]:
    payload = {"messages": [HumanMessage(content=text)]}
    if mode == "sync":
        return agent.invoke(payload, **kwargs)
    return asyncio.run(agent.ainvoke(payload, **kwargs))


# ── 1 + 3: sync and async end-to-end, memory on every model call ──────────


@pytest.mark.parametrize("mode", ["sync", "async"])
def test_create_agent_injects_memory_on_every_model_call(mode: str) -> None:
    backend = KeywordBackend({"health": "Health checks live at /api/v1/health."})
    mw = MarkAgentMiddleware(backend, include_skill=False)
    model, agent = _agent(mw)

    result = _invoke(agent, "Add a health check endpoint", mode)

    assert result["messages"][-1].content.startswith("Final answer")
    assert len(model.calls) == 2, "tool round trip should produce two model calls"
    for _, system in model.calls:
        assert MEMORY_TAG in system
        assert "/api/v1/health" in system
        assert system.count(MEMORY_TAG) == 1
    assert mw.active_runs == 0
    assert result.get("mark_context", "") == ""
    # The tool result and the final answer were archived at run end.
    assert any("Tool [lookup] returned" in item for item in backend.observed)
    assert any("Agent completed task" in item for item in backend.observed)
    assert backend.feedback_payloads
    assert backend.feedback_payloads[-1]["fragment_ids"] == ["frag-health"]


def test_existing_identical_block_is_not_duplicated() -> None:
    backend = KeywordBackend({"health": "Health checks live at /api/v1/health."})
    mw = MarkAgentMiddleware(backend, include_skill=False)
    state = {"messages": [HumanMessage(content="health check")]}
    update = mw.before_model(state, runtime=None)
    block = mw._memory_block(update["mark_context"])

    seen: list[Any] = []
    request = SimpleNamespace(
        state=update,
        messages=state["messages"],
        system_message=SystemMessage(content=f"base\n{block}"),
        override=lambda **kw: (_ for _ in ()).throw(AssertionError("must not override")),
    )
    mw.wrap_model_call(request, lambda r: seen.append(r))
    assert seen == [request]
    assert any(e["kind"] == "inject_skip" for e in mw.diagnostics)


# ── 2: concurrent runs are isolated ───────────────────────────────────────


def test_concurrent_ainvoke_runs_are_isolated() -> None:
    backend = KeywordBackend(
        {
            "alpha": "ALPHA-MEMORY: alpha service uses Postgres.",
            "beta": "BETA-MEMORY: beta service uses Redis.",
        },
        delay=0.05,
    )
    mw = MarkAgentMiddleware(backend, include_skill=False)
    model, agent = _agent(mw)

    async def both() -> list[dict[str, Any]]:
        return await asyncio.gather(
            agent.ainvoke({"messages": [HumanMessage(content="alpha task please")]}),
            agent.ainvoke({"messages": [HumanMessage(content="beta task please")]}),
        )

    results = asyncio.run(both())

    assert backend.max_in_flight >= 2, "runs should overlap"
    assert all(r["messages"][-1].content.startswith("Final answer") for r in results)
    assert len(model.calls) == 4
    for human, system in model.calls:
        if "alpha" in human:
            assert "ALPHA-MEMORY" in system and "BETA-MEMORY" not in system
        else:
            assert "BETA-MEMORY" in system and "ALPHA-MEMORY" not in system
    # Each run flushed its own observations exactly once.
    for topic in ("alpha", "beta"):
        tool_obs = [o for o in backend.observed if o.startswith("Tool [lookup]") and topic in o]
        assert len(tool_obs) == 1
    fragment_sets = sorted(tuple(p["fragment_ids"]) for p in backend.feedback_payloads)
    assert fragment_sets == [("frag-alpha",), ("frag-beta",)]
    assert mw.active_runs == 0


# ── 4: stale context never leaks into a later turn ───────────────────────


@pytest.mark.parametrize("mode", ["sync", "async"])
def test_checkpointed_context_not_reused_on_unrelated_turn(mode: str) -> None:
    backend = KeywordBackend({"alpha": "ALPHA-MEMORY: alpha service uses Postgres."})
    mw = MarkAgentMiddleware(backend, include_skill=False)
    model, agent = _agent(mw, checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": f"thread-{mode}"}}

    first = _invoke(agent, "alpha question", mode, config=config)
    assert first.get("mark_context", "") == ""
    assert all("ALPHA-MEMORY" in system for _, system in model.calls)

    model.calls.clear()
    _invoke(agent, "something unrelated entirely", mode, config=config)
    assert model.calls
    for _, system in model.calls:
        assert MEMORY_TAG not in system
        assert "ALPHA-MEMORY" not in system


def test_before_model_always_returns_mark_context() -> None:
    mw = MarkAgentMiddleware(KeywordBackend(), include_skill=False)
    assert mw.before_model({"messages": []}, runtime=None)["mark_context"] == ""
    update = mw.before_model({"messages": [HumanMessage(content="no hits here")]}, runtime=None)
    assert update["mark_context"] == ""


# ── 5 + 11: empty store and auto_heal ─────────────────────────────────────


def test_empty_result_injects_nothing() -> None:
    backend = KeywordBackend({})
    mw = MarkAgentMiddleware(backend, include_skill=False, auto_heal=False)
    model, agent = _agent(mw)
    agent.invoke({"messages": [HumanMessage(content="anything at all")]})
    assert all(MEMORY_TAG not in system for _, system in model.calls)
    assert all("No relevant memory" not in system for _, system in model.calls)
    assert backend.heal_calls == []


def test_auto_heal_runs_on_empty_result_and_is_truncated() -> None:
    backend = KeywordBackend({}, healed="H" * 5000)
    mw = MarkAgentMiddleware(backend, include_skill=False, auto_heal=True, max_context_chars=300)
    update = mw.before_model({"messages": [HumanMessage(content="unknown topic")]}, runtime=None)
    assert backend.heal_calls == ["unknown topic"]
    assert "HHHH" in update["mark_context"]
    assert len(update["mark_context"]) <= 300
    assert update["mark_context"].endswith("[MARK context truncated]")


@pytest.mark.parametrize("severity", [GapSeverity.HIGH, GapSeverity.CRITICAL, "high"])
def test_auto_heal_runs_on_high_gap_severity(severity: Any) -> None:
    backend = KeywordBackend({"topic": "weak match"}, healed="healed evidence", severity=severity)
    mw = MarkAgentMiddleware(backend, include_skill=False, auto_heal=True)
    update = mw.before_model({"messages": [HumanMessage(content="topic x")]}, runtime=None)
    assert backend.heal_calls
    assert "healed evidence" in update["mark_context"]


def test_auto_heal_skipped_on_low_severity() -> None:
    backend = KeywordBackend({"topic": "good match"}, healed="healed", severity=GapSeverity.LOW)
    mw = MarkAgentMiddleware(backend, include_skill=False, auto_heal=True)
    update = mw.before_model({"messages": [HumanMessage(content="topic x")]}, runtime=None)
    assert backend.heal_calls == []
    assert "good match" in update["mark_context"]


def test_local_backend_default_namespace_matches_simple_memory(tmp_path: Any) -> None:
    from mark.runtime import Mark

    with Mark.local(project_path=tmp_path) as mark:
        mw = MarkAgentMiddleware(LocalMarkBackend(mark), include_skill=False, write_outcomes=False)
        model, agent = _agent(mw)
        agent.invoke({"messages": [HumanMessage(content="How is authentication implemented?")]})
        assert all(MEMORY_TAG not in system for _, system in model.calls), "empty store injects nothing"

        mark.memory.block("architecture").write(
            "Authentication uses FastAPI dependencies and JWT.", importance=0.9
        )
        model.calls.clear()
        agent.invoke({"messages": [HumanMessage(content="How is authentication implemented?")]})
        assert model.calls
        assert all("JWT" in system for _, system in model.calls)


# ── 7: hooks never break the agent ────────────────────────────────────────


@pytest.mark.parametrize("mode", ["sync", "async"])
def test_backend_failures_never_break_agent(mode: str, caplog: Any) -> None:
    mw = MarkAgentMiddleware(ExplodingBackend(), include_skill=False, immediate_tool_observe=True)
    model, agent = _agent(mw)
    with caplog.at_level(logging.WARNING, logger="mark.adapters.langchain"):
        result = _invoke(agent, "health check", mode)
    assert result["messages"][-1].content.startswith("Final answer")
    errors = [e for e in mw.diagnostics if e["kind"] == "endpoint" and not e["ok"]]
    assert {e["endpoint"] for e in errors} >= {"retrieve", "observe", "feedback"}
    assert any(r.name == "mark.adapters.langchain" for r in caplog.records)
    assert mw.active_runs == 0


def test_internal_hook_error_is_recorded_not_raised(caplog: Any) -> None:
    mw = MarkAgentMiddleware(KeywordBackend(), include_skill=False)

    def boom(state: Any) -> Any:
        raise ValueError("bad state")

    mw._build_queries = boom  # type: ignore[method-assign]
    with caplog.at_level(logging.WARNING, logger="mark.adapters.langchain"):
        update = mw.before_model({"messages": []}, runtime=None)
    assert update["mark_context"] == ""
    assert any(e["kind"] == "hook_error" and e["hook"] == "before_model" for e in mw.diagnostics)


@pytest.mark.parametrize("mode", ["sync", "async"])
def test_run_state_released_when_model_raises(mode: str) -> None:
    mw = MarkAgentMiddleware(KeywordBackend({"x": "memory x"}), include_skill=False)
    model, agent = _agent(mw, ToolCallingFakeModel(fail=True))
    with pytest.raises(RuntimeError, match="model exploded"):
        _invoke(agent, "x task", mode)
    assert mw.active_runs == 0
    assert any(e["kind"] == "run_aborted" for e in mw.diagnostics)


def test_graph_interrupt_keeps_run_state() -> None:
    from langgraph.errors import GraphInterrupt

    mw = MarkAgentMiddleware(KeywordBackend(), include_skill=False)
    state = {"messages": [], "mark_run_id": "r1"}
    mw._run(state)

    def interrupt(_request: Any) -> Any:
        raise GraphInterrupt(())

    with pytest.raises(GraphInterrupt):
        mw.wrap_tool_call(SimpleNamespace(state=state, tool_call={"name": "t"}), interrupt)
    assert mw.active_runs == 1


# ── 8: background tasks are tracked, logged and drained ──────────────────


def test_async_background_observations_are_drained() -> None:
    class SlowObserve(KeywordBackend):
        async def observe(self, content: str, **options: Any) -> BackendResult:
            await asyncio.sleep(0.05)
            return await super().observe(content, **options)

    backend = SlowObserve()
    mw = MarkAgentMiddleware(backend, include_skill=False, immediate_tool_observe=True)
    _, agent = _agent(mw)
    asyncio.run(agent.ainvoke({"messages": [HumanMessage(content="health check")]}))
    assert any(o.startswith("Tool [lookup]") for o in backend.observed)
    assert mw._bg_tasks == set()


def test_background_task_exception_is_logged(caplog: Any) -> None:
    mw = MarkAgentMiddleware(KeywordBackend(), include_skill=False)

    async def failing() -> None:
        raise RuntimeError("bg failure")

    async def drive() -> None:
        run = mw._run({})
        await mw._spawn(run, failing())
        await mw._drain(run)
        await asyncio.sleep(0)

    with caplog.at_level(logging.WARNING, logger="mark.adapters.langchain"):
        asyncio.run(drive())
    assert any(e["kind"] == "hook_error" and e["hook"] == "background" for e in mw.diagnostics)
    assert "bg failure" in caplog.text
    assert mw._bg_tasks == set()


def test_sync_path_runs_immediate_observation_inline() -> None:
    backend = KeywordBackend()
    mw = MarkAgentMiddleware(backend, include_skill=False, immediate_tool_observe=True)
    request = SimpleNamespace(state={}, tool_call={"name": "read_docs"})
    mw.wrap_tool_call(request, lambda _r: SimpleNamespace(content="documentation " * 10))
    assert any(o.startswith("Tool [read_docs]") for o in backend.observed)


# ── 6 + 9: timeouts work and the loop is never blocked ────────────────────


class _BlockingMemory:
    def __init__(self, seconds: float) -> None:
        self.seconds = seconds

    def retrieve_sync(self, query: str, **options: Any) -> Any:
        time.sleep(self.seconds)
        return _Result(query, [_Fragment("f", "slow memory")])


def _blocking_mark(seconds: float) -> Any:
    memory = _BlockingMemory(seconds)
    return SimpleNamespace(runtime=SimpleNamespace(memory=lambda agent_id: memory))


def test_local_backend_does_not_block_event_loop() -> None:
    backend = LocalMarkBackend(_blocking_mark(0.3))

    async def drive() -> tuple[int, BackendResult]:
        ticks = 0
        done = asyncio.Event()

        async def heartbeat() -> None:
            nonlocal ticks
            while not done.is_set():
                ticks += 1
                await asyncio.sleep(0.01)

        beat = asyncio.create_task(heartbeat())
        result = await backend.retrieve("q")
        done.set()
        await beat
        return ticks, result

    ticks, result = asyncio.run(drive())
    assert result.ok
    assert ticks >= 10


@pytest.mark.parametrize("mode", ["sync", "async"])
def test_retrieval_timeout_is_enforced(mode: str) -> None:
    mw = MarkAgentMiddleware(
        LocalMarkBackend(_blocking_mark(1.0)), include_skill=False, retrieval_timeout_ms=50
    )
    state = {"messages": [HumanMessage(content="slow query")]}
    started = time.perf_counter()
    if mode == "sync":
        update = mw.before_model(state, runtime=None)
    else:
        update = asyncio.run(mw.abefore_model(state, runtime=None))
    elapsed = time.perf_counter() - started
    assert elapsed < 0.8
    assert update["mark_context"] == ""
    assert any(e.get("error") == "timeout" for e in mw.diagnostics)


# ── 10: list-content messages ─────────────────────────────────────────────


def test_list_content_messages_use_text_parts() -> None:
    backend = KeywordBackend({"health": "Health memory."})
    mw = MarkAgentMiddleware(backend, include_skill=False)
    message = HumanMessage(content=[
        {"type": "text", "text": "Add a health check"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
    ])
    update = mw.before_model({"messages": [message]}, runtime=None)
    assert backend.retrieve_calls == ["Add a health check"]
    assert "Health memory." in update["mark_context"]
    assert mw._get_human_query_from_messages([message]) == "Add a health check"


# ── 12: bounded structures ────────────────────────────────────────────────


def test_diagnostics_and_observation_hashes_are_bounded() -> None:
    mw = MarkAgentMiddleware(KeywordBackend(), include_skill=False)
    for i in range(3000):
        mw._record("noise", i=i)
    assert len(mw.diagnostics) == 1000
    run = mw._run({})
    for i in range(5000):
        mw._queue_observation(run, f"observation number {i} with enough text to be kept")
    assert len(mw._seen_observation_hashes) <= 4096


def test_runs_that_never_finish_are_evicted() -> None:
    mw = MarkAgentMiddleware(KeywordBackend(), include_skill=False)
    for i in range(300):
        mw._run({"mark_run_id": f"run-{i}"})
    assert mw.active_runs == 256


# ── 18: thinking model ────────────────────────────────────────────────────


class _ThinkingModel:
    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.sync_calls = 0
        self.async_calls = 0

    def invoke(self, prompt: str) -> Any:
        self.sync_calls += 1
        if self.fail:
            raise RuntimeError("thinking down")
        return SimpleNamespace(content="PLAN: do the thing")

    async def ainvoke(self, prompt: str) -> Any:
        self.async_calls += 1
        return SimpleNamespace(content="PLAN: do the thing")


def test_thinking_model_uses_sync_invoke_on_sync_path() -> None:
    thinker = _ThinkingModel()
    mw = MarkAgentMiddleware(KeywordBackend(), include_skill=False, thinking_model=thinker)
    update = mw.before_model({"messages": [HumanMessage(content="plan the design")]}, runtime=None)
    assert thinker.sync_calls == 1 and thinker.async_calls == 0
    assert "PLAN: do the thing" in update["mark_context"]

    asyncio.run(mw.abefore_model(
        {"messages": [HumanMessage(content="plan another design")], "mark_run_id": "r2"},
        runtime=None,
    ))
    assert thinker.async_calls == 1


def test_thinking_failure_is_logged(caplog: Any) -> None:
    mw = MarkAgentMiddleware(
        KeywordBackend(), include_skill=False, thinking_model=_ThinkingModel(fail=True)
    )
    with caplog.at_level(logging.WARNING, logger="mark.adapters.langchain"):
        update = mw.before_model({"messages": [HumanMessage(content="plan it")]}, runtime=None)
    assert update["mark_context"] == ""
    assert "thinking down" in caplog.text
    assert any(e["kind"] == "thinking" and e["ok"] is False for e in mw.diagnostics)


def test_prewarm_timeout_is_not_paid_twice() -> None:
    memory = _BlockingMemory(1.0)
    calls: list[str] = []
    original = memory.retrieve_sync

    def counting(query: str, **options: Any) -> Any:
        calls.append(query)
        return original(query, **options)

    memory.retrieve_sync = counting  # type: ignore[method-assign]
    mark = SimpleNamespace(runtime=SimpleNamespace(memory=lambda agent_id: memory))
    mw = MarkAgentMiddleware(LocalMarkBackend(mark), include_skill=False, retrieval_timeout_ms=150)

    async def drive() -> tuple[float, dict]:
        state: dict = {"messages": [HumanMessage(content="slow prewarmed query")]}
        state.update(await mw.abefore_agent(state, runtime=None) or {})
        started = time.perf_counter()
        update = await mw.abefore_model(state, runtime=None)
        return time.perf_counter() - started, update

    elapsed, update = asyncio.run(drive())
    assert update["mark_context"] == ""
    assert elapsed < 0.28  # one timeout budget, not two
    assert calls.count("slow prewarmed query") == 1
