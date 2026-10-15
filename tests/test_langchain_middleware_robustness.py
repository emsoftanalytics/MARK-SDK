"""Robustness tests for MarkAgentMiddleware: hook/backend failures never break the
agent, background observation tasks are tracked and drained, and retrieval
timeouts never block the event loop.
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

from langchain_core.messages import HumanMessage

from mark.adapters.backend import BackendResult, LocalMarkBackend
from mark.adapters.langchain.middleware import MarkAgentMiddleware

from _helpers_langchain_async import (
    KeywordBackend,
    ToolCallingFakeModel,
    _BlockingMemory,
    _agent,
    _invoke,
)


class ExplodingBackend(KeywordBackend):
    async def retrieve(self, query: str, **options: Any) -> BackendResult:
        raise RuntimeError("retrieve exploded")

    async def observe(self, content: str, **options: Any) -> BackendResult:
        raise RuntimeError("observe exploded")

    async def feedback(self, **payload: Any) -> BackendResult:
        raise RuntimeError("feedback exploded")


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
