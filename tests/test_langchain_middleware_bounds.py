"""MarkAgentMiddleware edge cases: list-content messages, bounded diagnostics and
run-state eviction, and thinking-model prewarm behaviour.
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

from mark.adapters.backend import LocalMarkBackend
from mark.adapters.langchain.middleware import MarkAgentMiddleware

from _helpers_langchain_async import KeywordBackend, _BlockingMemory


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
