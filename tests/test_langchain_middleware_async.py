"""End-to-end tests for MarkAgentMiddleware inside create_agent: memory injection
on every model call, concurrent-run isolation, stale-context handling, and
empty-store / auto_heal behaviour. Uses fakes from _helpers_langchain_async.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("langchain")
pytest.importorskip("langgraph")

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.memory import InMemorySaver

from mark.adapters.backend import LocalMarkBackend
from mark.adapters.langchain.middleware import MarkAgentMiddleware
from mark.types.gap import GapSeverity

from _helpers_langchain_async import KeywordBackend, _agent, _invoke


MEMORY_TAG = "[MARK project memory]"


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
