"""MarkAgentMiddleware: one token budget per model call, and history="mark" replacement.

Uses the fake tool-calling model and keyword backend from _helpers_langchain_async;
the model here additionally records the full message list of every call.
"""
from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("langchain")
pytest.importorskip("langgraph")

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import Field

from mark.adapters.backend import BackendResult, LocalMarkBackend
from mark.adapters.langchain.middleware import MarkAgentMiddleware, content_text
from mark.context.budget import estimate_tokens

from _helpers_langchain_async import KeywordBackend, ToolCallingFakeModel, _agent, _invoke

MEMORY_TAG = "[MARK memory]"


class RecordingModel(ToolCallingFakeModel):
    """Also records every request's non-system messages."""

    seen: list[list[Any]] = Field(default_factory=list)

    def _generate(self, messages: Any, stop: Any = None, run_manager: Any = None, **kwargs: Any) -> Any:
        self.seen.append([m for m in messages if not isinstance(m, SystemMessage)])
        return super()._generate(messages, stop, run_manager, **kwargs)


def _mark_block(system: str) -> str:
    start = system.index(MEMORY_TAG)
    end = system.index("[/MARK memory]") + len("[/MARK memory]")
    return system[start:end]


def _tool_pairs_valid(messages: list[Any]) -> bool:
    open_ids: set[str] = set()
    for message in messages:
        if isinstance(message, AIMessage) and message.tool_calls:
            open_ids = {call["id"] for call in message.tool_calls}
        elif isinstance(message, ToolMessage):
            if message.tool_call_id not in open_ids:
                return False
    return True


# ── budget ────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("mode", ["sync", "async"])
def test_budget_caps_the_whole_injection(mode: str) -> None:
    memories = {f"k{i}": f"MEMORY-{i}: " + "detail " * 30 for i in range(6)}
    backend = KeywordBackend(memories)
    mw = MarkAgentMiddleware(backend, include_skill=False, budget_tokens=150)
    model, agent = _agent(mw)
    _invoke(agent, "tell me about " + " ".join(memories), mode)
    assert model.calls
    for _, system in model.calls:
        block = _mark_block(system)
        assert estimate_tokens(block) <= 150
        assert "MEMORY-0" in block            # highest-ranked record kept first
        assert "MEMORY-5" not in block         # did not fit
    usage = [e for e in mw.diagnostics if e["kind"] == "budget"]
    assert usage and all(e["used_tokens"] <= 150 and e["dropped"] > 0 for e in usage)


def test_no_budget_keeps_char_truncation_behaviour() -> None:
    backend = KeywordBackend({"health": "Health checks live at /api/v1/health."})
    mw = MarkAgentMiddleware(backend, include_skill=False)
    assert mw._budget is None and mw._history == "full"
    model, agent = _agent(mw)
    _invoke(agent, "health check", "sync")
    assert all("<memory_context" in system for _, system in model.calls)
    assert not [e for e in mw.diagnostics if e["kind"] in ("budget", "history_window")]


def test_budget_combines_briefing_and_retrieval(tmp_path) -> None:
    from mark import Mark

    with Mark.local(project_path=tmp_path) as mark:
        memory = mark.runtime.memory("__mark__")
        memory.store_sync("[2026-03-01] atlas-core listens on port 6430 for the search cluster.")
        memory.facts.set("atlas-core", "port", "6430", valid_from="2026-03-01")
        memory.facts.set("atlas-core", "port", "4444", valid_from="2026-03-30")
        mw = MarkAgentMiddleware(LocalMarkBackend(mark), include_skill=False, write_outcomes=False,
                                 budget_tokens=120, token_counter=lambda s: len(s.split()))
        state = {"messages": [HumanMessage("What port does atlas-core use?"),
                              AIMessage("atlas-core listens on 6430."),
                              HumanMessage("Restart atlas-core please.")]}
        context = mw.before_model(state, runtime=None)["mark_context"]
        block = mw._memory_block(context)
    assert len(block.split()) <= 120
    assert "<mark_briefing>" in context and "4444" in context
    assert mw._run(state).last_usage.max_tokens == 120


# ── history="mark" ────────────────────────────────────────────────────────


@pytest.mark.parametrize("mode", ["sync", "async"])
def test_history_mark_windows_and_stores_dropped_turns(mode: str) -> None:
    backend = KeywordBackend({"hive": "User said: my hive is called Colmeia Azul."})
    mw = MarkAgentMiddleware(backend, include_skill=False, history="mark", keep_last=2)
    model = RecordingModel()
    _, agent = _agent(mw, model=model, checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": f"t-{mode}"}}
    for text in ["first question", "second question", "where is my hive?"]:
        _invoke(agent, text, mode, config=config)

    for messages in model.seen:
        assert _tool_pairs_valid(messages)
        assert any(isinstance(m, HumanMessage) for m in messages)   # the current question stays visible
        assert len(messages) <= 3
    last = model.seen[-1]
    assert [content_text(m.content) for m in last if isinstance(m, HumanMessage)] == ["where is my hive?"]
    # Everything dropped was stored first, verbatim, as a turn.
    assert "User said: first question" in backend.observed
    assert "User said: second question" in backend.observed
    assert any(o.startswith("Tool [lookup] returned: lookup result for first") for o in backend.observed)
    assert any(e["kind"] == "history_window" for e in mw.diagnostics)
    _, system = model.calls[-1]
    assert "Colmeia Azul" in system          # MARK stands in for the dropped history


def test_dropped_messages_are_archived_once() -> None:
    backend = KeywordBackend()
    mw = MarkAgentMiddleware(backend, include_skill=False, anticipate=False, history="mark", keep_last=1)
    state = {"messages": [HumanMessage("one", id="1"), AIMessage("two", id="2"), HumanMessage("three", id="3")]}
    mw.before_model(state, runtime=None)
    mw.before_model(state, runtime=None)
    assert backend.observed.count("User said: one") == 1
    assert backend.observed.count("Assistant replied: two") == 1


class _FailingObserve(KeywordBackend):
    async def observe(self, content: str, **options: Any) -> BackendResult:
        return BackendResult(False, error="disk full")


def test_history_is_not_dropped_when_storing_fails() -> None:
    mw = MarkAgentMiddleware(_FailingObserve(), include_skill=False, history="mark", keep_last=1)
    messages = [HumanMessage("old"), AIMessage("old answer"), HumanMessage("now")]
    seen: list[Any] = []

    class Request:
        def __init__(self, msgs: list[Any]) -> None:
            self.messages, self.state, self.system_message = msgs, {}, None

        def override(self, **kw: Any) -> "Request":
            return Request(kw.get("messages", self.messages))

    mw.wrap_model_call(Request(messages), seen.append)
    assert seen[0].messages == messages
    assert any(e["kind"] == "history_full" for e in mw.diagnostics)


@pytest.mark.parametrize("mode", ["sync", "async"])
def test_wrap_model_call_archives_before_dropping(mode: str) -> None:
    import asyncio

    backend = KeywordBackend()
    mw = MarkAgentMiddleware(backend, include_skill=False, history="mark", keep_last=2)
    messages = [HumanMessage("q1"), AIMessage("", tool_calls=[{"name": "lookup", "args": {}, "id": "c"}]),
                ToolMessage("tool output", tool_call_id="c"), AIMessage("a1"), HumanMessage("q2"),
                AIMessage("", tool_calls=[{"name": "lookup", "args": {}, "id": "d"}]),
                ToolMessage("second output", tool_call_id="d")]
    seen: list[Any] = []

    class Request:
        def __init__(self, msgs: list[Any]) -> None:
            self.messages, self.state, self.system_message = msgs, {}, None

        def override(self, **kw: Any) -> "Request":
            return Request(kw.get("messages", self.messages))

    if mode == "sync":
        mw.wrap_model_call(Request(messages), seen.append)
    else:
        async def handler(request: Any) -> None:
            seen.append(request)
        asyncio.run(mw.awrap_model_call(Request(messages), handler))
    kept = seen[0].messages
    assert kept == messages[4:]                      # q2 + its tool call + result, never split
    assert _tool_pairs_valid(kept)
    # The tool-call-only message has no words of its own: nothing stored for it (a10 N3).
    assert backend.observed == ["User said: q1", "Tool [tool] returned: tool output",
                                "Assistant replied: a1"]


def test_invalid_options() -> None:
    with pytest.raises(ValueError):
        MarkAgentMiddleware(KeywordBackend(), history="none")
    with pytest.raises(ValueError):
        MarkAgentMiddleware(KeywordBackend(), keep_last=0)
