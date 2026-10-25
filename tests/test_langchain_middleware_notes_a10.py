# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 Somtochukwu Owen Ikeanyi
"""MarkAgentMiddleware a10 fixes (N3/N4), unit level:

* a tool exception is stored once as a low-trust ``mark:tool`` + ``mark:error``
  memory and never reaches the ContextWatch;
* the middleware's own rendering never shows questions, requests, assistant
  replies or tool errors as notes (non-budget path and MarkMemoryMiddleware);
* ``MarkMemoryMiddleware.remember`` tags questions; its block fits
  ``max_context_chars`` wrapper included;
* the whole text MARK adds (wrapper + separator) fits ``budget_tokens``, and
  a budget too small for the wrapper plus one line injects nothing.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace as NS
from typing import Any

import pytest

pytest.importorskip("langchain")
pytest.importorskip("langgraph")

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from mark.adapters.backend import BackendResult
from mark.adapters.langchain.middleware import MarkAgentMiddleware, MarkMemoryMiddleware
from mark.context.budget import estimate_tokens

from _helpers_langchain_async import KeywordBackend, _agent, _invoke


class OptionsBackend(KeywordBackend):
    """Records observe() options; retrieve() returns tagged fragments."""

    def __init__(self, fragments: list[Any] | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.options: dict[str, dict[str, Any]] = {}
        self.fragments = fragments or []

    async def observe(self, content: str, **options: Any) -> BackendResult:
        self.options[content] = options
        return await super().observe(content, **options)

    async def retrieve(self, query: str, **options: Any) -> BackendResult:
        if not self.fragments:
            return await super().retrieve(query, **options)
        return BackendResult(True, _Value(query, list(self.fragments)))


class _Value:
    """Retrieval-result double: renders whatever fragments it holds."""

    def __init__(self, query: str, fragments: list[Any]) -> None:
        self.query, self.fragments, self.facts, self.gap_report = query, fragments, [], None
        self.scores = [1.0] * len(fragments)

    def as_context(self) -> str:
        return "\n".join(f.content for f in self.fragments)


def _frag(fid: str, content: str, *tags: str, source: str = "user") -> NS:
    return NS(id=fid, content=content, tags=list(tags), source=source)


MIXED = [
    _frag("s", "User said: Our supplier is Acme Metals.", "mark:user"),
    _frag("q", "User said: Who is our supplier?", "mark:question"),
    _frag("r", "User said: Check gadget stock too.", "mark:request"),
    _frag("a", "Assistant replied: Your supplier is Bolt Ltd.", "mark:assistant", source="assistant"),
    _frag("e", "Tool [check] failed: boom", "mark:tool", "mark:error", source="tool"),
]


class _Request:
    def __init__(self, name: str = "check", state: Any = None) -> None:
        self.tool_call = {"name": name, "args": {}, "id": "c1"}
        self.state = state or {}


@pytest.mark.parametrize("mode", ["sync", "async"])
def test_tool_exception_is_stored_as_low_trust_error(mode: str) -> None:
    backend = OptionsBackend()
    mw = MarkAgentMiddleware(backend, include_skill=False)

    def boom(request: Any) -> Any:
        raise RuntimeError("inventory service unreachable")

    async def aboom(request: Any) -> Any:
        boom(request)

    with pytest.raises(RuntimeError):
        if mode == "sync":
            mw.wrap_tool_call(_Request(), boom)
        else:
            asyncio.run(mw.awrap_tool_call(_Request(), aboom))
    [(content, options)] = backend.options.items()
    assert content == "Tool [check] failed: RuntimeError: inventory service unreachable"
    assert options["source"] == "tool" and {"mark:tool", "mark:error"} <= set(options["tags"])
    assert options["importance"] < 0.5 and options["metadata"] == {"trust": "low"}


def test_error_tool_message_is_stored_once_and_not_watched() -> None:
    backend = OptionsBackend()
    mw = MarkAgentMiddleware(backend, include_skill=False, history="mark", keep_last=1)
    error = ToolMessage("Error: item is a required field", tool_call_id="c1", name="check",
                        status="error")
    mw.wrap_tool_call(_Request(), lambda request: error)
    asyncio.run(mw._flush_observations_async(mw._run({})))
    # History archival of the same message (now with an id) must not store it again.
    stored = ToolMessage(error.content, tool_call_id="c1", name="check", status="error", id="x")
    state = {"messages": [HumanMessage("q", id="1"),
                          AIMessage("", id="2", tool_calls=[{"name": "check", "args": {}, "id": "c1"}]),
                          stored, AIMessage("Sorry, that failed.", id="3"), HumanMessage("next", id="4")]}
    mw.before_model(state, runtime=None)
    errors = [c for c in backend.options if c.startswith("Tool [check] failed")]
    assert errors == ["Tool [check] failed: Error: item is a required field"]
    assert "mark:error" in backend.options[errors[0]]["tags"]

    class Watch:
        seen: list[tuple[str, str]] = []

        def observe(self, text: str, role: str) -> None:
            self.seen.append((role, text))

        def observe_tool_call(self, name: str, args: Any) -> None:
            pass

    watch = Watch()
    mw._feed_watch(watch, NS(watch_seen=0), state)
    assert all("required field" not in text for _, text in watch.seen)


def test_rendering_skips_questions_requests_replies_and_errors() -> None:
    backend = OptionsBackend(MIXED)
    mw = MarkAgentMiddleware(backend, include_skill=False, anticipate=False, write_outcomes=False)
    context = mw.before_model({"messages": [HumanMessage("supplier details")]}, runtime=None)["mark_context"]
    assert "Acme Metals" in context
    for hidden in ("Who is our supplier?", "Check gadget", "Bolt Ltd", "boom"):
        assert hidden not in context
    assert list(mw._run({}).fragment_ids) == ["s"]       # feedback only for shown notes

    helper = MarkMemoryMiddleware(backend=OptionsBackend(MIXED), agent_id="a", max_context_chars=120)
    out = helper.augment_messages([{"role": "user", "content": "hi"}], "supplier")
    block = out[0]["content"]
    assert block.startswith("[MARK memory]") and "Acme" in block and "Bolt" not in block
    assert len(block) <= 120


def test_remember_tags_questions_and_roles() -> None:
    backend = OptionsBackend()
    helper = MarkMemoryMiddleware(backend=backend, agent_id="a", write_outcomes=True)
    helper.remember_sync("What is the staging database?")
    helper.remember_sync("Response: the staging database is db-7.", step=1)
    helper.remember_sync("db-7 is up", role="tool")
    assert "mark:question" in backend.options["What is the staging database?"]["tags"]
    assert backend.options["Response: the staging database is db-7."] == {"agent_id": "a", "metadata": {"step": 1}}
    assert "mark:tool" in backend.options["db-7 is up"]["tags"]


@pytest.mark.parametrize("budget", [60, 150])
def test_added_text_fits_the_budget_wrapper_included(budget: int) -> None:
    memories = {f"k{i}": f"MEMORY-{i}: " + "detail " * 30 for i in range(6)}
    mw = MarkAgentMiddleware(KeywordBackend(memories), include_skill=False, budget_tokens=budget)
    model, agent = _agent(mw)
    _invoke(agent, "tell me about " + " ".join(memories), "sync")
    for _, system in model.calls:              # no system prompt: all of it is MARK's
        assert "[MARK memory]" in system and estimate_tokens(system) <= budget


def test_budget_too_small_for_wrapper_injects_nothing() -> None:
    mw = MarkAgentMiddleware(KeywordBackend({"k": "MEMORY: " + "detail " * 30}),
                             include_skill=False, budget_tokens=15)
    model, agent = _agent(mw)
    _invoke(agent, "tell me about k", "sync")
    assert all(system == "" for _, system in model.calls)
