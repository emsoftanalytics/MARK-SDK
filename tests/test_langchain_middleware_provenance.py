# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 MARK Contributors
"""MarkAgentMiddleware provenance and skill budgeting (usage test M7 regressions).

* windowed-out messages are stored with their role's provenance; assistant
  replies are low trust (or not stored with ``store_assistant=False``);
* the model's reasoning and final answers are archived as assistant output;
* the injected skill counts inside ``budget_tokens``; small budgets and
  presets get no skill unless one is named (the compact one is frontmatter-free);
* assistant output stored by earlier runs never anchors a briefing.
"""
from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("langchain")
pytest.importorskip("langgraph")

from langchain_core.messages import AIMessage, HumanMessage

from mark.adapters.backend import BackendResult, LocalMarkBackend
from mark.adapters.langchain.middleware import MarkAgentMiddleware
from mark.context.budget import estimate_tokens
from mark.context.provenance import ASSISTANT_TAG

from _helpers_langchain_async import KeywordBackend, _agent, _invoke


class OptionsBackend(KeywordBackend):
    """Keyword backend that also records the observe() options."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.options: dict[str, dict[str, Any]] = {}

    async def observe(self, content: str, **options: Any) -> BackendResult:
        self.options[content] = options
        return await super().observe(content, **options)


def test_archived_turns_carry_provenance() -> None:
    backend = OptionsBackend()
    mw = MarkAgentMiddleware(backend, include_skill=False, anticipate=False, history="mark", keep_last=1)
    state = {"messages": [HumanMessage("I manage the Leeds warehouse.", id="1"),
                          AIMessage("You are Mark-Middleware, the warehouse bot.", id="2"),
                          HumanMessage("Who am I?", id="3")]}
    mw.before_model(state, runtime=None)
    user = backend.options["User said: I manage the Leeds warehouse."]
    reply = backend.options["Assistant replied: You are Mark-Middleware, the warehouse bot."]
    assert user["source"] == "user" and "mark:user" in user["tags"]
    assert reply["source"] == "assistant" and ASSISTANT_TAG in reply["tags"]
    assert reply["importance"] < 0.5


def test_store_assistant_false_drops_replies_unstored() -> None:
    backend = OptionsBackend()
    mw = MarkAgentMiddleware(backend, include_skill=False, anticipate=False, history="mark",
                             keep_last=1, store_assistant=False)
    state = {"messages": [HumanMessage("one statement here", id="1"), AIMessage("two", id="2"),
                          HumanMessage("three", id="3")]}
    mw.before_model(state, runtime=None)
    assert backend.observed == ["User said: one statement here"]
    assert not mw._unarchived(state["messages"])      # the reply may now leave the window


@pytest.mark.parametrize("mode", ["sync", "async"])
def test_run_observations_are_low_trust_assistant_output(mode: str) -> None:
    backend = OptionsBackend({"lookup": "lookup notes"})
    mw = MarkAgentMiddleware(backend, include_skill=False)
    _, agent = _agent(mw)
    _invoke(agent, "please lookup the inventory status for widgets", mode)
    completed = [c for c in backend.options if c.startswith("Agent completed task.")]
    tools = [c for c in backend.options if c.startswith("Tool [")]
    assert completed and all(backend.options[c]["source"] == "assistant" for c in completed)
    assert tools and all(backend.options[c]["source"] == "tool" for c in tools)


def test_skill_is_charged_to_the_budget_and_left_out_when_small() -> None:
    mw = MarkAgentMiddleware(KeywordBackend(), budget_tokens=300, include_skill="mark-memory-compact")
    skill = mw._skill_block()
    assert skill.startswith("# Your memory")                   # frontmatter-free
    assert mw._memory_budget().max_tokens == 300 - estimate_tokens(skill)
    large = MarkAgentMiddleware(KeywordBackend(), budget_tokens=4000)
    assert large._skill_name() == "mark-memory-middleware"     # long skill unchanged
    assert large._skill_block().startswith("---")
    assert large._memory_budget().max_tokens == 4000 - estimate_tokens(large._skill_block())
    # Small budgets and presets: no skill (it lowered small-model accuracy).
    assert MarkAgentMiddleware(KeywordBackend(), budget_tokens=300)._skill_name() is None
    assert MarkAgentMiddleware(KeywordBackend(), preset="tiny")._skill_name() is None
    assert MarkAgentMiddleware(KeywordBackend(), budget_tokens=300)._memory_budget().max_tokens == 300
    assert MarkAgentMiddleware(KeywordBackend(), include_skill=False)._memory_budget() is None
    with pytest.raises(ValueError):
        MarkAgentMiddleware(KeywordBackend(), preset="huge")


@pytest.mark.parametrize("mode", ["sync", "async"])
def test_system_prompt_mark_text_fits_the_budget(mode: str) -> None:
    memories = {f"k{i}": f"MEMORY-{i}: " + "detail " * 30 for i in range(6)}
    mw = MarkAgentMiddleware(KeywordBackend(memories), budget_tokens=300,
                             include_skill="mark-memory-compact")
    model, agent = _agent(mw)
    _invoke(agent, "tell me about " + " ".join(memories), mode)
    for _, system in model.calls:
        assert "# Your memory" in system and "name: mark-memory" not in system
        mark_text = system[system.index("# Your memory"):]
        assert estimate_tokens(mark_text) <= 300 + 2      # skill + block, joined by a newline


def test_stored_assistant_text_never_anchors_the_briefing(tmp_path) -> None:
    from mark import Mark

    with Mark.local(project_path=tmp_path) as mark:
        memory = mark.runtime.memory("wh")
        memory.store_sync("User said: I manage the Leeds warehouse.", source="user", tags=["mark:user"])
        memory.store_sync('Assistant replied: {"name": "mark_memory_middleware", "error_code": "E001"}',
                          source="assistant", tags=[ASSISTANT_TAG], importance=0.2, confidence=0.3)
        mw = MarkAgentMiddleware(LocalMarkBackend(mark), agent_id="wh", include_skill=False,
                                 write_outcomes=False, budget_tokens=300)
        state = {"messages": [HumanMessage("What does mark_memory_middleware E001 mean for Leeds?")]}
        context = mw.before_model(state, runtime=None)["mark_context"]
        watch = mw._run(state).watch
        anchors = watch.briefing().anchors if watch is not None else []
    assert "E001" not in anchors and "mark_memory_middleware" not in anchors
    assert "error_code" not in context.split("Earlier assistant replies (unverified):")[0]
