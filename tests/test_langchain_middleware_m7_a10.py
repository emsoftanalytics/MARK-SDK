# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 Somtochukwu Owen Ikeanyi
"""Offline replay of usage test M7 (a10 findings N3/N4) with a scripted fake model.

The real scenario (``integ/lc/agent_test.py``) ran llama3.2:1b through
``create_agent`` with ``MarkAgentMiddleware(history="mark", keep_last=4,
budget_tokens=300)``. Here a deterministic fake model drives the same agent
over a real local store, so the assertions are about what MARK *stores* and
*injects*, not about model quality:

* each assistant reply is stored exactly once (low trust), with no
  "Agent reasoning" / "Agent completed task" copies and nothing for
  tool-call-only messages;
* tool results are ``source="tool"`` + ``mark:tool``; tool errors also carry
  ``mark:error`` and low trust;
* records tagged ``mark:question`` never appear in the injected block;
* everything MARK adds to the system prompt (wrapper included) fits
  ``budget_tokens`` under the default estimator.
"""
from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("langchain")
pytest.importorskip("langgraph")

from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import Field

from mark.adapters.backend import LocalMarkBackend
from mark.adapters.langchain.middleware import MarkAgentMiddleware, content_text
from mark.context.budget import estimate_tokens

PROMPT = "You are a concise warehouse assistant. Use check_inventory for stock questions."
BUDGET = 300
INVENTORY = {"widget": 42, "gadget": 7}
TURNS = [
    "Hi, I'm Priya and I manage the Leeds warehouse.",
    "Our main supplier is Acme Metals; contract renews on 2026-12-01.",
    "How many widgets do we have? Use the tool.",
    "How many sprockets are in stock? Use the tool.",
    "Who is our main supplier?",
    "Who is our main supplier and when does the contract renew?",
    "What's my name and which warehouse do I manage?",
]


@tool
def check_inventory(item: str) -> str:
    """Return the number of units in stock for an item (widget, gadget, sprocket)."""
    return f"{item}: {INVENTORY.get(item.lower().strip(), 'unknown item')} units in stock"


class ScriptedModel(BaseChatModel):
    """Stock questions -> one tool call (sprockets: invalid args -> tool error);
    after a tool result or for any other turn -> a plain text reply."""

    systems: list[str] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "scripted-m7"

    def bind_tools(self, tools: Any, **kwargs: Any) -> "ScriptedModel":
        return self

    def _generate(self, messages: Any, stop: Any = None, run_manager: Any = None, **kw: Any) -> ChatResult:
        self.systems.append("\n".join(content_text(m.content) for m in messages
                                      if isinstance(m, SystemMessage)))
        human = content_text([m for m in messages if isinstance(m, HumanMessage)][-1].content)
        last = messages[-1]
        if isinstance(last, ToolMessage):
            reply = AIMessage(f"Stock check done: {content_text(last.content)[:60]}")
        elif "Use the tool" in human:
            args = {"item": "widget"} if "widget" in human else {"name": "sprocket"}  # bad args
            reply = AIMessage("", tool_calls=[{"name": "check_inventory", "args": args,
                                               "id": f"call-{len(self.systems)}", "type": "tool_call"}])
        else:
            reply = AIMessage(f"Reply {len(self.systems)}: noted about {human[:40]}")
        return ChatResult(generations=[ChatGeneration(message=reply)])


class RecordingBackend(LocalMarkBackend):
    """Real local backend that also records every observe() call."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.stored: list[tuple[str, dict[str, Any]]] = []

    def observe_sync(self, content: str, **options: Any) -> Any:
        self.stored.append((content, dict(options)))
        return super().observe_sync(content, **options)


def _run(tmp_path: Any, mode: str) -> tuple[RecordingBackend, ScriptedModel, list[Any]]:
    import asyncio

    from mark import Mark

    mark = Mark.local(project_path=tmp_path)
    backend = RecordingBackend(mark)
    mw = MarkAgentMiddleware(backend, agent_id="wh", anticipate=True, budget_tokens=BUDGET,
                             history="mark", keep_last=4)
    model = ScriptedModel()
    agent = create_agent(model, [check_inventory], middleware=[mw], checkpointer=InMemorySaver(),
                         system_prompt=PROMPT)
    config = {"configurable": {"thread_id": f"m7-{mode}"}}
    out: dict[str, Any] = {}
    try:
        for text in TURNS:
            payload = {"messages": [HumanMessage(text)]}
            out = (agent.invoke(payload, config) if mode == "sync"
                   else asyncio.run(agent.ainvoke(payload, config)))
    finally:
        mark.runtime.shutdown()
    return backend, model, list(out["messages"])


@pytest.fixture(scope="module", params=["sync", "async"])
def m7(request: Any, tmp_path_factory: Any) -> tuple[RecordingBackend, ScriptedModel, list[Any]]:
    return _run(tmp_path_factory.mktemp(f"m7-{request.param}"), request.param)


def test_each_assistant_reply_is_stored_once(m7: Any) -> None:
    backend, _, transcript = m7
    contents = [c for c, _ in backend.stored]
    replies = [content_text(m.content) for m in transcript if isinstance(m, AIMessage) and m.content]
    assert replies
    for reply in replies[:-1]:     # the last reply of the last run is stored at its run end too
        matches = [c for c, o in backend.stored if reply in c]
        assert len(matches) == 1, (reply, matches)
        options = dict(backend.stored)[matches[0]]
        assert options["source"] == "assistant" and "mark:assistant" in options["tags"]
        assert options["importance"] < 0.5
    assert len(contents) == len(set(contents)), "something was stored twice"
    assert not [c for c in contents if c.startswith(("Agent reasoning", "Agent completed task"))]
    assert not [c for c in contents if "[called" in c or c.strip() == "Assistant replied:"]


def test_tool_results_and_errors_carry_tool_provenance(m7: Any) -> None:
    backend, _, _ = m7
    tools = [(c, o) for c, o in backend.stored if o.get("source") == "tool"]
    errors = [(c, o) for c, o in tools if "mark:error" in o["tags"]]
    normal = [(c, o) for c, o in tools if "mark:error" not in o["tags"]]
    assert errors and normal
    assert all("mark:tool" in o["tags"] for _, o in tools)
    assert any("widget: 42" in c for c, _ in normal)
    for _, options in errors:
        assert options["importance"] < 0.5 and options["metadata"]["trust"] == "low"


def test_questions_are_never_injected_and_block_fits_budget(m7: Any) -> None:
    backend, model, _ = m7
    asks = ("mark:question", "mark:request")
    questions = [c.split(": ", 1)[-1] for c, o in backend.stored
                 if "mark:question" in o.get("tags", ())]
    requests = [c.split(": ", 1)[-1] for c, o in backend.stored
                if any(tag in o.get("tags", ()) for tag in asks)]
    assert questions, "pure questions must be stored with mark:question"
    injected = [system[len(PROMPT):] for system in model.systems]
    assert any("[MARK memory]" in text for text in injected)
    for text in injected:
        assert "project memory" not in text
        assert estimate_tokens(text) <= BUDGET, text
        notes = text.split("[MARK memory]", 1)[-1]
        for asked in requests:
            assert asked not in notes, asked
