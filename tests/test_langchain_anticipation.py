# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 Somtochukwu Owen Ikeanyi
"""End-to-end: MarkAgentMiddleware anticipates context through a real LangChain agent.

The scripted model never calls a memory tool. MARK must still put the right
facts in front of it (via ContextWatch), and correct the stale value the model
states in its first step before the second model call.
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("langchain")

from langchain.agents import create_agent
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool
from pydantic import Field

from mark import Mark
from mark.adapters.backend import LocalMarkBackend
from mark.adapters.langchain.middleware import MarkAgentMiddleware, content_text

LOG = [
    "[2026-03-01] INVENTORY: atlas-core is owned by Liam O'Connor, listens on port 6430.",
    "[2026-03-17] CHANGE: atlas-core moved from port 6430 to port 9239.",
    "[2026-03-30] CHANGE: atlas-core moved from port 9239 to port 4444.",
    "[2026-03-12] TICKET HAL-4433 opened (SEV1): intermittent 502 errors — affects atlas-core, assigned to Mei Lin.",
]


class ScriptedModel(BaseChatModel):
    """Step 1: states a stale port and calls a tool. Step 2: answers. Records system prompts."""

    systems: list[str] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "scripted-fake"

    def bind_tools(self, tools: Any, **kwargs: Any) -> "ScriptedModel":
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.systems.append("\n".join(content_text(m.content) for m in messages if isinstance(m, SystemMessage)))
        if isinstance(messages[-1], ToolMessage):
            reply = AIMessage(content="Done: atlas-core restarted.")
        else:
            reply = AIMessage(
                content="I'll restart atlas-core on port 9239.",
                tool_calls=[{"name": "restart", "args": {"service": "atlas-core"}, "id": "c1", "type": "tool_call"}],
            )
        return ChatResult(generations=[ChatGeneration(message=reply)])


@tool
def restart(service: str) -> str:
    """Restart a service."""
    return f"{service} restarted"


@pytest.fixture
def mark(tmp_path: Path, monkeypatch):
    # Deterministic under load: the first briefing waits for the watch's own
    # prefetch (cold_wait), not for whatever a wall-clock bound allows.
    from mark.context.watch import ContextWatch

    monkeypatch.setattr(ContextWatch, "cold_wait", 60.0)
    with Mark.local(tmp_path) as instance:
        memory = instance.runtime.memory("__mark__")
        memory.store_many_sync(LOG)
        for value, day in [("6430", "2026-03-01"), ("9239", "2026-03-17"), ("4444", "2026-03-30")]:
            memory.facts.set("atlas-core", "port", value, valid_from=day)
        yield instance


def _run(mark: Any, mode: str, **options: Any) -> ScriptedModel:
    model = ScriptedModel()
    middleware = MarkAgentMiddleware(LocalMarkBackend(mark), include_skill=False, **options)
    agent = create_agent(model=model, tools=[restart], middleware=[middleware])
    payload = {"messages": [HumanMessage(content="Please look into ticket HAL-4433.")]}
    if mode == "sync":
        agent.invoke(payload)
    else:
        asyncio.run(agent.ainvoke(payload))
    return model


@pytest.mark.parametrize("mode", ["sync", "async"])
def test_context_is_anticipated_and_stale_claims_corrected(mark, mode: str) -> None:
    model = _run(mark, mode)

    first, second = model.systems
    # First call: the ticket, plus the linked service's current port, without any memory tool.
    assert "<mark_briefing>" in first
    assert "TICKET HAL-4433 opened" in first
    assert "atlas-core port = 4444" in first
    # Second call: the stale value the model stated is corrected.
    assert "9239 is outdated" in second
    assert second.index("Corrections") < second.index("TICKET HAL-4433")


@pytest.mark.parametrize("mode", ["sync", "async"])
def test_anticipation_can_be_disabled(mark, mode: str) -> None:
    model = _run(mark, mode, anticipate=False)
    assert all("<mark_briefing>" not in system for system in model.systems)


def test_backends_without_memory_access_skip_anticipation() -> None:
    class PlainBackend:
        async def retrieve(self, query: str, **_: Any) -> Any:
            from mark.adapters.backend import BackendResult

            return BackendResult(ok=True, value=None)

    middleware = MarkAgentMiddleware(PlainBackend(), include_skill=False)
    assert middleware._watch_for(middleware._run({})) is None
