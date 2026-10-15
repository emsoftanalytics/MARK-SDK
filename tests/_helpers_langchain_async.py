"""Shared fakes for the MarkAgentMiddleware create_agent tests.

A small in-process fake chat model (with ``bind_tools`` support) and
fake/local backends — no network, no API keys.
"""
from __future__ import annotations

import asyncio
import time
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
from pydantic import Field

from mark.adapters.backend import BackendResult
from mark.adapters.langchain.middleware import MarkAgentMiddleware, content_text
from mark.types.gap import GapReport, GapSeverity


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


def _agent(mw: MarkAgentMiddleware, model: ToolCallingFakeModel | None = None, **kwargs: Any) -> Any:
    model = model or ToolCallingFakeModel()
    return model, create_agent(model=model, tools=[lookup], middleware=[mw], **kwargs)


def _invoke(agent: Any, text: str, mode: str, **kwargs: Any) -> dict[str, Any]:
    payload = {"messages": [HumanMessage(content=text)]}
    if mode == "sync":
        return agent.invoke(payload, **kwargs)
    return asyncio.run(agent.ainvoke(payload, **kwargs))


# ── 6 + 9: timeouts work and the loop is never blocked ────────────────────


class _BlockingMemory:
    def __init__(self, seconds: float) -> None:
        self.seconds = seconds

    def retrieve_sync(self, query: str, **options: Any) -> Any:
        time.sleep(self.seconds)
        return _Result(query, [_Fragment("f", "slow memory")])
