from __future__ import annotations

from types import SimpleNamespace
from typing import Any

try:
    import pytest
except ImportError:  # pragma: no cover - direct smoke execution without pytest
    pytest = None  # type: ignore[assignment]

if pytest is not None:
    pytest.importorskip("langchain")

from mark.adapters.backend import BackendResult
from mark.adapters.langchain.middleware import MarkAgentMiddleware


class _Context:
    def __init__(self, text: str) -> None:
        self._text = text

    def as_context(self) -> str:
        return self._text


class _Backend:
    def __init__(self, text: str = "FastAPI routes use /api/v1 and APIRouter.") -> None:
        self.text = text
        self.retrieve_calls: list[tuple[str, dict[str, Any]]] = []
        self.observed: list[str] = []
        self.feedback_payloads: list[dict[str, Any]] = []

    async def retrieve(self, query: str, **options: Any) -> BackendResult:
        self.retrieve_calls.append((query, options))
        return BackendResult(True, _Context(self.text))

    async def write(self, content: str, **options: Any) -> BackendResult:
        return BackendResult(True, content)

    async def observe(self, content: str, **options: Any) -> BackendResult:
        self.observed.append(content)
        return BackendResult(True, "fragment-id", {"fragment_id": "fragment-id"})

    async def compress(self, query: str, candidates: list[Any], **options: Any) -> BackendResult:
        return BackendResult(True, candidates)

    async def heal_gap(self, query: str, **options: Any) -> BackendResult:
        return await self.retrieve(query, **options)

    async def list_sessions(self, agent_id: str, **options: Any) -> BackendResult:
        return BackendResult(True, [])

    async def feedback(self, **payload: Any) -> BackendResult:
        self.feedback_payloads.append(payload)
        return BackendResult(True, payload)


def _msg(role: str, content: str, **extra: Any) -> SimpleNamespace:
    return SimpleNamespace(type=role, content=content, **extra)


def test_sync_before_model_retrieves_and_bounds_context() -> None:
    backend = _Backend("x" * 5000)
    mw = MarkAgentMiddleware(backend, agent_id="coder", max_context_chars=120)

    state = {"messages": [_msg("human", "Implement a FastAPI product router")]}

    update = mw.before_model(state, runtime=None)

    assert update is not None
    assert update["mark_context"].endswith("[MARK context truncated]")
    assert len(update["mark_context"]) <= 120
    assert backend.retrieve_calls
    assert backend.retrieve_calls[0][1]["agent_id"] == "coder"
    assert backend.retrieve_calls[0][1]["compress"] is False
    assert any(
        event["kind"] == "endpoint" and event["endpoint"] == "retrieve"
        for event in mw.diagnostics
    )


def test_tool_results_are_queued_until_session_end() -> None:
    backend = _Backend()
    mw = MarkAgentMiddleware(
        backend,
        agent_id="coder",
        observe_tool_results=True,
        immediate_tool_observe=False,
    )
    request = SimpleNamespace(tool_call={"name": "read_file"})

    result = mw.wrap_tool_call(
        request,
        lambda _request: SimpleNamespace(
            content="from fastapi import APIRouter\nrouter = APIRouter()\n" * 3
        ),
    )

    assert "APIRouter" in result.content
    assert backend.observed == []

    mw.after_agent(
        {"messages": [_msg("ai", "Implemented the Product CRUD router and registered it.")]},
        runtime=None,
    )

    assert any("Tool [read_file] returned" in item for item in backend.observed)
    assert backend.feedback_payloads


def test_write_tool_result_does_not_trigger_step_retrieval() -> None:
    backend = _Backend()
    mw = MarkAgentMiddleware(backend, agent_id="coder")
    human = _msg("human", "Implement a FastAPI product router")

    first = mw.before_model({"messages": [human]}, runtime=None)
    calls_after_first = len(backend.retrieve_calls)
    state = {
        "messages": [
            human,
            _msg("tool", "Wrote 1200 chars to 'routers/products.py'.", name="write_file"),
        ]
    }

    second = mw.before_model(state, runtime=None)

    assert first is not None
    assert second is not None
    assert len(backend.retrieve_calls) == calls_after_first


def test_wrap_model_call_logs_skill_quality() -> None:
    backend = _Backend()
    mw = MarkAgentMiddleware(backend, agent_id="coder")
    agent_state = {"messages": [_msg("human", "Implement a FastAPI product router")]}
    update = mw.before_model(agent_state, runtime=None)
    assert update is not None

    class _System:
        content = (
            "---\n"
            "name: mark-memory-middleware\n"
            "description: Use MARK middleware as transparent memory.\n"
            "---\n"
            "Recall, ground, act, remember. Do not store secrets."
        )
        content_blocks = [{"type": "text", "text": content}]

    class _Request:
        state = update
        messages = agent_state["messages"]
        system_message = _System()

        def override(self, **kwargs: Any) -> Any:
            clone = SimpleNamespace()
            clone.state = self.state
            clone.messages = self.messages
            clone.system_message = kwargs.get("system_message", self.system_message)
            return clone

    mw.wrap_model_call(_Request(), lambda request: request)

    skill_events = [event for event in mw.diagnostics if event["kind"] == "skill"]
    assert skill_events
    assert skill_events[-1]["present"] is True
    assert skill_events[-1]["score"] >= 0.85


def test_wrap_model_call_handles_missing_system_message() -> None:
    backend = _Backend()
    mw = MarkAgentMiddleware(backend, agent_id="coder")
    agent_state = {"messages": [_msg("human", "Implement a FastAPI product router")]}
    update = mw.before_model(agent_state, runtime=None)
    assert update is not None

    class _Request:
        state = update
        messages = agent_state["messages"]
        system_message = None

        def override(self, **kwargs: Any) -> Any:
            clone = SimpleNamespace()
            clone.state = self.state
            clone.messages = self.messages
            clone.system_message = kwargs.get("system_message", self.system_message)
            return clone

    result = mw.wrap_model_call(_Request(), lambda request: request)

    assert result.system_message is not None
    assert any(
        "MARK project memory" in str(block.get("text", ""))
        for block in result.system_message.content
    )


def test_sync_hooks_work_inside_running_event_loop() -> None:
    """Jupyter regression: ipykernel keeps a loop running in the main thread,
    so LangGraph's sync path invokes sync hooks while a loop is active.
    _run_sync must bridge to a worker thread instead of raising."""
    import asyncio

    backend = _Backend()
    mw = MarkAgentMiddleware(backend=backend, agent_id="nb-agent")
    human = _msg("human", "How are routes structured?")

    async def drive() -> dict[str, Any] | None:
        # Called synchronously while this coroutine's loop is running —
        # exactly what agent.invoke() does inside a notebook cell.
        mw.before_agent({"messages": [human]}, runtime=None)
        return mw.before_model({"messages": [human]}, runtime=None)

    update = asyncio.run(drive())
    assert backend.retrieve_calls, "retrieval should have run via the worker-thread bridge"
    assert update is not None
