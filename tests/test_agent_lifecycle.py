"""Lifecycle tests for MarkAgent, plugin registration, adapter backends and MCP."""
from __future__ import annotations

import asyncio
import time
from typing import Any, Callable

import pytest

from mark import AgentRun, BaseMiddleware, Mark, MiddlewareContext
from mark.adapters.backend import LocalMarkBackend
from mark.plugins import MarkCorePlugin, PluginRegistry


class _CountingMiddleware(BaseMiddleware):
    def __init__(self, name: str) -> None:
        self.name = name
        self.bind_calls = 0
        self.before_calls = 0

    def bind(self, runtime: Any) -> None:
        self.bind_calls += 1

    def before(self, context: MiddlewareContext) -> None:
        self.before_calls += 1


async def _llm(prompt: str) -> str:
    return "ok"


# ── MarkAgent ─────────────────────────────────────────────────────────────


def test_use_does_not_mutate_runtime_stack(tmp_path: Any) -> None:
    runtime_mw = _CountingMiddleware("runtime-level")
    with Mark.local(tmp_path, middleware=[runtime_mw]) as mark:
        runtime_stack = mark.runtime.middleware
        before = runtime_stack.list()

        agent = mark.wrap_agent(_llm)  # middleware=None → runtime stack
        extra = _CountingMiddleware("agent-only")
        agent.use(extra)

        assert runtime_stack.list() == before
        assert extra not in runtime_stack.list()
        other = mark.wrap_agent(_llm)
        asyncio.run(other.run("hello"))
        assert extra.before_calls == 0

        asyncio.run(agent.run("hello"))
        assert extra.before_calls == 1
        assert runtime_mw.before_calls >= 1


def test_runtime_middleware_bound_once_when_agent_adds_middleware(tmp_path: Any) -> None:
    runtime_mw = _CountingMiddleware("runtime-level")
    with Mark.local(tmp_path, middleware=[runtime_mw]) as mark:
        assert runtime_mw.bind_calls == 1
        agent_mw = _CountingMiddleware("agent-level")
        agent = mark.wrap_agent(_llm, middleware=[agent_mw])
        assert runtime_mw.bind_calls == 1
        assert agent_mw.bind_calls == 1

        later = _CountingMiddleware("later")
        agent.use(later)
        assert later.bind_calls == 1
        assert runtime_mw.bind_calls == 1


def test_run_does_not_mutate_reusable_agent_run(tmp_path: Any) -> None:
    with Mark.local(tmp_path) as mark:
        agent = mark.wrap_agent(_llm, blocks=["project"])
        template = AgentRun(prompt="hello", metadata={"source": "template"})

        result = asyncio.run(agent.run(template, top_k=3, metadata={"override": True}))

        assert result.output == "ok"
        assert template.top_k is None
        assert template.metadata == {"source": "template"}
        assert template.blocks is None
        assert template.policy is None


def test_middleware_payload_edits_do_not_leak_into_template(tmp_path: Any) -> None:
    class Tagger(BaseMiddleware):
        name = "tagger"

        def before(self, context: MiddlewareContext) -> None:
            context.payload["request"].metadata["tagged"] = True

    with Mark.local(tmp_path) as mark:
        agent = mark.wrap_agent(_llm, middleware=[Tagger()])
        template = AgentRun(prompt="hello")
        asyncio.run(agent.run(template))
        asyncio.run(agent.run(template))
        assert template.metadata == {}


def test_sync_llm_does_not_block_event_loop(tmp_path: Any) -> None:
    def slow_llm(prompt: str) -> str:
        time.sleep(0.3)
        return "slow ok"

    with Mark.local(tmp_path) as mark:
        agent = mark.wrap_agent(slow_llm)

        async def drive() -> tuple[int, str]:
            ticks = 0
            done = asyncio.Event()

            async def heartbeat() -> None:
                nonlocal ticks
                while not done.is_set():
                    ticks += 1
                    await asyncio.sleep(0.01)

            beat = asyncio.create_task(heartbeat())
            result = await agent.run("hello")
            done.set()
            await beat
            return ticks, result.output

        ticks, output = asyncio.run(drive())
        assert output == "slow ok"
        assert ticks >= 10


# ── PluginRegistry ────────────────────────────────────────────────────────


class _Plugin(MarkCorePlugin):
    def __init__(self, name: str, hooks: set[str], fail: bool = False) -> None:
        self._name = name
        self._hooks = hooks
        self._fail = fail

    @property
    def name(self) -> str:
        return self._name

    @property
    def version(self) -> str:
        return "1.0"

    @property
    def provides(self) -> set[str]:
        return self._hooks

    def get_hook(self, hook_name: str) -> Callable[..., Any]:
        return lambda *a, **k: hook_name

    def on_register(self, runtime: Any) -> None:
        if self._fail:
            raise RuntimeError("on_register failed")


def test_plugin_registration_rolls_back_when_on_register_fails() -> None:
    registry = PluginRegistry()
    with pytest.raises(RuntimeError, match="on_register failed"):
        registry.register(_Plugin("bad", {"scoring", "critic"}, fail=True), runtime=object())

    assert registry.list_plugins() == []
    assert not registry.has("scoring")
    assert not registry.has("critic")
    # The hooks are free again for a healthy plugin with the same name.
    registry.register(_Plugin("bad", {"scoring"}), runtime=object())
    assert registry.has("scoring")


# ── LocalMarkBackend.feedback ─────────────────────────────────────────────


def test_local_backend_feedback_forwards_fragment_scores() -> None:
    calls: list[tuple[str, float]] = []

    class _Memory:
        def feedback(self, record_id: str, score: float) -> None:
            calls.append((record_id, score))

    class _Mark:
        memory = _Memory()

    backend = LocalMarkBackend(_Mark())
    result = asyncio.run(backend.feedback(
        agent_id="__mark__", outcome="completed", summary="done", fragment_ids=["a", "b"],
    ))
    assert result.ok
    assert result.metadata == {"applied": 2, "score": 1.0}
    assert calls == [("a", 1.0), ("b", 1.0)]

    calls.clear()
    asyncio.run(backend.feedback(fragment_id="c", score=-3))
    assert calls == [("c", -1.0)]

    skipped = asyncio.run(backend.feedback(outcome="completed", summary="no ids"))
    assert skipped.ok and skipped.metadata == {"skipped": "no_fragment_ids"}


def test_local_backend_feedback_against_real_mark(tmp_path: Any) -> None:
    with Mark.local(tmp_path) as mark:
        result = asyncio.run(LocalMarkBackend(mark).feedback(fragment_ids=["x"], outcome="failed"))
        assert result.ok
        assert result.metadata == {"applied": 1, "score": -1.0}


# ── MCP adapter ───────────────────────────────────────────────────────────


def _mcp_server(tmp_path: Any, **kwargs: Any) -> tuple[Any, Any]:
    pytest.importorskip("mcp", reason="mcp extra not installed")
    from mark.adapters.mcp import create_mark_mcp_server

    mark = Mark.local(tmp_path)
    return mark, create_mark_mcp_server(LocalMarkBackend(mark), **kwargs)


def test_mcp_importance_schema_is_bounded(tmp_path: Any) -> None:
    mark, server = _mcp_server(tmp_path)
    try:
        async def drive() -> tuple[dict[str, Any], Exception | None]:
            tools = {t.name: t for t in await server.list_tools()}
            tool = tools["mark_observe"]
            schema = getattr(tool, "input_schema", None) or getattr(tool, "inputSchema")
            error: Exception | None = None
            try:
                await server.call_tool(
                    "mark_observe", {"content": "an observation that is long", "importance": 3.0}
                )
            except Exception as exc:  # mcp raises a ToolError on validation failure
                error = exc
            return schema["properties"]["importance"], error

        importance, error = asyncio.run(drive())
        assert importance.get("minimum") == 0.0
        assert importance.get("maximum") == 1.0
        assert error is not None
    finally:
        mark.shutdown()


def test_mcp_allowed_agent_ids_rejects_other_namespaces(tmp_path: Any) -> None:
    mark, server = _mcp_server(tmp_path, allowed_agent_ids={"team-a"})
    try:
        async def drive() -> tuple[str, str, str]:
            denied = await server.call_tool("mark_write", {"content": "secret plan", "agent_id": "team-b"})
            allowed = await server.call_tool("mark_write", {"content": "team a fact", "agent_id": "team-a"})
            default = await server.call_tool("mark_write", {"content": "default namespace fact"})
            return str(denied), str(allowed), str(default)

        denied, allowed, default = asyncio.run(drive())
        assert "not allowed" in denied
        assert "written" in allowed
        assert "written" in default
    finally:
        mark.shutdown()


def test_mcp_tool_does_not_block_event_loop() -> None:
    pytest.importorskip("mcp", reason="mcp extra not installed")
    from types import SimpleNamespace

    from mark.adapters.mcp import create_mark_mcp_server

    class _SlowMemory:
        def retrieve_sync(self, query: str, **options: Any) -> str:
            time.sleep(0.3)
            return "slow memory"

    memory = _SlowMemory()
    fake_mark = SimpleNamespace(runtime=SimpleNamespace(memory=lambda agent_id: memory))
    server = create_mark_mcp_server(LocalMarkBackend(fake_mark))

    async def drive() -> int:
        ticks = 0
        done = asyncio.Event()

        async def heartbeat() -> None:
            nonlocal ticks
            while not done.is_set():
                ticks += 1
                await asyncio.sleep(0.01)

        beat = asyncio.create_task(heartbeat())
        await server.call_tool("mark_retrieve", {"query": "q"})
        done.set()
        await beat
        return ticks

    assert asyncio.run(drive()) >= 10


# ── LangChain tools / namespaces ──────────────────────────────────────────


def test_langchain_tools_default_namespace_matches_simple_memory(tmp_path: Any) -> None:
    pytest.importorskip("langchain_core")
    from mark.adapters.langchain import create_mark_tools

    with Mark.local(tmp_path) as mark:
        mark.memory.block("architecture").write("Deployments use blue-green rollout.", importance=0.9)
        tools = {t.name: t for t in create_mark_tools(LocalMarkBackend(mark))}
        assert "blue-green" in tools["mark_retrieve"].invoke({"query": "How do deployments work?"})
        assert "blue-green" in asyncio.run(
            tools["mark_retrieve"].ainvoke({"query": "How do deployments work?"})
        )


def test_langgraph_adapter_points_at_existing_extra() -> None:
    try:
        import tomllib
    except ImportError:  # Python < 3.11
        import tomli as tomllib  # type: ignore[no-redef]
    from pathlib import Path

    import mark.adapters.langgraph as langgraph_adapter

    extras = tomllib.loads(
        (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text()
    )["project"]["optional-dependencies"]
    doc = langgraph_adapter.__doc__ or ""
    assert "mark-sdk[langgraph]" not in doc
    assert "mark-sdk[adapters]" in doc and "adapters" in extras
    assert any("langgraph" in dep for dep in extras["tutorial"])
