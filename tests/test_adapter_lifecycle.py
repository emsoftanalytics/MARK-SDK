"""Lifecycle tests for adapters: LocalMarkBackend.feedback, the MCP adapter
(schema bounds, namespace allowlist, non-blocking tools), and LangChain/LangGraph
namespace and extras wiring.
"""
from __future__ import annotations

import asyncio
import time
from typing import Any

import pytest

from mark import Mark
from mark.adapters.backend import LocalMarkBackend


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
            try:   # a denied namespace is an isError result (ToolError from call_tool)
                await server.call_tool("mark_write", {"content": "secret plan", "agent_id": "team-b"})
                denied = "accepted"
            except Exception as exc:
                denied = str(exc)
            allowed = await server.call_tool("mark_write", {"content": "team a fact", "agent_id": "team-a"})
            default = await server.call_tool("mark_write", {"content": "default namespace fact"})
            return denied, str(allowed), str(default)

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
