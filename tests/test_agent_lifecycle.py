"""Lifecycle tests for MarkAgent (middleware binding, reusable runs, payload
isolation, non-blocking sync LLMs) and PluginRegistry rollback.
"""
from __future__ import annotations

import asyncio
import time
from typing import Any, Callable

import pytest

from mark import AgentRun, BaseMiddleware, Mark, MiddlewareContext
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
