# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 MARK Contributors
"""Security hardening: middleware stack robustness (after-hooks on errors,
gap-healing evidence routing, skill permissions, registry, lifecycle, sync deltas)
and RL/logging bounds.
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from mark import Mark
from mark.middlewares import (
    BaseMiddleware,
    GapHealingMiddleware,
    GovernanceMiddleware,
    LifecycleMiddleware,
    MediaContinuityMiddleware,
    MiddlewareContext,
    MiddlewareStack,
    SyncMiddleware,
)
from mark.middlewares.skills import Skill, SkillResult
from mark.middlewares.skills import web_search
from mark.types import MemoryFragment

from _helpers_security import Recorder, _runtime


class _EvidenceSkill(Skill):
    name = "fake_web"
    permissions: set[str] = set()

    def __init__(self, content: str) -> None:
        self._content = content

    async def run(self, input, context):
        return SkillResult(
            content="evidence",
            fragments=[MemoryFragment(content=self._content, agent_id=context["agent_id"],
                                      source="https://example.test/page")],
        )


def test_gap_healing_evidence_goes_through_middleware(tmp_path: Path) -> None:
    recorder = Recorder()
    with Mark.local(tmp_path, middleware=[recorder, GovernanceMiddleware()]) as mark:
        mem = mark.runtime.memory("healer")
        mem.retrieve_sync("What colour is the vault door?", session_id="s1", heal_gaps=True,
                          web_skill=_EvidenceSkill("The vault door is painted blue."))
        stores = [payload for phase, op, payload in recorder.events if phase == "before" and op == "store"]
        evidence = [f for f in mem.list() if "gap_healing" in f.tags]

    assert stores and stores[0]["source"] == "web"
    assert "source:web" in stores[0]["tags"]
    assert evidence and evidence[0].metadata["mark_trust"] == "external"
    assert evidence[0].metadata["source_url"] == "https://example.test/page"
    assert evidence[0].confidence <= 0.5


def test_gap_healing_rejected_evidence_is_quarantined(tmp_path: Path) -> None:
    with Mark.local(tmp_path, middleware=[GovernanceMiddleware()]) as mark:
        mem = mark.runtime.memory("healer")
        result = mem.retrieve_sync("What happened?", session_id="s1", heal_gaps=True,
                                   web_skill=_EvidenceSkill("Traceback (most recent call last)"))
        quarantined = mem.quarantined()
    assert result.external_lookup_used is False
    assert len(quarantined) == 1


def test_default_web_skill_requires_web_permission(tmp_path: Path, monkeypatch) -> None:
    calls: list[Any] = []
    monkeypatch.setattr(web_search.WebSearchSkill, "run_sync", lambda self, *a, **k: calls.append(a))
    runtime = _runtime(tmp_path)
    result = runtime.memory("a").retrieve_sync("unknown thing", session_id="s", heal_gaps=True)
    assert calls == []
    assert result.external_lookup_used is False
    runtime.shutdown()


# ── 8. middleware stack and batteries ─────────────────────────────────────────

def test_after_hooks_run_when_handler_raises() -> None:
    outer, inner = Recorder(), Recorder()
    seen: list[BaseException | None] = []

    class ErrorSpy(BaseMiddleware):
        name = "spy"

        def after(self, context: MiddlewareContext) -> None:
            seen.append(context.error)

    stack = MiddlewareStack([outer, ErrorSpy(), inner])

    def handler(ctx):
        raise RuntimeError("handler failed")

    with pytest.raises(RuntimeError, match="handler failed"):
        stack.run("store", runtime=None, handler=handler)
    assert [e[0] for e in outer.events] == ["before", "after"]
    assert [e[0] for e in inner.events] == ["before", "after"]
    assert isinstance(seen[0], RuntimeError)


def test_after_hooks_run_for_completed_befores_when_a_before_raises() -> None:
    outer = Recorder()

    class Exploding(BaseMiddleware):
        name = "boom"

        def before(self, context: MiddlewareContext) -> None:
            raise ValueError("bad before")

    with pytest.raises(ValueError):
        MiddlewareStack([outer, Exploding()]).run("store", runtime=None, handler=lambda c: None)
    assert [e[0] for e in outer.events] == ["before", "after"]


def test_gap_healing_and_media_continuity_apply_configured_values(tmp_path: Path) -> None:
    skill = _EvidenceSkill("x")
    recorder = Recorder()
    with Mark.local(tmp_path, middleware=[GapHealingMiddleware(web_skill=skill),
                                          MediaContinuityMiddleware(memory_type="episode"),
                                          recorder]) as mark:
        mem = mark.runtime.memory("m")
        mem.observe("Elena enters the Vault.")
        mem.observe("Marco enters the Dock.", memory_type="shot")
        mem.retrieve_sync("Where is Elena?")
    observes = [p for phase, op, p in recorder.events if phase == "before" and op == "observe"]
    retrieves = [p for phase, op, p in recorder.events if phase == "before" and op == "retrieve"]
    assert observes[0]["memory_type"] == "episode"
    assert observes[1]["memory_type"] == "shot"
    assert retrieves[0]["web_skill"] is skill


def test_skill_run_sync_does_not_mask_runtime_error() -> None:
    runs: list[int] = []

    class Failing(Skill):
        name = "failing"

        async def run(self, input, context):
            runs.append(1)
            raise RuntimeError("real failure")

    with pytest.raises(RuntimeError, match="real failure"):
        Failing().run_sync({}, {})
    assert runs == [1]

    async def inside_running_loop() -> None:
        Failing().run_sync({}, {})

    with pytest.raises(RuntimeError, match="real failure"):
        asyncio.run(inside_running_loop())
    assert runs == [1, 1]


def test_registry_alias_and_builtin_shadowing(tmp_path: Path) -> None:
    from mark.middlewares.skills import SkillRegistry
    from mark.middlewares.skills.registry import SkillRegistry as Alias

    assert Alias is SkillRegistry
    skill_dir = tmp_path / "skills" / "echo"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("---\nname: echo\n---\nshadowed!", encoding="utf-8")
    custom = tmp_path / "skills" / "custom"
    custom.mkdir()
    (custom / "SKILL.md").write_text("---\nname: custom\n---\nhello", encoding="utf-8")

    registry = SkillRegistry.with_builtins()
    builtin_echo = registry.get("echo")
    assert registry.from_skills_dir(tmp_path / "skills") == ["custom"]
    assert registry.get("echo") is builtin_echo
    assert registry.from_skills_dir(tmp_path / "skills") == ["custom"]  # dir skills can overwrite each other
    assert "echo" in registry.from_skills_dir(tmp_path / "skills", allow_builtin_override=True)
    assert registry.get("echo") is not builtin_echo


def test_lifecycle_ignores_rejected_and_stopped_ops() -> None:
    cycles: list[str] = []

    class FakeRuntime:
        def run_cycle(self, agent_id, **kw):
            cycles.append(agent_id)
            return {}

    lifecycle = LifecycleMiddleware(every_n_writes=1)
    rejected = MiddlewareContext("store", FakeRuntime(), "a",
                                 payload={"metadata": {"mark_governance": {"passed": False}}})
    stopped = MiddlewareContext("store", FakeRuntime(), "a")
    stopped.stop(None)
    failed = MiddlewareContext("store", FakeRuntime(), "a", error=RuntimeError())
    ok = MiddlewareContext("store", FakeRuntime(), "a")
    for ctx in (rejected, stopped, failed):
        lifecycle.after(ctx)
    assert cycles == [] and lifecycle._write_count == 0
    lifecycle.after(ok)
    assert cycles == ["a"]


def test_sync_middleware_prepares_only_deltas_and_surfaces_upload_errors(tmp_path: Path, caplog) -> None:
    class FlakyCloud:
        def __init__(self) -> None:
            self.fail = True
            self.uploads: list[int] = []

        def upload_delta(self, delta):
            if self.fail:
                raise ConnectionError("cloud down")
            self.uploads.append(len(delta.fragments))

    cloud = FlakyCloud()
    sync = SyncMiddleware(cloud=cloud, upload=True)
    with Mark.local(tmp_path, middleware=[sync]) as mark:
        mem = mark.runtime.memory("s")
        mem.store_sync("first fact")
        assert sync.last_stats.errors == 1
        assert "ConnectionError" in sync.last_stats.error_messages[0]

        cloud.fail = False
        mem.store_sync("second fact")  # failed delta is retried together with the new one
        mem.store_sync("third fact")
        assert cloud.uploads == [2, 1]

        strict = SyncMiddleware(cloud=FlakyCloud(), upload=True, raise_on_error=True)
        mark.runtime.middleware.add(strict)
        with pytest.raises(ConnectionError):
            mem.store_sync("fourth fact")


# ── 13. RL bounds ─────────────────────────────────────────────────────────────

def test_replay_buffer_rejects_non_positive_capacity() -> None:
    from mark.rl.buffer import ReplayBuffer

    for bad in (0, -1):
        with pytest.raises(ValueError):
            ReplayBuffer(capacity=bad)


def test_routing_logger_is_bounded() -> None:
    from mark.rl.collectors import RoutingDecisionLogger

    logger = RoutingDecisionLogger(max_entries=3)
    for i in range(10):
        logger.log(None, f"h{i}", "FAST")
    assert [e["query_hash"] for e in logger.recent(10)] == ["h7", "h8", "h9"]
    assert [e["query_hash"] for e in logger.recent(1)] == ["h9"]
