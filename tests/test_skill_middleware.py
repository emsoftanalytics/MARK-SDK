# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 MARK Contributors
"""Packaged skill loader, SkillMiddleware battery, and adapter auto-injection."""

from __future__ import annotations

import tempfile
import asyncio

import pytest

from mark import Mark, SkillMiddleware, agent_skill, list_packaged_skills, load_skill_text


def test_packaged_skills_are_discoverable():
    names = list_packaged_skills()
    assert "mark-usage" in names
    assert "mark-memory-middleware" in names


def test_load_skill_text_and_prompt_form():
    raw = load_skill_text("mark-usage")
    assert raw.startswith("---")          # SKILL.md frontmatter
    assert "mark-usage" in raw
    prefixed = agent_skill("mark-usage")
    assert prefixed.startswith("\n\n")    # safe to concatenate onto any prompt
    assert agent_skill("does-not-exist") == ""


def test_skill_middleware_binds_prompt_to_runtime():
    with tempfile.TemporaryDirectory() as d:
        with Mark.local(project_path=d, middleware=[SkillMiddleware()]) as mark:
            assert hasattr(mark.runtime, "skill_prompt")
            prompt = mark.runtime.skill_prompt()
            assert "mark-usage" in prompt
            assert "mark-memory-middleware" in prompt


def test_skill_middleware_injects_wrapped_mark_agent_prompt():
    seen = {}

    def llm(prompt: str) -> str:
        seen["prompt"] = prompt
        return "ok"

    with tempfile.TemporaryDirectory() as d:
        with Mark.local(project_path=d, middleware=[SkillMiddleware(skills=["mark-usage"])]) as mark:
            agent = mark.wrap_agent(llm)
            result = asyncio.run(agent.run("Create a continuity plan."))

    assert result.output == "ok"
    assert "mark-usage" in seen["prompt"]
    assert "Create a continuity plan." in seen["prompt"]


def test_skill_middleware_subset_and_texts():
    mw = SkillMiddleware(skills=["mark-usage"])
    mw.bind(runtime=object())
    assert set(mw.texts()) == {"mark-usage"}
    assert "mark-usage" in mw.prompt()


def test_adapter_middleware_auto_injects_skill():
    pytest.importorskip("langchain")
    from types import SimpleNamespace
    from test_langchain_middleware import _Backend, _msg
    from mark.adapters.langchain.middleware import MarkAgentMiddleware, _SystemMessage

    mw = MarkAgentMiddleware(backend=_Backend(), agent_id="skill-agent")
    human = _msg("human", "How are routes structured?")
    update = mw.before_model({"messages": [human]}, runtime=None)

    class _Request:
        state = update
        messages = [human]
        system_message = _SystemMessage(content=[{"type": "text", "text": "Base prompt."}])

        def override(self, **kwargs):
            clone = SimpleNamespace(state=self.state, messages=self.messages)
            clone.system_message = kwargs.get("system_message", self.system_message)
            return clone

    seen = {}
    def handler(request):
        seen["system"] = "".join(
            block.get("text", "") for block in request.system_message.content_blocks
            if isinstance(block, dict)
        )
        return request

    mw.wrap_model_call(_Request(), handler)
    assert "mark-memory-middleware" in seen["system"], "skill should be auto-injected"
    assert "[MARK project memory]" in seen["system"], "context block should still inject"

    # Opt-out keeps the old behavior
    mw_off = MarkAgentMiddleware(backend=_Backend(), agent_id="skill-agent", include_skill=False)
    update_off = mw_off.before_model({"messages": [human]}, runtime=None)
    request = _Request()
    request.state = update_off
    seen.clear()
    mw_off.wrap_model_call(request, handler)
    assert "mark-memory-middleware" not in seen["system"]
