# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 Somtochukwu Owen Ikeanyi
"""ContextWatch in plain agent loops: OpenAI-style message lists, inject(), small preset."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from mark import ContextWatch
from mark.context.watch import inject
from mark.memory.runtime import MarkRuntime

LOG = [
    "[2026-03-12] TICKET HAL-4433 opened (SEV1): intermittent 502 errors — affects atlas-core, assigned to Mei Lin.",
    "[2026-03-02] HR: Mei Lin joined the Search team, working from Berlin.",
    "[2026-03-05] DECISION ADR-007: all internal service-to-service calls use gRPC with mTLS. " + "x " * 200,
]


@pytest.fixture
def memory(tmp_path: Path):
    runtime = MarkRuntime.local(store_path=tmp_path / "m.db")
    mem = runtime.memory("ops")
    mem.store_many_sync(LOG)
    for value, day in [("6430", "2026-03-01"), ("9239", "2026-03-17"), ("4444", "2026-03-30")]:
        mem.facts.set("atlas-core", "port", value, valid_from=day)
    yield mem
    runtime.shutdown()


class Recorder(ContextWatch):
    """Records what observe_messages routes where (prefetch runs inline)."""

    def __init__(self, memory):
        super().__init__(memory, background=False)
        self.events: list[tuple[str, str]] = []

    def observe(self, text, *, role="user"):
        self.events.append((role, text))
        return super().observe(text, role=role)


def _conversation() -> list[dict]:
    return [
        {"role": "system", "content": "You are on call.\n<mark_briefing>\nold HAL-9999\n</mark_briefing>"},
        {"role": "user", "content": [{"type": "text", "text": "Look into HAL-4433."},
                                     {"type": "image_url", "image_url": {"url": "data:..."}}]},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "c1", "type": "function",
             "function": {"name": "lookup_owner", "arguments": json.dumps({"service": "atlas-core"})}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "atlas-core is owned by Mei Lin."},
        {"role": "assistant", "content": [{"type": "output_text", "text": "Restarting atlas-core on port 9239."}]},
    ]


def test_mixed_content_tool_calls_and_results_are_routed(memory) -> None:
    with Recorder(memory) as watch:
        new = watch.observe_messages(_conversation())
        brief = watch.briefing()
    roles = [role for role, _ in watch.events]
    assert roles == ["user", "tool", "tool", "assistant"]          # system skipped
    assert watch.events[0][1] == "Look into HAL-4433."            # image part ignored
    assert watch.events[1][1] == "atlas-core"                     # parsed JSON args, not tool name
    assert all("HAL-9999" not in text for _, text in watch.events)  # briefing never re-observed
    assert "hal 4433" in new                                       # normalised anchor keys
    assert brief.corrections == ["atlas-core port is 4444 now (since 2026-03-30); 9239 is outdated."]


def test_only_unseen_messages_are_fed(memory) -> None:
    messages = _conversation()
    with Recorder(memory) as watch:
        watch.observe_messages(messages)
        count = len(watch.events)
        assert watch.observe_messages(messages) == []              # same list again: nothing new
        assert len(watch.events) == count
        messages.append({"role": "user", "content": "Who is Mei Lin?"})
        watch.observe_messages(messages)
    assert watch.events[count:] == [("user", "Who is Mei Lin?")]


def test_truncated_or_rewritten_lists_resume_safely(memory) -> None:
    messages = _conversation()
    with Recorder(memory) as watch:
        watch.observe_messages(messages)
        # Sliding window: system kept, oldest turns dropped, one new message appended.
        trimmed = [messages[0]] + messages[3:] + [{"role": "user", "content": "And ADR-007?"}]
        watch.observe_messages(trimmed)
        assert watch.events[-1] == ("user", "And ADR-007?")
        before = len(watch.events)
        # Fully rewritten (summarised) history: only the latest user turn onwards is fed.
        rewritten = [{"role": "user", "content": "Summary: we discussed HAL-4433."},
                     {"role": "assistant", "content": "Noted."},
                     {"role": "user", "content": "Now check Berlin."},
                     {"role": "assistant", "content": "Checking."}]
        watch.observe_messages(rewritten)
        assert watch.events[before:] == [("user", "Now check Berlin."), ("assistant", "Checking.")]
        # Shrunk to nothing, then a fresh conversation.
        watch.observe_messages([])
        watch.observe_messages([{"role": "user", "content": "Hello again"}])
        assert watch.events[-1] == ("user", "Hello again")


def test_long_history_is_bounded_and_bad_messages_never_raise(memory) -> None:
    history = [{"role": "user", "content": f"message {i}"} for i in range(100)]
    history += [None, 42, {"role": "assistant", "tool_calls": [{"function": {"arguments": "{not json"}}]},
                {"content": "no role"}, {"role": "user", "content": object()}]
    with Recorder(memory) as watch:
        watch.observe_messages(history)
        watch.observe_messages(None)  # type: ignore[arg-type]
    assert len(watch.events) <= 20
    assert ("tool", "{not json") in watch.events                   # raw arguments still scanned


def test_inject_does_not_mutate_and_replaces_old_briefing(memory) -> None:
    messages = _conversation()
    original = copy.deepcopy(messages)
    with ContextWatch(memory, background=False) as watch:
        watch.observe_messages(messages)
        brief = watch.briefing()
    out = inject(messages, brief)
    assert messages == original and out is not messages
    system = out[0]["content"]
    assert system.startswith("You are on call.")
    assert system.count("<mark_briefing>") == 1 and "HAL-9999" not in system
    assert inject(out, brief)[0]["content"] == system              # idempotent per turn
    assert out[1:] == messages[1:]


def test_inject_without_system_message_and_with_parts() -> None:
    plain = [{"role": "user", "content": "hi"}]
    out = inject(plain, "<mark_briefing>\nB\n</mark_briefing>")
    assert out[0] == {"role": "system", "content": "<mark_briefing>\nB\n</mark_briefing>"}
    assert plain == [{"role": "user", "content": "hi"}]
    assert inject(plain, "") == plain and inject(plain, "") is not plain
    parts = [{"role": "system", "content": [{"type": "text", "text": "Rules."},
                                            {"type": "text", "text": "<mark_briefing>\nold\n</mark_briefing>"}]}]
    out = inject(parts, "<mark_briefing>\nnew\n</mark_briefing>")
    assert [p["text"] for p in out[0]["content"]] == ["Rules.", "<mark_briefing>\nnew\n</mark_briefing>"]
    assert len(parts[0]["content"]) == 2


def test_small_model_preset_is_tighter(memory) -> None:
    question = [{"role": "user", "content": "HAL-4433, ADR-007, Mei Lin, atlas-core and Berlin: what do we know?"}]
    with ContextWatch(memory, background=False) as full, ContextWatch.for_small_models(memory, background=False) as small:
        full.observe_messages(question)
        small.observe_messages(question)
        big, tight = full.briefing(), small.briefing()
    assert (small.max_anchors, small.per_anchor, small.line_chars, small.follow_links) == (3, 1, 160, 1)
    assert tight.usage.used_tokens <= small.budget.max_tokens
    assert len(tight.anchors) <= 3
    assert len(tight.text) < len(big.text)
    assert all(len(line) <= 162 for line in tight.text.splitlines())
    assert ContextWatch.for_small_models(memory, budget_chars=500, background=False).budget_chars == 500
