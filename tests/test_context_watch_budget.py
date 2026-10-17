"""ContextWatch under a token budget: cap, priority, usage, presets, inject() and MCP."""
from __future__ import annotations

from pathlib import Path

import pytest

from mark import ContextWatch
from mark.context.budget import ContextBudget, estimate_tokens
from mark.context.watch import inject
from mark.memory.runtime import MarkRuntime

LOG = [
    "[2026-03-01] INVENTORY: atlas-core is owned by Liam O'Connor, listens on port 6430.",
    "[2026-03-17] CHANGE: atlas-core moved from port 6430 to port 9239.",
    "[2026-03-30] CHANGE: atlas-core moved from port 9239 to port 4444.",
    "[2026-03-12] TICKET HAL-4433 opened (SEV1): intermittent 502 errors — affects atlas-core, assigned to Mei Lin.",
    "[2026-03-02] HR: Mei Lin joined the Search team, working from Berlin.",
    "[2026-03-05] DECISION ADR-007: all internal service-to-service calls use gRPC with mTLS.",
]
QUESTION = "HAL-4433, ADR-007, Mei Lin, atlas-core and Zorbatron-9: what do we know?"


@pytest.fixture
def memory(tmp_path: Path):
    runtime = MarkRuntime.local(store_path=tmp_path / "b.db")
    mem = runtime.memory("ops")
    mem.store_many_sync(LOG)
    for value, day in [("6430", "2026-03-01"), ("9239", "2026-03-17"), ("4444", "2026-03-30")]:
        mem.facts.set("atlas-core", "port", value, valid_from=day)
    yield mem
    runtime.shutdown()


def _primed(memory, **kwargs) -> ContextWatch:
    watch = ContextWatch(memory, background=False, **kwargs)
    watch.observe(QUESTION, role="user")
    watch.observe("I'll restart atlas-core on port 9239.", role="assistant")
    return watch


def test_no_budget_keeps_the_original_output(memory) -> None:
    with _primed(memory) as plain:
        brief = plain.briefing()
    assert brief.usage is None and brief.items == ()
    assert brief.text.startswith("<mark_briefing>\nCorrections (memory has newer values):")
    assert "Not in memory: Zorbatron-9" in brief.text


@pytest.mark.parametrize("cap", [0, 10, 25, 40, 60, 90, 150, 300, 1000])
def test_briefing_never_exceeds_the_cap(memory, cap) -> None:
    with _primed(memory, budget_tokens=cap) as watch:
        brief = watch.briefing()
    assert estimate_tokens(brief.text) <= cap
    assert brief.usage.used_tokens == estimate_tokens(brief.text) and brief.usage.max_tokens == cap


def test_small_budget_keeps_correction_and_unknown_first(memory) -> None:
    with _primed(memory, budget_tokens=70) as watch:
        brief = watch.briefing()
    assert "port is 4444 now" in brief.text
    assert "Not in memory: Zorbatron-9" in brief.text
    sections = brief.usage.sections
    assert sections["corrections"].included == 1 and sections["unknown"].included == 1
    assert brief.usage.dropped > 0


def test_custom_counter_and_per_call_override(memory) -> None:
    words = ContextBudget(30, token_counter=lambda s: len(s.split()))
    with _primed(memory, budget_tokens=words) as watch:
        brief = watch.briefing()
        assert len(brief.text.split()) <= 30
        wider = watch.briefing(budget_tokens=2000)
        assert wider.usage.max_tokens == 2000 and len(wider.text) > len(brief.text)


def test_fill_lifts_count_limits(memory) -> None:
    with _primed(memory, budget_tokens=2000, per_anchor=1, max_anchors=1) as capped, \
            _primed(memory, budget_tokens=2000, per_anchor=1, max_anchors=1, budget_fill=True) as filled:
        few, many = capped.briefing(), filled.briefing()
    assert len(few.memory_ids) <= 2 < len(many.memory_ids)
    assert len(many.anchors) > len(few.anchors) == 1


def test_budgeted_briefing_is_deterministic(memory) -> None:
    with _primed(memory, budget_tokens=80) as a, _primed(memory, budget_tokens=80) as b:
        assert a.briefing().text == b.briefing().text


def test_small_model_preset_sets_a_token_budget(memory) -> None:
    with ContextWatch.for_small_models(memory, background=False) as small:
        assert small.budget.max_tokens == 175
        small.observe(QUESTION, role="user")
        assert small.briefing().usage.used_tokens <= 175
    assert ContextWatch.for_small_models(memory, budget_tokens=None, background=False).budget is None


def test_inject_applies_a_budget(memory) -> None:
    messages = [{"role": "system", "content": "Rules."}, {"role": "user", "content": QUESTION}]
    with _primed(memory, budget_tokens=1000) as watch:
        brief = watch.briefing()
    out = inject(messages, brief, budget_tokens=40)
    block = out[0]["content"].split("Rules.\n\n", 1)[1]
    assert estimate_tokens(block) <= 40 and "4444 now" in block
    # Plain text: whole lines, the <mark_briefing> wrapper kept intact.
    text = "<mark_briefing>\nline one is here\n" + "x" * 300 + "\nline three\n</mark_briefing>"
    cut = inject([], text, budget_tokens=20)[0]["content"]
    assert cut == "<mark_briefing>\nline one is here\nline three\n</mark_briefing>"
    assert inject([], "a\nb", budget_tokens=0) == []


def test_mcp_briefing_passes_the_budget_through(memory) -> None:
    from mark.adapters.mcp._briefing import BriefingSessions, _briefing_sync

    sessions = BriefingSessions(lambda agent: ContextWatch(memory, background=False))
    try:
        full = _briefing_sync(sessions, "ops", "s", QUESTION, "user", None, 0.0)
        small = _briefing_sync(sessions, "ops", "s", QUESTION, "user", None, 0.0, 30)
    finally:
        sessions.close_all()
    assert estimate_tokens(small) <= 30 < estimate_tokens(full)


def test_refresh_rechecks_subjects_stored_after_first_mention(memory) -> None:
    with ContextWatch(memory, background=False) as watch:
        watch.observe("Tell me about Quokka-77.", role="user")
        assert watch.briefing().unknown == ["Quokka-77"]
        memory.store_sync("Quokka-77 is the staging cluster in Dublin.")
        watch.refresh(unknown_only=True)
        watch.observe("And Quokka-77?", role="user")
        brief = watch.briefing()
    assert brief.unknown == [] and "staging cluster in Dublin" in brief.text
