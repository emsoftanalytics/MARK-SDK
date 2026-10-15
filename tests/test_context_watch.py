# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 MARK Contributors
"""ContextWatch: anticipatory context, served without the agent asking."""
from __future__ import annotations

from pathlib import Path

import pytest

from mark import ContextWatch
from mark.memory.runtime import MarkRuntime

LOG = [
    "[2026-03-01] INVENTORY: atlas-core is owned by Liam O'Connor, listens on port 6430.",
    "[2026-03-17] CHANGE: atlas-core moved from port 6430 to port 9239.",
    "[2026-03-30] CHANGE: atlas-core moved from port 9239 to port 4444.",
    "[2026-03-12] TICKET HAL-4433 opened (SEV1): intermittent 502 errors — affects atlas-core, assigned to Mei Lin.",
    "[2026-03-02] HR: Mei Lin joined the Search team, working from Berlin.",
    "[2026-03-05] DECISION ADR-007: all internal service-to-service calls use gRPC with mTLS.",
]


@pytest.fixture
def memory(tmp_path: Path):
    runtime = MarkRuntime.local(store_path=tmp_path / "w.db")
    mem = runtime.memory("ops")
    mem.store_many_sync(LOG)
    for value, day in [("6430", "2026-03-01"), ("9239", "2026-03-17"), ("4444", "2026-03-30")]:
        mem.facts.set("atlas-core", "port", value, valid_from=day)
    yield mem
    runtime.shutdown()


def test_briefing_is_ready_without_asking(memory) -> None:
    with ContextWatch(memory) as watch:
        watch.observe("Please look into ticket HAL-4433.", role="user")
        brief = watch.briefing(wait=2.0)
    assert brief.text.startswith("<mark_briefing>")
    assert "TICKET HAL-4433 opened" in brief.text
    assert brief.pending == 0


def test_links_are_anticipated(memory) -> None:
    with ContextWatch(memory) as watch:
        watch.observe("Please look into ticket HAL-4433.", role="user")
        brief = watch.briefing(wait=2.0)
    # The ticket names atlas-core: its current port is ready before anyone asks.
    assert "atlas-core (linked):" in brief.text
    assert "port 4444" in brief.text


def test_briefing_never_blocks_by_default(memory) -> None:
    with ContextWatch(memory) as watch:
        watch.observe("What is the port of atlas-core?", role="user")
        brief = watch.briefing()  # no wait: returns whatever is ready
        assert brief.latency_ms < 50
        watch.flush(timeout=2.0)
        assert "atlas-core" in watch.briefing().text


def test_stale_claims_are_corrected_but_guesses_are_not(memory) -> None:
    with ContextWatch(memory, background=False) as watch:
        watch.observe("Investigate atlas-core.", role="user")
        watch.observe("I'll restart atlas-core on port 7777.", role="assistant")  # never a known value
        assert watch.briefing().corrections == []
        watch.observe("Restarting atlas-core on port 9239.", role="assistant")  # superseded value
        brief = watch.briefing()
        assert brief.corrections == ["atlas-core port is 4444 now (since 2026-03-30); 9239 is outdated."]
        assert brief.text.splitlines()[1].startswith("Corrections")
        watch.observe("Using atlas-core port 4444.", role="assistant")
        assert watch.briefing().corrections == []


def test_unknown_subjects_are_flagged_and_tool_names_ignored(memory) -> None:
    with ContextWatch(memory, background=False) as watch:
        watch.observe("Check the billing incident.", role="user")
        watch.observe_tool_call("lookup_owner", {"service": "titan-billing"})
        brief = watch.briefing()
    assert brief.unknown == ["titan-billing"]
    assert "lookup" not in brief.text


def test_budget_is_respected(memory) -> None:
    with ContextWatch(memory, background=False, budget_chars=200) as watch:
        watch.observe("HAL-4433 and atlas-core and Mei Lin and ADR-007", role="user")
        assert len(watch.briefing().text) <= 200 + len("<mark_briefing>\n\n</mark_briefing>")


def test_anchors_expire_after_ttl_turns(memory) -> None:
    with ContextWatch(memory, background=False, ttl_turns=1) as watch:
        watch.observe("Look at atlas-core.", role="user")
        assert "atlas-core" in watch.briefing().anchors
        watch.observe("Unrelated question one.", role="user")
        watch.observe("Unrelated question two.", role="user")
        assert "atlas-core" not in watch.briefing().anchors


def test_prefetch_failures_never_reach_the_agent(memory, monkeypatch, caplog) -> None:
    def boom(*_, **__):
        raise RuntimeError("disk on fire")

    monkeypatch.setattr(memory, "retrieve_many_sync", boom)
    with ContextWatch(memory) as watch:
        watch.observe("Look at atlas-core.", role="user")
        assert watch.flush(timeout=2.0)
        assert watch.briefing().text == ""
    assert "disk on fire" in caplog.text


def test_invalid_role_rejected(memory) -> None:
    with ContextWatch(memory) as watch, pytest.raises(ValueError):
        watch.observe("hi", role="robot")


def test_subjects_only_the_model_named_stay_quiet_until_confirmed(memory) -> None:
    with ContextWatch(memory, background=False) as watch:
        watch.observe("Any update on the outage?", role="user")
        # The model's own wording must not become "Not in memory" noise...
        watch.observe("Olá Mei! The nectar-rich fix for HAL-4433 is underway.", role="assistant")
        brief = watch.briefing()
        assert brief.unknown == []
        assert "nectar-rich" not in brief.text and "Olá Mei" not in brief.anchors
        # ...but a subject the model names that memory knows is still served.
        assert "HAL-4433" in brief.anchors and "TICKET HAL-4433 opened" in brief.text
        # Once the user names an unknown subject it is reported as usual.
        watch.observe("What is nectar-rich?", role="user")
        assert watch.briefing().unknown == ["nectar-rich"]
