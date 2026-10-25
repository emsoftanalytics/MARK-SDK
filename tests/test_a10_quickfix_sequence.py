# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 Somtochukwu Owen Ikeanyi
"""a10 quick fix: the middle parts of a multi-part sequence reach the briefing.

Small-model run: "Step 2: … Step 3: …" and "Days 1 to 3 in Tokyo, then days
4 to 6 in Kyoto" were dropped because the retrieval floor (0.3 of the top
calibrated relevance) and the briefing floor (0.5 of the top ranking score
for lines sharing a question word) each cut them. A part of a sequence now
comes along once another part of the same sequence is kept, and a part that
shares a word with the question is never floored in the briefing.
"""
from __future__ import annotations

from mark.context.relevance import above_floor, is_sequence, sequence_unit
from mark.intelligence.retrieval_finish import apply_floor
from mark.types import MemoryFragment

STEP1 = "Here is my grandmother's pao de lo recipe. Step 1: beat 6 eggs with 200 g sugar for 15 minutes."
STEP23 = "Step 2: gently fold in 120 g of sifted flour. Step 3: bake at 180 C for 35 minutes in a paper-lined tin."
STEP4 = "Step 4, the last one: let it cool upside down for 10 minutes. That's the whole recipe."
JOBS = "And I changed jobs: I'm now a nurse manager at Hospital de Braga, no longer at Sao Joao."
PLAN = "Let's plan my trip to Japan in April. It will be 10 days."
DAYS13 = "Days 1 to 3 in Tokyo, then days 4 to 6 in Kyoto."
DAYS710 = "Then days 7 and 8 in Osaka, and the last 2 days, 9 and 10, back in Tokyo."
P14 = "How many steps does the pao de lo recipe have?"
P22 = "In which city will I be on day 5 of my Japan trip?"


def _f(text: str) -> MemoryFragment:
    return MemoryFragment(content=text, agent_id="a", importance=0.5)


def test_sequence_unit() -> None:
    assert sequence_unit(_f(STEP23)) == "step" and sequence_unit(_f(DAYS13)) == "day"
    assert sequence_unit(_f(PLAN)) is None and not is_sequence(_f(PLAN))


def test_retrieval_floor_keeps_the_siblings_of_a_kept_part() -> None:
    frags = [_f(STEP1), _f(STEP4), _f(STEP23), _f(JOBS), _f(DAYS710)]
    kept, _, _ = apply_floor(P14, frags, [1.0, 0.9, 0.8, 0.7, 0.6], [1.0, 0.4, 0.2, 0.2, 0.15])
    texts = [f.content for f in kept]
    assert texts == [STEP1, STEP4, STEP23]   # the jobs line and the other sequence stay floored


def test_retrieval_floor_still_drops_near_zero_evidence_parts() -> None:
    frags = [_f(STEP1), _f(STEP23)]
    kept, _, _ = apply_floor(P14, frags, [1.0, 0.5], [1.0, 0.05])
    assert [f.content for f in kept] == [STEP1]


def test_briefing_floor_keeps_question_sharing_parts_not_other_sequences() -> None:
    pairs = [(_f(PLAN), 0.743), (_f(STEP4), 0.444), (_f(DAYS13), 0.371), (_f(DAYS710), 0.371)]
    texts = [f.content for f in above_floor(pairs, P22, 0.85)]
    assert DAYS13 in texts and DAYS710 in texts and STEP4 not in texts


def test_briefing_floor_pulls_in_the_middle_step() -> None:
    pairs = [(_f(STEP1), 0.907), (_f(STEP4), 0.484), (_f(JOBS), 0.255), (_f(STEP23), 0.2)]
    texts = [f.content for f in above_floor(pairs, P14, 0.85)]
    assert {STEP1, STEP4, STEP23} <= set(texts) and JOBS not in texts
