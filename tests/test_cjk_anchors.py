# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 MARK Contributors
"""Anchors in unsegmented CJK text: names confirm in timelines and fact lookups.

``\\w`` boundaries cannot work inside Chinese or Japanese sentences (there are
no spaces, so a name is followed by another word character), which made CJK
names never match. Containment is CJK-aware: no boundary is required next to
a CJK character, while Latin identifiers keep whole-token matching.
"""
from __future__ import annotations

import pytest

from mark import Mark
from mark.intelligence.query_understanding import _anchor_norm, _contains_anchor, anchor_present


@pytest.mark.parametrize(("content", "anchor", "expected"), [
    ("王小明昨天搬到了上海。", "王小明", True),
    ("李华昨天搬到了上海。", "王小明", False),
    ("田中さんは東京に引っ越しました。", "田中", True),
    ("佐藤さんは大阪にいます。", "田中", False),
    ("김민수는 서울에 산다", "김민수", True),  # Korean particles attach to names
    ("我在acme工作", "acme", True),  # a Latin name inside CJK text
    ("我在acmeco工作", "acme", False),  # ...still a whole Latin token
    ("atlas cored was retired", "atlas core", False),
    ("the atlas core service", "atlas core", True),
    ("王小明和acme团队", "王小明 acme", False),  # parts present, phrase not
])
def test_contains_anchor_is_cjk_aware(content: str, anchor: str, expected: bool) -> None:
    assert _contains_anchor(_anchor_norm(content), _anchor_norm(anchor)) is expected


def test_anchor_present_accepts_cjk_parts() -> None:
    assert anchor_present(_anchor_norm("王小明和acme团队开会"), "王小明 acme")
    assert not anchor_present(_anchor_norm("李华和acme团队开会"), "王小明 acme")


def test_timeline_finds_chinese_and_japanese_names(tmp_path) -> None:
    with Mark.local(tmp_path) as mark:
        memory = mark.runtime.memory("cjk")
        memory.store_many_sync([
            "[2026-03-01] 王小明搬到了上海。",
            "[2026-04-01] 王小明加入了支付团队。",
            "[2026-04-02] 李华去了北京。",
            "[2026-05-01] 田中さんは東京に引っ越しました。",
        ])
        wang = memory.timeline("王小明")
        assert [e.text.split("] ")[-1] for e in wang] == ["王小明搬到了上海。", "王小明加入了支付团队。"]
        assert [e.text.split("] ")[-1] for e in memory.timeline("田中")] == ["田中さんは東京に引っ越しました。"]
        assert list(memory.timeline("张伟")) == []
        assert list(memory.timeline("佐藤")) == []


def test_fact_subjects_named_in_a_cjk_question_are_found(tmp_path) -> None:
    with Mark.local(tmp_path) as mark:
        memory = mark.runtime.memory("cjk")
        memory.store_sync("团队周会记录：讨论了发布计划。")
        memory.facts.set("王小明", "城市", "上海")
        result = memory.retrieve_sync("王小明住在哪个城市？")
        assert [(f.subject, f.value) for f in result.facts] == [("王小明", "上海")]
        assert memory.retrieve_sync("张伟住在哪个城市？").facts == []
