# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 MARK Contributors
"""Known-subject lookup on recall stays fast with ten thousand fact subjects."""
from __future__ import annotations

import time

from mark.intelligence.query_anchors import _anchor_norm
from mark.intelligence.retrieval_reporting import _ReportingMixin
from mark.intelligence.query_understanding import TemporalIntent
from mark.intelligence.subject_index import SubjectIndex
from mark.types import MemoryFragment

SUBJECTS = [f"service-{i:05d}" for i in range(10_000)] + ["Alice", "atlas core", "王小明", "acme", "me"]


def test_subject_index_matches_whole_tokens_only() -> None:
    index = SubjectIndex(tuple(SUBJECTS))
    assert index.named_in(_anchor_norm("Who owns service-00042 and Atlas-Core?")) == ["service-00042", "atlas core"]
    assert index.named_in(_anchor_norm("atlas cored service-000421")) == []
    assert index.named_in(_anchor_norm("王小明搬到了上海")) == ["王小明"]
    assert index.named_in(_anchor_norm("我在acme工作")) == ["acme"]


class _Store:
    def fact_subjects(self, agent_id):
        return list(SUBJECTS)

    def fact_current(self, agent_id, subject=None):
        return []


class _Host(_ReportingMixin):
    _user_subject = "user"

    def __init__(self) -> None:
        self._store = _Store()


def test_matching_facts_with_ten_thousand_subjects_is_fast() -> None:
    host = _Host()
    fragments = [MemoryFragment(content=f"Note {i}: service-0{i}000 was restarted by Alice.", agent_id="a")
                 for i in range(5)]
    host._matching_facts("What port does service-00042 use?", "a", TemporalIntent.NONE, fragments)  # warm
    start = time.process_time()
    for _ in range(5):
        host._matching_facts("What port does service-00042 use?", "a", TemporalIntent.NONE, fragments)
        host._matching_facts("Who restarted it?", "a", TemporalIntent.NONE, fragments)
    assert (time.process_time() - start) / 10 < 0.1
