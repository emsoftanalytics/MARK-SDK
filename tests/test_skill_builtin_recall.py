# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 MARK Contributors
"""The built-in mark_memory skill reads through recall(), not the deprecated retrieve()."""
from __future__ import annotations

import asyncio
import warnings

from mark import Mark
from mark.middlewares.skills.builtin import MarkMemorySkill


def test_retrieve_action_uses_recall_without_deprecation(tmp_path) -> None:
    with Mark.local(tmp_path) as mark:
        mark.memory.remember("payments-api uses Postgres 16")
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            result = asyncio.run(MarkMemorySkill(mark.memory).run(
                {"action": "retrieve", "query": "Which database does payments-api use?"}, None))
    assert "Postgres 16" in result.content
    assert result.metadata["records"][0]["content"] == "payments-api uses Postgres 16"
