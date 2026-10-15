# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 MARK Contributors
"""Shared fixtures and helpers for the LocalMemoryStore schema-v3 test modules."""
from __future__ import annotations

from pathlib import Path

import pytest

from mark.store import LocalMemoryStore
from mark.types import MemoryFragment, MemoryState


AGENT = "agent-a"


def _frag(content: str, **kwargs) -> MemoryFragment:
    kwargs.setdefault("agent_id", AGENT)
    kwargs.setdefault("state", MemoryState.UNVERIFIED)
    return MemoryFragment(content=content, **kwargs)


@pytest.fixture(params=["memory", "file"])
def store(request, tmp_path: Path):
    path = ":memory:" if request.param == "memory" else tmp_path / "mem.db"
    s = LocalMemoryStore(path)
    yield s
    s.close()
