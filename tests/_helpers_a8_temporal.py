# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 Somtochukwu Owen Ikeanyi
"""Shared fixtures for the v0.2.0a8 temporal/facts test modules."""
from __future__ import annotations

from pathlib import Path

import pytest

from mark.memory.runtime import MarkRuntime


@pytest.fixture
def runtime(tmp_path: Path):
    rt = MarkRuntime.local(store_path=tmp_path / "mem.db")
    yield rt
    rt.shutdown()
