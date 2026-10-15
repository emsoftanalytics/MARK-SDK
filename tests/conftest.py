# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 MARK Contributors
"""Suite-wide isolation from the developer's environment.

``MARK_EMBEDDER`` is an explicit operator opt-in that changes the default
embedder of ``MarkRuntime.local``; a value exported in a developer shell (for
example an Ollama model) must never make the offline suite depend on a server.
"""
from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _no_embedder_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MARK_EMBEDDER", raising=False)
    monkeypatch.delenv("MARK_EMBEDDER_API_KEY", raising=False)
