# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 MARK Contributors
"""Shared fixtures and fakes for the test_security_* modules."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from mark.embeddings import HashEmbeddingProvider
from mark.memory.runtime import MarkRuntime
from mark.middlewares import BaseMiddleware, MiddlewareContext


class Recorder(BaseMiddleware):
    name = "recorder"

    def __init__(self) -> None:
        self.events: list[tuple[str, str, dict[str, Any]]] = []

    def before(self, context: MiddlewareContext) -> None:
        self.events.append(("before", context.operation, dict(context.payload)))

    def after(self, context: MiddlewareContext) -> None:
        self.events.append(("after", context.operation, dict(context.payload)))


def _runtime(tmp_path: Path, **kw) -> MarkRuntime:
    return MarkRuntime.local(store_path=tmp_path / "m.db", embedder=HashEmbeddingProvider(dim=32), **kw)
