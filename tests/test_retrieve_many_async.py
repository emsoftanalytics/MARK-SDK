# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 MARK Contributors
"""MarkMemory.retrieve_many: the async twin of retrieve_many_sync."""
from __future__ import annotations

import asyncio
import inspect

from mark.memory.mark_memory import MarkMemory
from mark.memory.runtime import MarkRuntime


def test_retrieve_many_matches_sync_and_keeps_query_order() -> None:
    runtime = MarkRuntime.local()
    try:
        memory = runtime.memory("agent")
        memory.store_sync("The deploy window is Friday at 18:00.", tags=["ops"])
        memory.store_sync("Atlas-core runs in eu-west-1.", tags=["infra"])
        queries = ["When is the deploy window?", "Where does atlas-core run?"]
        results = asyncio.run(memory.retrieve_many(queries, tags=["infra"]))
        expected = memory.retrieve_many_sync(queries, tags=["infra"])
        assert [r.query for r in results] == queries
        assert [[f.id for f in r.fragments] for r in results] == [[f.id for f in r.fragments] for r in expected]
        assert all(f.tags == ["infra"] for r in results for f in r.fragments)
    finally:
        runtime.shutdown()


def test_retrieve_many_signature_mirrors_sync() -> None:
    assert inspect.iscoroutinefunction(MarkMemory.retrieve_many)
    sync = inspect.signature(MarkMemory.retrieve_many_sync).parameters
    assert inspect.signature(MarkMemory.retrieve_many).parameters.keys() == sync.keys()
