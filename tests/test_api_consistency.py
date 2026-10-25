# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 Somtochukwu Owen Ikeanyi
"""D1/D5: one API — same verbs, sync/async convention and defaults on every memory object."""
from __future__ import annotations

import asyncio
import inspect
import warnings
from pathlib import Path

import pytest

from mark import Mark
from mark.errors import InvalidInputError, MarkError, UnknownBlockError

#: The canonical verbs; plain = sync, a-prefix = async twin.
VERBS = ("remember", "recall", "forget", "get", "list", "count", "update", "supersede", "resolve",
         "timeline", "export_jsonl", "import_jsonl", "vacuum", "embedder_info")
ASYNC_TWINS = ("aremember", "arecall", "aforget", "aupdate", "asupersede", "aresolve")


@pytest.fixture()
def mark(tmp_path: Path):
    with Mark.local(tmp_path) as instance:
        yield instance


def test_both_memory_objects_offer_the_same_verbs(mark) -> None:
    simple, agent = mark.memory, mark.runtime.memory("agent")
    for name in (*VERBS, "facts"):
        assert hasattr(simple, name) and hasattr(agent, name), name
    for name in VERBS:
        assert not inspect.iscoroutinefunction(getattr(simple, name)), name
        assert not inspect.iscoroutinefunction(getattr(agent, name)), name
    for name in ASYNC_TWINS:
        assert inspect.iscoroutinefunction(getattr(simple, name)), name
        assert inspect.iscoroutinefunction(getattr(agent, name)), name


def test_same_defaults_everywhere(mark) -> None:
    block_record = mark.memory.block("prefs").write("I'm vegetarian.")
    simple = mark.memory.get(mark.memory.remember("Tea, no sugar."))
    agent_mem = mark.runtime.memory("a")
    agent = agent_mem.get(agent_mem.remember("Coffee, black."))
    legacy = agent_mem.get(agent_mem.store_sync("Water."))
    for item in (block_record, simple, agent, legacy):
        assert (item.importance, item.confidence) == (0.5, 1.0)


def test_recall_and_arecall_return_the_same_result_type(mark) -> None:
    mark.memory.remember("The guest Wi-Fi is halcyon-7.", block="office")
    agent = mark.runtime.memory("a")
    agent.remember("The guest Wi-Fi is halcyon-7.")

    async def both():
        return await mark.memory.arecall("guest wifi"), await agent.arecall("guest wifi")

    sync_results = (mark.memory.recall("guest wifi"), agent.recall("guest wifi"))
    async_results = asyncio.run(both())
    for result in (*sync_results, *async_results):
        assert type(result).__name__ == "RetrievalResult"
        assert "halcyon-7" in result.as_context()
    assert mark.memory.recall("guest wifi", blocks=["office"]).fragments[0].content.endswith("halcyon-7.")


def test_aremember_runs_the_sync_path(mark) -> None:
    async def write():
        return await mark.runtime.memory("a").aremember("Async note.", kind="note")

    fid = asyncio.run(write())
    assert "kind:note" in mark.runtime.memory("a").get(fid).tags


def test_unknown_blocks_raise_and_list_the_known_ones(mark) -> None:
    mark.memory.block("prefs").write("I'm vegetarian.")
    mark.memory.block("empty-but-created")
    with pytest.raises(UnknownBlockError, match="known blocks: empty-but-created, prefs"):
        mark.memory.retrieve("diet", blocks=["prefz"])
    with pytest.raises(UnknownBlockError):
        mark.memory.recall("diet", blocks=["nope"])
    assert mark.memory.retrieve("diet", blocks=["empty-but-created"]).memories == []
    assert isinstance(UnknownBlockError([], []), (MarkError, ValueError))


def test_block_kind_is_honoured(mark) -> None:
    prefs = mark.memory.block("prefs", kind="preferences", description="user prefs")
    record = prefs.write("Prefers dark mode.")
    assert prefs.to_dict() == {"id": "prefs", "label": "prefs", "kind": "preferences",
                               "description": "user prefs"}
    assert "kind:preference" in mark.memory.get(record.id).tags
    labelled = mark.memory.block("misc", kind="project_context").write("Free-form label only.")
    assert mark.memory.get(labelled.id).metadata["block_kind"] == "project_context"


def test_invalid_input_raises_mark_errors_not_pydantic(mark) -> None:
    for call in (lambda: mark.memory.remember(""),
                 lambda: mark.memory.remember("x", importance=5),
                 lambda: mark.memory.block("b").write("x", confidence=-1),
                 lambda: mark.runtime.memory("a").recall("x", k=0)):
        with pytest.raises(InvalidInputError) as info:
            call()
        assert "pydantic" not in type(info.value).__module__


def test_list_paginates_and_warns_instead_of_capping_silently(mark) -> None:
    agent = mark.runtime.memory("bulk")
    agent.store_many_sync([f"note {i}" for i in range(105)])
    with pytest.warns(UserWarning, match="first 100 of 105"):
        assert len(agent.list()) == 100
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert len(agent.list(limit=None)) == 105
        pages = [agent.list(limit=50, offset=o) for o in (0, 50, 100)]
    assert [len(p) for p in pages] == [50, 50, 5]
    assert len({f.id for page in pages for f in page}) == 105
    assert agent.count() == 105


def test_mark_local_accepts_a_db_file_like_the_cli(tmp_path: Path) -> None:
    db = tmp_path / "notes.db"
    with Mark.local(db) as first:
        fid = first.memory.remember("Stored through a .db path.")
        assert first.db_path == db.resolve()
    assert db.is_file()
    with Mark.local(tmp_path, store_path=db) as second:
        assert second.memory.get(fid) is not None
    with Mark.local(tmp_path / "proj") as third:
        assert third.db_path == (tmp_path / "proj" / ".mark" / "memory.db").resolve()


def test_wrap_agent_is_callable_sync_and_async(mark) -> None:
    mark.memory.block("prefs").write("I'm vegetarian.")
    agent = mark.wrap_agent(lambda prompt, **_: f"echo:{prompt[-40:]}", blocks=["prefs", "later"])
    assert agent("What do I eat?").output.startswith("echo:")
    assert agent.invoke("What do I eat?").output.startswith("echo:")

    async def run():
        first = await agent.ainvoke("What do I eat?")
        with pytest.raises(MarkError, match="ainvoke"):
            agent.invoke("blocked inside a loop")
        return first

    assert asyncio.run(run()).output.startswith("echo:")


def test_store_wide_agents(mark) -> None:
    mark.memory.remember("simple")
    mark.runtime.memory("coder").remember("agent")
    assert mark.agents() == ["__mark__", "coder"]
