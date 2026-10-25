# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 Somtochukwu Owen Ikeanyi
"""One API: mark.memory has every verb of runtime.memory(id); errors and kinds are public."""
from __future__ import annotations

import inspect
import subprocess
import sys
import warnings
from pathlib import Path

import pytest

import mark
from mark.kinds import kind_from_text, kind_of
from mark.memory.mark_memory import MarkMemory
from mark.memory.simple import SimpleMemory
from mark.runtime import Mark

#: Verbs every memory object offers (plain = sync, a-prefix = async).
VERBS = ("remember", "aremember", "remember_many", "recall", "arecall", "forget", "aforget",
         "get", "list", "count", "update", "aupdate", "supersede", "asupersede", "resolve",
         "aresolve", "facts", "timeline", "brief", "store_document", "remember_file", "documents",
         "images", "store_many_sync", "export_jsonl", "import_jsonl", "vacuum", "embedder_info")


@pytest.mark.parametrize("verb", VERBS)
def test_every_verb_exists_on_both_memories(verb: str) -> None:
    assert hasattr(MarkMemory, verb), verb
    assert hasattr(SimpleMemory, verb), verb


def test_simple_memory_documents_files_images_and_remember_many(tmp_path: Path) -> None:
    doc = tmp_path / "guide.md"
    doc.write_text("# Deploy\n\nRun `make deploy` from the repo root.\n", encoding="utf-8")
    with Mark.local(tmp_path) as m:
        record = m.memory.store_document("# Ops\n\nThe pager rota is weekly.\n", title="ops")
        assert m.memory.documents.get(record.id) is not None
        assert m.memory.remember_file(doc) is not None
        assert m.memory.documents.count() == 2
        assert m.memory.images is not None
        report = m.memory.remember_many(["TODO: add retries", "", "DECISION: use SQLite"])
        assert report.stored == 2 and len(report.skipped) == 1 and report[1] is None
        assert {kind_of(m.memory.get(i)) for i in report.ids} == {"todo", "decision"}
        again = m.memory.remember_many(["TODO: add retries", "", "DECISION: use SQLite"],
                                       idempotency_key="k")
        same = m.memory.remember_many(["TODO: add retries", "", "DECISION: use SQLite"],
                                      idempotency_key="k")
        assert same.ids == again.ids and same.existing == 2


def test_simple_retrieve_is_deprecated_for_callers_but_still_works(tmp_path: Path) -> None:
    with Mark.local(tmp_path) as m:
        m.memory.remember("We chose SQLite for storage.")
        with pytest.warns(DeprecationWarning, match="recall"):
            bundle = m.memory.retrieve("database")
        assert "SQLite" in bundle.text
        agent = m.wrap_agent(lambda prompt: "ok")
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)   # internal callers stay quiet
            agent("which database?")


def test_errors_are_exported_lazily_from_mark() -> None:
    for name in ("MarkError", "StoreVersionError", "InvalidMemoryError", "MemoryTooLargeError",
                 "StoreClosedError", "InvalidInputError"):
        assert name in mark.__all__
        assert issubclass(getattr(mark, name), mark.MarkError)
    out = subprocess.run([sys.executable, "-c", "import mark, sys; mark.StoreClosedError; "
                          "print('mark.runtime' in sys.modules, 'mark.memory' in sys.modules)"],
                         capture_output=True, text=True, check=True).stdout
    assert out.split() == ["False", "False"]


def test_close_is_an_alias_of_shutdown(tmp_path: Path) -> None:
    m = Mark.local(tmp_path)
    m.close()
    from mark.errors import StoreClosedError

    with pytest.raises(StoreClosedError):
        m.memory.remember("after close")


@pytest.mark.parametrize("text,kind", [
    ("TODO: add retries", "todo"), ("todo: lower case", "todo"), ("DECISION: SQLite", "decision"),
    ("ADR-7: no ORM", "decision"), ("BUG: crash on start", "bug"), ("FIX: guard None", "fix"),
    ("FIXED: the leak", "fix"), ("DONE: pagination", "note"), ("NOTE: rota is weekly", "note"),
    ("API: GET /tasks", "api"), ("SNIPPET: uv run pytest", "snippet"),
    ("[2026-10-02] TODO: dated", "todo"), ("TODO - dash form", "todo"),
    ("API-key rotation is monthly", None), ("Todo list app", None), ("We chose SQLite", None),
])
def test_kind_from_text(text: str, kind: str | None) -> None:
    assert kind_from_text(text) == kind


def test_kind_of_reads_tags_or_fragments(tmp_path: Path) -> None:
    assert kind_of(["kind:bug", "x"]) == "bug"
    with Mark.local(tmp_path) as m:
        fid = m.memory.remember("BUG: crash on start")
        assert kind_of(m.memory.get(fid)) == "bug"
        assert inspect.signature(kind_of)
