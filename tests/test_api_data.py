# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 MARK Contributors
"""D2 data management (get/count/export/import) and recall filters (session/tag/kind/since/until)."""
from __future__ import annotations

import json
import stat
from pathlib import Path

import pytest

from mark import Mark
from mark.errors import InvalidInputError, StoreEncryptionError
from mark.memory.runtime import MarkRuntime
from mark.types import MemoryDocument


def _seed(mark: Mark) -> dict[str, str]:
    ids = {
        "todo": mark.memory.remember("TODO: rotate the API keys", kind="todo", tags=["ops"]),
        "fact": mark.memory.remember("[2026-03-01] atlas-core runs on port 6430.", extract_facts=True),
        "coder": mark.runtime.memory("coder").remember("Tests run with pytest -q.", kind="snippet"),
    }
    mark.memory.resolve(ids["todo"])
    mark.memory.facts.set("me", "city", "Lisbon", valid_from="2024-01-01")
    mark.memory.facts.set("me", "city", "Porto", valid_from="2026-01-01")
    mark.runtime.store.store_document(MemoryDocument(title="Runbook", content="Restart with care.",
                                                     agent_id="coder", fragment_ids=[ids["coder"]]))
    return ids


def test_export_import_round_trip(tmp_path: Path) -> None:
    backup = tmp_path / "backup.jsonl"
    with Mark.local(tmp_path / "a.db") as source:
        ids = _seed(source)
        counts = source.export_jsonl(backup)
    assert counts == {"memories": 3, "facts": 3, "documents": 1}
    assert stat.S_IMODE(backup.stat().st_mode) == 0o600
    header = json.loads(backup.read_text().splitlines()[0])
    assert header["format"] == "mark-jsonl" and header["agents"] == ["__mark__", "coder"]

    with Mark.local(tmp_path / "b.db") as target:
        assert target.import_jsonl(backup) == {"memories": 3, "facts": 3, "documents": 1, "skipped": 0}
        again = target.import_jsonl(backup)  # idempotent restore
        assert again["memories"] == again["facts"] == again["documents"] == 0
        restored = target.memory.get(ids["todo"])
        assert "mark:resolved" in restored.tags and restored.metadata["lifecycle"] == "resolved"
        assert [h.value for h in target.memory.facts.history("me", "city")] == ["Lisbon", "Porto"]
        assert target.memory.facts.get("atlas-core", "port").source_fragment_id == ids["fact"]
        coder = target.memory_for("coder")
        assert coder.recall("how do tests run").fragments[0].id == ids["coder"]  # re-embedded
        assert target.runtime.store.get_document(target.runtime.store.document_rows("coder")[0]["id"]).title == "Runbook"
        assert target.agents() == ["__mark__", "coder"]


def test_memory_level_export_imports_into_another_agent(tmp_path: Path) -> None:
    with Mark.local(tmp_path) as mark:
        coder = mark.memory_for("coder")
        coder.remember("Use ruff.", kind="preference")
        assert coder.export_jsonl(tmp_path / "c.jsonl")["memories"] == 1
        reviewer = mark.memory_for("reviewer")
        assert reviewer.import_jsonl(tmp_path / "c.jsonl")["memories"] == 0  # same id exists
        mark.memory_for("coder").forget(match="ruff")
        assert reviewer.import_jsonl(tmp_path / "c.jsonl")["memories"] == 1
        assert [f.content for f in reviewer.list(kind="preference")] == ["Use ruff."]


def test_import_rejects_files_that_are_not_exports(tmp_path: Path) -> None:
    bogus = tmp_path / "x.jsonl"
    bogus.write_text('{"hello": 1}\n')
    with Mark.local(tmp_path) as mark:
        with pytest.raises(InvalidInputError, match="not a MARK export"):
            mark.import_jsonl(bogus)


def test_encrypted_store_exports_plaintext_only_when_asked(tmp_path: Path) -> None:
    pytest.importorskip("cryptography")
    from mark.security.encryption import FernetEncryptionProvider

    runtime = MarkRuntime.local(store_path=tmp_path / "e.db",
                                encryption=FernetEncryptionProvider(passphrase="pw"))
    memory = runtime.memory("a")
    memory.remember("The safe code is 4821.")
    with pytest.raises(StoreEncryptionError, match="decrypt=True"):
        memory.export_jsonl(tmp_path / "e.jsonl")
    assert memory.export_jsonl(tmp_path / "e.jsonl", decrypt=True)["memories"] == 1
    assert "4821" in (tmp_path / "e.jsonl").read_text()
    runtime.shutdown()


def test_recall_filters(tmp_path: Path) -> None:
    runtime = MarkRuntime.local(store_path=tmp_path / "r.db")
    mem = runtime.memory("a")
    old = mem.remember("Deploy step: run the migration script.", occurred_at="2026-01-10", session_id="s1")
    new = mem.remember("Deploy step: warm the cache.", occurred_at="2026-05-02", session_id="s2",
                       tags=["infra"], kind="note")
    ids = lambda **kw: [f.id for f in mem.recall("deploy step", **kw).fragments]  # noqa: E731
    assert set(ids()) == {old, new}
    assert ids(session_id="s1") == [old]
    assert ids(tags=["infra"]) == [new]
    assert ids(kind="note") == [new]
    assert ids(since="2026-03-01") == [new]
    assert ids(until="2026-03-01") == [old]
    assert ids(k=1) and len(ids(k=1)) == 1
    assert ids(order="chronological") == [old, new]
    with pytest.raises(InvalidInputError):
        mem.recall("x", since="last tuesday")
    runtime.shutdown()


def test_get_count_and_isolation(tmp_path: Path) -> None:
    runtime = MarkRuntime.local(store_path=tmp_path / "g.db")
    a, b = runtime.memory("a"), runtime.memory("b")
    fid = a.remember("only in a")
    b.remember("only in b")
    assert a.get(fid).content == "only in a" and b.get(fid) is None
    assert a.count() == 1 and runtime.store.agent_ids() == ["a", "b"]
    runtime.shutdown()
