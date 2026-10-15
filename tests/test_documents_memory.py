# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 MARK Contributors
"""memory.store_document / memory.documents / memory.remember_file.

A document is stored as chunk memories linked to one parent record; recall
finds a detail in the middle of a long document; the parent can be listed,
fetched and forgotten (with its chunks); files are routed by their content.
"""
from __future__ import annotations

import random
from pathlib import Path

import pytest

from mark.documents import DocumentRecord, UnsupportedFileError
from mark.embeddings import HashEmbeddingProvider
from mark.memory.runtime import MarkRuntime
from tests._helpers_media import write_png

NEEDLE = "The replication lag alarm for the Kestrel ledger fires when lag exceeds 742 milliseconds."


def _memory(**kw):
    runtime = MarkRuntime.local(store_path=":memory:", embedder=HashEmbeddingProvider(dim=256), **kw)
    return runtime.memory("docs")


def _long_doc(words: int = 10_000) -> str:
    rnd = random.Random(3)
    parts = ["# Kestrel Platform Reference"]
    vocab = ["ingestion", "gateway", "buffers", "partitions", "tenants", "replicates", "segments", "planner",
             "throttles", "requests", "cache", "window", "timeout", "region", "snapshot", "compaction"]
    count, section = 0, 0
    while count < words:
        section += 1
        parts.append(f"## {section}. Subsystem notes")
        for _ in range(4):
            sentence_list = [" ".join(rnd.choice(vocab) for _ in range(12)).capitalize() + "." for _ in range(5)]
            if section == 12 and not any(NEEDLE in p for p in parts):
                sentence_list.insert(2, NEEDLE)
            paragraph = " ".join(sentence_list)
            parts.append(paragraph)
            count += len(paragraph.split())
    return "\n\n".join(parts)


def test_store_document_chunks_links_and_recalls_the_middle() -> None:
    memory = _memory()
    text = _long_doc()
    record = memory.store_document(text, title="Kestrel reference", chunk_tokens=300, overlap=40)
    assert isinstance(record, DocumentRecord) and record.chunk_count == len(record.chunk_ids) > 20
    assert record.title == "Kestrel reference" and record.word_count >= 10_000
    chunk = memory._store.get(record.chunk_ids[0])
    assert chunk.metadata["document_id"] == record.id and chunk.metadata["source"] == "document"
    result = memory.retrieve_sync("When does the Kestrel replication lag alarm fire?")
    assert NEEDLE in result.fragments[0].content
    assert "Kestrel Platform Reference › 12. Subsystem notes" in result.fragments[0].content


def test_documents_view_lists_gets_and_forgets() -> None:
    memory = _memory()
    first = memory.store_document("# A\n\nAlpha text.", title="A")
    second = memory.store_document("Beta text about billing.", title="B", metadata={"team": "ops"})
    assert [d.title for d in memory.documents.list()] == ["B", "A"]
    assert memory.documents.count() == 2
    fetched = memory.documents.get(second.id)
    assert fetched.metadata["team"] == "ops" and fetched.text == "Beta text about billing."
    assert [c.id for c in memory.documents.chunks(second)] == list(second.chunk_ids)
    assert memory.documents.forget(second) == len(second.chunk_ids)
    assert memory.documents.get(second.id) is None
    assert all(memory._store.get(cid) is None for cid in second.chunk_ids)
    assert [d.id for d in memory.documents.list()] == [first.id]


def test_storing_the_same_document_twice_is_idempotent() -> None:
    memory = _memory()
    one = memory.store_document("Same content.", title="x")
    two = memory.store_document("Same content.", title="x")
    assert one.id == two.id and memory.documents.count() == 1


def test_store_document_from_path_and_facts(tmp_path: Path) -> None:
    memory = _memory()
    path = tmp_path / "service.md"
    path.write_text("# billing-api\n\n- owner: Team Otter\n- port: 8443\n\nbilling-api runs in eu-west-1.\n")
    record = memory.store_document(path, extract_facts=True)
    assert record.title == "service.md" and record.source == str(path.resolve())
    assert record.facts_extracted >= 3
    assert memory.facts.get("billing-api", "port").value == "8443"


def test_unknown_binary_files_are_refused(tmp_path: Path) -> None:
    memory = _memory()
    blob = tmp_path / "data.bin"
    blob.write_bytes(b"\x00\x01\x02binary\x00")
    with pytest.raises(UnsupportedFileError):
        memory.store_document(blob)
    with pytest.raises(UnsupportedFileError):
        memory.remember_file(blob)


def test_remember_file_routes_by_content(tmp_path: Path) -> None:
    memory = _memory()
    note = tmp_path / "note.txt"
    note.write_text("Dana owns the ledger-sync service.")
    assert isinstance(memory.remember_file(note), DocumentRecord)
    png = write_png(tmp_path / "frame.png", (255, 0, 0))
    image = memory.remember_file(png, caption="red frame")
    assert image.caption == "red frame"
    renamed = tmp_path / "actually_text.png"
    renamed.write_text("plain words, not pixels")
    assert isinstance(memory.remember_file(renamed), DocumentRecord)


def test_audio_needs_a_transcriber(tmp_path: Path) -> None:
    from mark.documents import TranscriberRequiredError

    memory = _memory()
    wav = tmp_path / "standup.wav"
    wav.write_bytes(b"RIFF\x24\x00\x00\x00WAVEfmt " + b"\x00" * 32)
    with pytest.raises(TranscriberRequiredError):
        memory.remember_file(wav)

    class Fake:
        def transcribe(self, path: str) -> str:
            return "Standup: Dana moved the deploy to Friday. ledger-sync runs on port 7000."

    record = memory.remember_file(wav, transcriber=Fake(), extract_facts=True)
    assert record.metadata["modality"] == "audio" and record.metadata["media_path"] == str(wav.resolve())
    assert memory.facts.get("ledger-sync", "port").value == "7000"


def test_document_text_title_and_source_are_encrypted_at_rest(tmp_path: Path) -> None:
    import sqlite3

    pytest.importorskip("cryptography")
    from mark.security import FernetEncryptionProvider

    db = tmp_path / "enc.db"
    runtime = MarkRuntime.local(store_path=db, embedder=HashEmbeddingProvider(dim=32),
                                encryption=FernetEncryptionProvider(passphrase="pw"))
    memory = runtime.memory("enc")
    note = tmp_path / "secret-plan.md"
    note.write_text("# Merger plan\n\nAcquire Zenith Labs in Q3.")
    record = memory.store_document(note)
    assert memory.documents.get(record.id).text.endswith("Acquire Zenith Labs in Q3.")
    runtime.shutdown()
    raw = sqlite3.connect(db).execute("SELECT title, content, source FROM documents").fetchone()
    assert all("Zenith" not in str(v) and "secret-plan" not in str(v) for v in raw)
