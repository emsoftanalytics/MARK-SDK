# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 Somtochukwu Owen Ikeanyi
"""Image memory: remember/verify by path + hash, recall, search, vector partition.

All offline: PNGs come from zlib/struct and the CLIP model is a fake.
"""
from __future__ import annotations

import json
import sqlite3

import pytest
from _helpers_media import FakeClip, memory, runtime, write_png  # noqa: F401 - fixtures

from mark.media import ImageEmbeddingProvider, ImageMemory, ImageRecord
from mark.media.files import image_size, sha256_file


def _walk(value):
    """Yield every scalar inside nested dicts/lists."""
    if isinstance(value, dict):
        for item in value.values():
            yield from _walk(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _walk(item)
    else:
        yield value


def test_remember_stores_path_hash_and_size_not_bytes(memory, tmp_path):
    png = write_png(tmp_path / "frame.png", (200, 10, 10), width=3, height=5)
    record = memory.images.remember(png, caption="Elena at the station", subjects=["Elena"])
    assert isinstance(record, ImageRecord)
    assert record.path == str(png.resolve()) and record.sha256 == sha256_file(png)
    assert (record.width, record.height) == (3, 5)
    assert record.caption == "Elena at the station" and record.subjects == ("Elena",)
    fragment = memory._store.get(record.id)
    assert fragment.metadata["modality"] == "image"
    assert "Elena at the station" in fragment.content and "Elena" in fragment.content
    assert "mark:image" in fragment.tags
    # Never raw bytes: no bytes objects and no PNG signature anywhere in metadata.
    assert not any(isinstance(v, (bytes, bytearray)) for v in _walk(fragment.metadata))
    assert "PNG" not in json.dumps(fragment.metadata)
    # Caption text lives in (possibly encrypted) content, not plaintext metadata.
    assert "station" not in json.dumps(fragment.metadata)


def test_image_bytes_never_reach_sqlite(memory, runtime, tmp_path):
    png = write_png(tmp_path / "frame.png", (1, 2, 3))
    memory.images.remember(png, caption="a frame")
    payload = png.read_bytes()
    conn = sqlite3.connect(str(tmp_path / "mem.db"))
    try:
        for (table,) in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall():
            for row in conn.execute(f'SELECT * FROM "{table}"').fetchall():
                for cell in row:
                    blob = cell if isinstance(cell, bytes) else str(cell).encode()
                    assert payload not in blob, f"image bytes found in {table}"
    finally:
        conn.close()


def test_metadata_with_raw_bytes_or_reserved_keys_is_rejected(memory, tmp_path):
    png = write_png(tmp_path / "frame.png", (1, 2, 3))
    with pytest.raises(TypeError, match="raw bytes"):
        memory.images.remember(png, metadata={"thumb": {"data": b"\x89PNG"}})
    with pytest.raises(ValueError, match="reserved"):
        memory.images.remember(png, metadata={"sha256": "forged"})
    with pytest.raises(FileNotFoundError):
        memory.images.remember(tmp_path / "missing.png")
    record = memory.images.remember(png, metadata={"scene": 4, "seed": 1234})
    assert record.metadata["scene"] == 4 and record.metadata["seed"] == 1234


def test_verify_detects_changed_and_missing_files(memory, tmp_path):
    a = write_png(tmp_path / "a.png", (255, 0, 0))
    b = write_png(tmp_path / "b.png", (0, 255, 0))
    ra, rb = memory.images.remember(a), memory.images.remember(b)
    assert [r.status for r in memory.images.verify()] == ["ok", "ok"]
    write_png(a, (0, 0, 255))  # regenerated frame: same path, new pixels
    b.unlink()
    changed, missing = memory.images.verify(ra), memory.images.verify(rb)
    assert changed.status == "changed" and changed.actual_sha256 == sha256_file(a)
    assert missing.status == "missing" and not missing.ok
    # Lookup by path resolves to the latest memory of that file.
    assert memory.images.get(a).id == ra.id


def test_caption_recall_and_fallback_find_without_clip(memory, tmp_path):
    memory.store_sync("Unrelated note about invoices and quarterly taxes")
    station = memory.images.remember(
        write_png(tmp_path / "s.png", (9, 9, 9)), caption="Elena waits on the rainy train platform",
        subjects=["Elena"],
    )
    memory.images.remember(write_png(tmp_path / "m.png", (8, 8, 8)), caption="Marco cooks pasta in the kitchen")
    # Plain MARK recall sees image captions like any other memory.
    result = memory.retrieve_sync("rainy train platform")
    assert result.fragments and result.fragments[0].id == station.id
    # find() without an embedder falls back to text recall over images only.
    hits = memory.images.find("train platform", k=3)
    assert hits[0].record.id == station.id and hits[0].method == "text"
    assert all(h.record.metadata["modality"] == "image" for h in hits)
    # An image query without CLIP can only match the exact same file contents.
    copy = tmp_path / "copy.png"
    copy.write_bytes((tmp_path / "s.png").read_bytes())
    exact = memory.images.find(copy)
    assert [h.record.id for h in exact] == [station.id] and exact[0].method == "sha256"
    assert memory.images.find(write_png(tmp_path / "new.png", (1, 1, 1))) == []


def test_list_filters_by_subject_and_session(memory, tmp_path):
    memory.images.remember(write_png(tmp_path / "1.png", (1, 0, 0)), subjects=["Elena"], session_id="ep1")
    memory.images.remember(write_png(tmp_path / "2.png", (2, 0, 0)), subjects=["Marco"], session_id="ep1")
    memory.images.remember(write_png(tmp_path / "3.png", (3, 0, 0)), subjects=["elena"], session_id="ep2")
    assert [r.path.rsplit("/", 1)[-1] for r in memory.images.list(subject="ELENA")] == ["1.png", "3.png"]
    assert len(memory.images.list(session_id="ep1")) == 2
    assert memory.images.list(subject="Nobody") == []


def test_find_with_fake_clip_text_and_image_queries(memory, tmp_path):
    clip = FakeClip()
    assert isinstance(clip, ImageEmbeddingProvider)
    images = ImageMemory(memory, embedder=clip)
    red = images.remember(write_png(tmp_path / "red.png", (250, 0, 0)), caption="frame one")
    blue = images.remember(write_png(tmp_path / "blue.png", (0, 0, 250)), caption="frame two")
    assert [h.record.id for h in images.find("a blue scene", k=2)] == [blue.id, red.id]
    hits = images.find(write_png(tmp_path / "q.png", (240, 5, 5)), k=1)
    assert hits[0].record.id == red.id and hits[0].method == "clip" and hits[0].score > 0.99
    # Vectors were stored at remember-time; find() does not re-embed them.
    embedded = [c for c in clip.image_calls if c.endswith(("red.png", "blue.png"))]
    assert len(embedded) == 2


def test_images_remembered_before_clip_are_indexed_lazily_if_unchanged(memory, tmp_path):
    kept = memory.images.remember(write_png(tmp_path / "g.png", (0, 250, 0)))
    edited = memory.images.remember(write_png(tmp_path / "e.png", (0, 240, 0)))
    write_png(tmp_path / "e.png", (250, 0, 0))  # changed after remembering
    images = ImageMemory(memory, embedder=FakeClip())
    hits = images.find("green", k=5)
    # The edited file no longer shows what was remembered, so it is not indexed.
    assert [h.record.id for h in hits] == [kept.id]
    assert edited.id not in {h.record.id for h in hits}


def test_image_vectors_live_in_their_own_partition(memory, tmp_path):
    images = ImageMemory(memory, embedder=FakeClip())
    record = images.remember(write_png(tmp_path / "p.png", (10, 20, 30)), caption="pixel")
    text_vector = memory._store.get(record.id).embedding
    image_vectors = images._index.get_all(FakeClip.fingerprint)
    assert set(image_vectors) == {record.id}
    sha, vector = image_vectors[record.id]
    assert sha == record.sha256 and len(vector) == 3
    # The fragment keeps its text embedding; the CLIP vector never replaces it.
    assert text_vector is None or len(text_vector) != 3
    assert images._index.get_all("other-model") == {}
    assert images.forget(record) is True
    assert images._index.get_all(FakeClip.fingerprint) == {}


def test_image_size_is_optional_and_never_raises(tmp_path):
    junk = tmp_path / "x.webp"
    junk.write_bytes(b"not an image")
    assert image_size(junk) is None
    assert image_size(tmp_path / "missing.png") is None
    gif = tmp_path / "x.gif"
    gif.write_bytes(b"GIF89a" + (7).to_bytes(2, "little") + (4).to_bytes(2, "little") + b"\0" * 8)
    assert image_size(gif) == (7, 4)
