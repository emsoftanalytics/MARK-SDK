"""Encrypted stores: timelines decrypt only sketch candidates after the first one."""
from __future__ import annotations

from pathlib import Path

import pytest

from mark.memory.runtime import MarkRuntime
from mark.store._sqlite.sketches import SKETCH_TABLE, TokenSketcher, sketch_tokens

pytest.importorskip("cryptography")
from mark.security import FernetEncryptionProvider  # noqa: E402


class _CountingFernet(FernetEncryptionProvider):
    """Counts decryptions, so tests can see how much of the store was read."""

    decrypts = 0

    def decrypt(self, data: bytes) -> bytes:
        type(self).decrypts += 1
        return super().decrypt(data)


def _runtime(path: Path) -> MarkRuntime:
    return MarkRuntime.local(store_path=path, encryption=_CountingFernet(passphrase="pw"))


NOISE = [f"svc-{i:03d} was restarted after alarm {i}" for i in range(300)]
TARGET = ["Tomasz Nowak joined the Atlas team.", "Tomasz Nowak moved to the Platform team."]


def test_sketch_tokens_follow_the_anchor_rules() -> None:
    assert sketch_tokens("Atlas_Core went live; Titan's day") == {"atlas", "core", "went", "live", "titan", "s", "day"}
    sketcher = TokenSketcher(b"k" * 32)
    doc = sketcher.sketch("Tomasz Nowak joined")
    assert doc & sketcher.query(["tomasz", "nowak"]) == sketcher.query(["tomasz", "nowak"])
    assert TokenSketcher(b"j" * 32).query(["tomasz"]) != sketcher.query(["tomasz"])  # keyed


def test_repeat_timelines_decrypt_only_candidates(tmp_path: Path) -> None:
    rt = _runtime(tmp_path / "enc.db")
    try:
        mem = rt.memory("t")
        mem.store_many_sync(NOISE + TARGET)
        assert [e.text for e in mem.timeline("Tomasz Nowak")] == TARGET
        _CountingFernet.decrypts = 0
        assert [e.text for e in mem.timeline("Tomasz Nowak")] == TARGET
        assert _CountingFernet.decrypts < 20  # not ~300
        # A new memory is sketched incrementally; an edit re-sketches the row.
        mem.store_sync("Tomasz Nowak now leads Platform.")
        assert len(mem.timeline("Tomasz Nowak")) == 3
        frag = rt.store.get(mem.store_sync("Placeholder note."))
        rt.store.store(frag.model_copy(update={"content": "Tomasz Nowak signed the plan."}))
        assert len(mem.timeline("Tomasz Nowak")) == 4
        assert [e.text for e in mem.timeline("svc-007")] == ["svc-007 was restarted after alarm 7"]
    finally:
        rt.shutdown()


def test_persisted_sketches_survive_restart_and_can_be_dropped(tmp_path: Path) -> None:
    path = tmp_path / "enc.db"
    rt = _runtime(path)
    try:
        rt.store.persist_token_sketches()
        ids = rt.memory("t").store_many_sync(NOISE + TARGET)
        assert len(rt.memory("t").timeline("Tomasz Nowak")) == 2
        assert rt.store._fetchall(f"SELECT count(*) FROM {SKETCH_TABLE}")[0][0] == len(NOISE) + 2
    finally:
        rt.shutdown()
    rt = _runtime(path)
    try:
        _CountingFernet.decrypts = 0
        assert len(rt.memory("t").timeline("Tomasz Nowak")) == 2
        assert _CountingFernet.decrypts < 20  # warm from disk
        rt.store.delete(ids[0])  # a noise memory: its sketch row goes with it
        assert rt.store._fetchall(f"SELECT count(*) FROM {SKETCH_TABLE}")[0][0] == len(NOISE) + 1
        rt.store.persist_token_sketches(False)
        tables = {row[0] for row in rt.store._fetchall("SELECT name FROM sqlite_master")}
        assert SKETCH_TABLE not in tables
        assert len(rt.memory("t").timeline("Tomasz Nowak")) == 2  # in-memory sketches still work
    finally:
        rt.shutdown()


def test_unencrypted_stores_ignore_sketch_persistence(tmp_path: Path) -> None:
    rt = MarkRuntime.local(store_path=tmp_path / "plain.db")
    try:
        rt.store.persist_token_sketches()
        tables = {row[0] for row in rt.store._fetchall("SELECT name FROM sqlite_master")}
        assert SKETCH_TABLE not in tables
    finally:
        rt.shutdown()
