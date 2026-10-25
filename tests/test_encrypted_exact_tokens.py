"""m8: encrypted stores rank exact identifier matches first and warm their sketches.

Without FTS (it would hold plaintext) an encrypted store only had the
vector channel, so "uniq5" ranked "uniq5911" first. Identifier-like query
tokens (with a digit) now get a lexical channel from the keyed token
sketches with whole-token confirmation on the decrypted candidates. The
sketches are built once per process: at write time for new memories, and
by ``warm_up(sketches=True)`` (``mark serve``) for the rest.
"""
from __future__ import annotations

import pytest

pytest.importorskip("cryptography")

from mark.memory.runtime import MarkRuntime  # noqa: E402
from mark.security import FernetEncryptionProvider  # noqa: E402

AGENT = "a"


@pytest.fixture
def encrypted(tmp_path):
    key = FernetEncryptionProvider(ephemeral=True)
    runtime = MarkRuntime.local(store_path=tmp_path / "e.db", encryption=key)
    texts = [f"uniq{i} proj{i % 7} status update: build {i % 97} deployed" for i in range(1200)]
    runtime.memory(AGENT).store_many_sync(texts)
    yield runtime, key, tmp_path / "e.db"
    runtime.shutdown()


def test_exact_identifier_ranks_first(encrypted) -> None:
    runtime, *_ = encrypted
    memory = runtime.memory(AGENT)
    for query in ("uniq5", "uniq50", "uniq7", "what happened in uniq120?"):
        token = query.strip("?").split()[-1]
        top = memory.retrieve_sync(query).fragments[0].content
        assert top.split()[0] == token, (query, top)


def test_lexical_channel_only_returns_whole_token_matches(encrypted) -> None:
    runtime, *_ = encrypted
    store = runtime.store
    hits = store.lexical_search(AGENT, "uniq5", limit=10)
    contents = [store.get(fid).content for fid, _ in hits]
    assert contents and all(text.split()[0] == "uniq5" for text in contents)
    assert store.lexical_search(AGENT, "status update deployed", limit=10) == []  # words: vectors answer


def test_warm_up_builds_sketches_and_new_writes_are_sketched_at_write_time(encrypted, monkeypatch) -> None:
    runtime, key, path = encrypted
    runtime.shutdown()
    runtime = MarkRuntime.local(store_path=path, encryption=key)
    runtime.warm_up(sketches=True)
    store = runtime.store
    decrypted: list[int] = []
    original = store._sketch_rows
    monkeypatch.setattr(store, "_sketch_rows", lambda sketcher, rowids: decrypted.append(len(rowids)) or original(sketcher, rowids))
    runtime.memory(AGENT).store_sync("uniq99999 shipped the hotfix")
    assert runtime.memory(AGENT).timeline("proj3")  # no first-timeline decrypt pass
    assert runtime.memory(AGENT).retrieve_sync("uniq99999").fragments[0].content.startswith("uniq99999")
    assert sum(decrypted) == 0
    runtime.shutdown()
