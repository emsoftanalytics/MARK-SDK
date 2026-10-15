# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 MARK Contributors
"""LocalMemoryStore schema v3: retrievable_ids cache invalidation, cross-connection
visibility, TTL expiry, and encrypted-store content hashing.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

from mark.store import LocalMemoryStore
from mark.types import MemoryFragment, MemoryScope, MemoryState, MemoryTier

from _helpers_store_v3 import store


# ── retrievable_ids cache ─────────────────────────────────────────────────────

def test_retrievable_ids_cache_invalidates_on_structural_writes(tmp_path) -> None:
    from mark.store import LocalMemoryStore
    from mark.types import MemoryFragment, MemoryScope, MemoryState, MemoryTier

    store = LocalMemoryStore(tmp_path / "cache.db")

    def frag(text: str, **kw) -> MemoryFragment:
        return MemoryFragment(content=text, scope=MemoryScope.AGENT, tier=MemoryTier.SEMANTIC,
                              agent_id="c", state=MemoryState.UNVERIFIED, **kw)

    a = store.store(frag("alpha"))
    assert store.retrievable_ids("c") == {a}
    b = store.store(frag("bravo"))
    assert store.retrievable_ids("c") == {a, b}

    # Non-structural writes keep the cache; results stay correct.
    store.touch_many([a, b])
    store.update_importance_many({a: 0.9})
    assert store.retrievable_ids("c") == {a, b}

    store.update_state(b, MemoryState.CONTRADICTED)
    assert store.retrievable_ids("c") == {a}
    store.delete(a)
    assert store.retrievable_ids("c") == set()
    store.close()


def test_retrievable_ids_sees_writes_from_another_connection(tmp_path) -> None:
    from mark.store import LocalMemoryStore
    from mark.types import MemoryFragment, MemoryScope, MemoryState, MemoryTier

    path = tmp_path / "shared.db"
    reader = LocalMemoryStore(path)
    writer = LocalMemoryStore(path)

    def frag(text: str) -> MemoryFragment:
        return MemoryFragment(content=text, scope=MemoryScope.AGENT, tier=MemoryTier.SEMANTIC, agent_id="x",
                              state=MemoryState.UNVERIFIED)

    assert reader.retrievable_ids("x") == set()
    fid = writer.store(frag("written elsewhere"))
    assert reader.retrievable_ids("x") == {fid}
    reader.close()
    writer.close()


def test_retrievable_ids_cache_honours_ttl_expiry(tmp_path) -> None:
    import time as _time
    from datetime import datetime, timedelta, timezone

    from mark.store import LocalMemoryStore
    from mark.types import MemoryFragment, MemoryScope, MemoryState, MemoryTier

    store = LocalMemoryStore(tmp_path / "ttl.db")
    created = datetime.now(timezone.utc) - timedelta(seconds=9)
    fid = store.store(MemoryFragment(content="short lived", scope=MemoryScope.AGENT,
                                     tier=MemoryTier.WORKING, agent_id="t",
                                     ttl_seconds=10, created_at=created,
                                     state=MemoryState.UNVERIFIED))
    assert store.retrievable_ids("t") == {fid}
    _time.sleep(1.3)
    assert store.retrievable_ids("t") == set()
    store.close()


def test_encrypted_store_keeps_no_plain_content_hash(tmp_path) -> None:
    import hashlib
    import json
    import sqlite3

    pytest.importorskip("cryptography")
    from mark.security.encryption import FernetEncryptionProvider
    from mark.store import LocalMemoryStore
    from mark.types import MemoryFragment, MemoryScope, MemoryState, MemoryTier

    path = tmp_path / "enc.db"
    store = LocalMemoryStore(path, encryption=FernetEncryptionProvider(passphrase="correct horse"))
    secret = "PIN is 4821"
    fid = store.store(MemoryFragment(content=secret, scope=MemoryScope.AGENT, tier=MemoryTier.SEMANTIC,
                                     agent_id="e", state=MemoryState.UNVERIFIED))
    assert store.get(fid).content == secret
    assert store.get(fid).state == MemoryState.UNVERIFIED  # keyed digest verifies
    store.close()

    raw = sqlite3.connect(path).execute("SELECT metadata FROM fragments WHERE id = ?", (fid,)).fetchone()[0]
    plain = hashlib.sha256(secret.encode()).hexdigest()
    assert plain not in raw
    assert "hmac-sha256:" in json.dumps(json.loads(raw))
