# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 Somtochukwu Owen Ikeanyi
"""Graph documents stored with store.store_document() are encrypted at rest in an encrypted store."""
from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("cryptography")

from mark.security import FernetEncryptionProvider  # noqa: E402
from mark.store.sqlite import LocalMemoryStore  # noqa: E402
from mark.types import MemoryDocument  # noqa: E402


def _bytes(folder: Path) -> bytes:
    return b"".join(p.read_bytes() for p in folder.glob("g.db*") if p.is_file())


def test_encrypted_store_keeps_graph_document_text_encrypted(tmp_path: Path) -> None:
    key = FernetEncryptionProvider.generate_key() if hasattr(FernetEncryptionProvider, "generate_key") else None
    if key is None:
        from cryptography.fernet import Fernet

        key = Fernet.generate_key().decode()
    store = LocalMemoryStore(tmp_path / "g.db", encryption=FernetEncryptionProvider(key=key))
    doc = MemoryDocument(title="Vault handbook qzx-title", content="The vault opens at dawn qzx-body.",
                         source="handbook-qzx.md", agent_id="a")
    store.store_document(doc)
    back = store.get_document(doc.id)
    assert (back.title, back.content, back.source) == (doc.title, doc.content, doc.source)
    assert "mark_encrypted_text" not in back.metadata
    store.close()
    data = _bytes(tmp_path)
    assert b"qzx-title" not in data and b"qzx-body" not in data and b"handbook-qzx" not in data


def test_plain_store_documents_are_unchanged(tmp_path: Path) -> None:
    store = LocalMemoryStore(tmp_path / "g.db")
    doc = MemoryDocument(title="T", content="plain body", agent_id="a")
    store.store_document(doc)
    assert store.get_document(doc.id).content == "plain body"
    store.close()
