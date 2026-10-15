"""Keyed token sketches: find mentions in an encrypted store without decrypting it all.

An encrypted store has no FTS index (it would hold plaintext), so a
timeline used to decrypt every memory of the agent. A *sketch* is a small
keyed Bloom filter of a memory's words: 512 bits, two bits per word, the
bit positions taken from HMAC-SHA256 under a key derived from the store's
encryption key. A phrase can only occur in memories whose sketch has every
bit of every phrase word, so only those are decrypted (the caller still
confirms the match on the plaintext; sketches only skip work).

Where sketches live:
  - by default only in process memory: nothing new is written to disk, so
    there is no additional leakage. The first timeline in a process
    decrypts each memory once; later ones decrypt only new or changed
    memories plus the few candidates;
  - with ``persist_token_sketches()`` (opt-in) also in the
    ``fragment_sketches`` table, so a fresh process starts warm.

What a persisted sketch reveals to someone holding the database file but
not the key: roughly how many distinct words each memory has, and which
memories probably share words (shared bits), which over many memories
supports frequency analysis. Words cannot be tested for without the key,
and no plaintext or unkeyed hash is stored.

Staleness: each sketch records the last 16 bytes of the row's ciphertext
(part of the Fernet MAC, new on every encryption), so an edited memory is
re-sketched; rows deleted from ``fragments`` are dropped by a trigger.
"""
from __future__ import annotations

import hashlib
import re
from typing import Callable, Iterable

SKETCH_BITS = 512
_BITS_PER_TOKEN = 2
_WORD = re.compile(r"\w+")
SKETCH_TABLE = "fragment_sketches"
DDL_SKETCHES = f"""
    CREATE TABLE IF NOT EXISTS {SKETCH_TABLE} (
        rowid INTEGER PRIMARY KEY,
        stamp BLOB NOT NULL,
        bits BLOB NOT NULL
    )
"""
DDL_SKETCH_TRIGGER = f"""
    CREATE TRIGGER IF NOT EXISTS {SKETCH_TABLE}_delete AFTER DELETE ON fragments
    BEGIN DELETE FROM {SKETCH_TABLE} WHERE rowid = old.rowid; END
"""
# SQL expression for a row's stamp (see module docstring).
STAMP_SQL = "substr(content, -16)"


def sketch_tokens(text: str) -> set[str]:
    """Words as the timeline anchor check sees them (lowercase, "_" splits words)."""
    return set(_WORD.findall(text.lower().replace("_", " ")))


class TokenSketcher:
    """Computes keyed sketches; one per store key."""

    def __init__(self, key: bytes) -> None:
        self._key = key
        self._cache: dict[str, int] = {}

    @classmethod
    def for_store(cls, digest: Callable[[str], str]) -> "TokenSketcher":
        """Derive the sketch key from the store's keyed content digest."""
        return cls(hashlib.sha256(digest("mark-token-sketch-key-v1").encode("utf-8")).digest()[:32])

    def token_bits(self, token: str) -> int:
        bits = self._cache.get(token)
        if bits is None:
            # Keyed BLAKE2b is a MAC (a PRF under the key) and one C call per word.
            mac = hashlib.blake2b(token.encode("utf-8"), key=self._key, digest_size=8).digest()
            bits = 0
            for i in range(_BITS_PER_TOKEN):
                bits |= 1 << (int.from_bytes(mac[4 * i:4 * i + 4], "little") % SKETCH_BITS)
            if len(self._cache) >= 65_536:
                self._cache.clear()
            self._cache[token] = bits
        return bits

    def sketch(self, text: str) -> int:
        cache = self._cache
        bits = 0
        for token in sketch_tokens(text):
            token_bits = cache.get(token)  # inlined hit path: ~10 words per memory
            bits |= token_bits if token_bits is not None else self.token_bits(token)
        return bits

    def query(self, tokens: Iterable[str]) -> int:
        bits = 0
        for token in tokens:
            bits |= self.token_bits(token)
        return bits


def to_blob(bits: int) -> bytes:
    return bits.to_bytes(SKETCH_BITS // 8, "little")


def from_blob(blob: bytes) -> int:
    return int.from_bytes(blob, "little")
