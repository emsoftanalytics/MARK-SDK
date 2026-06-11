"""Content hashing for fragment integrity checks."""
from __future__ import annotations

import hashlib

from mark.types import MemoryFragment


class ContentHasher:
    """Content integrity helper for local memory fragments."""

    ALGORITHM = "sha256"
    METADATA_KEY = "content_hash"

    @staticmethod
    def hash(content: str) -> str:
        """Return the SHA-256 hex digest of the content."""
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    @staticmethod
    def verify(content: str, expected_hash: str) -> bool:
        """Return True when the content matches the expected hash."""
        return ContentHasher.hash(content) == expected_hash

    @staticmethod
    def attach(fragment: MemoryFragment) -> MemoryFragment:
        """Attach to a backing store."""
        metadata = dict(fragment.metadata)
        metadata[ContentHasher.METADATA_KEY] = ContentHasher.hash(fragment.content)
        return fragment.model_copy(update={"metadata": metadata})

    @staticmethod
    def check(fragment: MemoryFragment) -> bool:
        """Check the content and return the gate outcome."""
        expected = fragment.metadata.get(ContentHasher.METADATA_KEY)
        if not isinstance(expected, str):
            return True
        return ContentHasher.verify(fragment.content, expected)
