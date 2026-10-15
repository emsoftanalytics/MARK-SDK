"""Content hashing for fragment integrity checks."""
from __future__ import annotations

import hashlib
import hmac
from typing import TYPE_CHECKING

from mark.types import MemoryFragment

if TYPE_CHECKING:
    from mark.security.encryption import EncryptionProvider

_HMAC_PREFIX = "hmac-sha256:"


class ContentHasher:
    """Content integrity helper for local memory fragments.

    Pass the store's :class:`EncryptionProvider` to :meth:`attach` and
    :meth:`check` so encrypted stores record a keyed HMAC
    (``provider.content_digest``) instead of a bare SHA-256 of the plaintext.
    Legacy bare SHA-256 digests still verify.
    """

    ALGORITHM = "sha256"
    METADATA_KEY = "content_hash"

    @staticmethod
    def hash(content: str) -> str:
        """Return the SHA-256 hex digest of the content."""
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    @staticmethod
    def digest(content: str, provider: "EncryptionProvider | None" = None) -> str:
        """Return the provider's content digest (plain SHA-256 without a provider)."""
        if provider is None:
            return ContentHasher.hash(content)
        return provider.content_digest(content)

    @staticmethod
    def verify(
        content: str,
        expected_hash: str,
        provider: "EncryptionProvider | None" = None,
    ) -> bool:
        """Return True when the content matches the expected digest."""
        if expected_hash.startswith(_HMAC_PREFIX):
            if provider is None:
                return False
            actual = provider.content_digest(content)
        else:
            actual = ContentHasher.hash(content)
        return hmac.compare_digest(actual, expected_hash)

    @staticmethod
    def attach(
        fragment: MemoryFragment,
        provider: "EncryptionProvider | None" = None,
    ) -> MemoryFragment:
        """Return a copy with the content digest stored in metadata."""
        metadata = dict(fragment.metadata)
        metadata[ContentHasher.METADATA_KEY] = ContentHasher.digest(fragment.content, provider)
        return fragment.model_copy(update={"metadata": metadata})

    @staticmethod
    def check(
        fragment: MemoryFragment,
        provider: "EncryptionProvider | None" = None,
    ) -> bool:
        """Return True when the fragment's stored digest matches its content."""
        expected = fragment.metadata.get(ContentHasher.METADATA_KEY)
        if not isinstance(expected, str):
            return True
        return ContentHasher.verify(fragment.content, expected, provider)
