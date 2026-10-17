# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 MARK Contributors
"""Value types returned by :class:`mark.media.ImageMemory`.

An :class:`ImageRecord` is a read-only view over one memory fragment whose
``metadata["modality"] == "image"``. The fragment is the source of truth;
records are rebuilt from it on every read, so they never drift from storage.

Privacy invariant: text that a store may encrypt at rest (the caption, fact
values) is kept in the fragment *content* or the facts table, never copied
into plaintext metadata. Metadata holds only the path, hash, size, subject
names and the *ids* of linked facts.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from mark.types import MemoryFragment

#: Tag carried by every image fragment; lets listing and recall filter cheaply.
IMAGE_TAG = "mark:image"
MODALITY = "image"
#: Separates the caption from the subject line inside the fragment content.
SUBJECTS_PREFIX = "\nSubjects: "


def image_content(caption: str | None, subjects: tuple[str, ...], filename: str) -> str:
    """The searchable text of an image memory: caption, then its subjects."""
    head = caption.strip() if caption and caption.strip() else f"Image {filename}"
    return head + (SUBJECTS_PREFIX + ", ".join(subjects) if subjects else "")


@dataclass(frozen=True)
class ImageRecord:
    """One remembered image: where it lives, its hash, and what it shows.

    ``linked_facts`` maps subject -> {attribute: fact id}: the facts that
    were current when the image was remembered (or linked later), i.e. what
    the image *should* show. Resolve values with ``ImageMemory.expected()``.
    """
    id: str
    path: str
    sha256: str
    caption: str | None = None
    subjects: tuple[str, ...] = ()
    width: int | None = None
    height: int | None = None
    session_id: str | None = None
    linked_facts: dict[str, dict[str, str]] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_fragment(cls, fragment: MemoryFragment) -> ImageRecord:
        """Build a record from an image fragment (raises if it is not one)."""
        meta = dict(fragment.metadata or {})
        if meta.get("modality") != MODALITY:
            raise ValueError(f"fragment {fragment.id!r} is not an image memory")
        caption = None
        if meta.get("has_caption"):
            caption = fragment.content.split(SUBJECTS_PREFIX, 1)[0].strip() or None
        return cls(
            id=fragment.id,
            path=str(meta.get("path", "")),
            sha256=str(meta.get("sha256", "")),
            caption=caption,
            subjects=tuple(meta.get("subjects") or ()),
            width=meta.get("width"),
            height=meta.get("height"),
            session_id=fragment.session_id,
            linked_facts={str(k): dict(v) for k, v in (meta.get("linked_facts") or {}).items()},
            metadata=meta,
        )


@dataclass(frozen=True)
class VerifyResult:
    """Outcome of re-hashing an image file: ``ok``, ``changed`` or ``missing``."""
    id: str
    path: str
    status: str
    expected_sha256: str
    actual_sha256: str | None = None

    @property
    def ok(self) -> bool:
        """True when the file exists and still has the remembered hash."""
        return self.status == "ok"


@dataclass(frozen=True)
class ImageMatch:
    """One search hit: the record and its score (higher is better)."""
    record: ImageRecord
    score: float
    method: str  # "clip", "text" or "sha256"


@dataclass(frozen=True)
class ConsistencyReport:
    """Attribute-by-attribute comparison of an image with a subject's facts.

    Each entry is ``{"attribute", "expected", "observed", "status"}`` with
    status ``match``, ``mismatch`` or ``unverified``. ``unverified`` means the
    describer said nothing usable about that attribute; it is never counted
    as a match.
    """
    image_id: str
    subject: str
    matches: list[dict[str, Any]] = field(default_factory=list)
    mismatches: list[dict[str, Any]] = field(default_factory=list)
    unverified: list[dict[str, Any]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """No attribute contradicts the facts (unverified ones may remain)."""
        return not self.mismatches

    @property
    def verified(self) -> bool:
        """Every expected attribute was observed and matched."""
        return not self.mismatches and not self.unverified

    def as_dict(self) -> dict[str, Any]:
        """Plain-dict form for logs, JSON or prompts."""
        return {
            "image_id": self.image_id,
            "subject": self.subject,
            "matches": list(self.matches),
            "mismatches": list(self.mismatches),
            "unverified": list(self.unverified),
        }
