# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 MARK Contributors
"""ImageMemory: remember generated frames and keep them consistent with facts.

An image is stored as an ordinary memory fragment:

* content  = caption + subject names, so plain text recall finds it;
* tags     = ``["mark:image", ...]``;
* metadata = ``{modality: "image", path, sha256, width, height, subjects,
  has_caption, linked_facts}``.

Invariants:

* image bytes are never written to SQLite -- only the absolute path and the
  SHA-256, which ``verify()`` uses to detect changed or missing files;
* image vectors (optional, CLIP) live in their own partition
  (:mod:`mark.media.vector_index`) and never mix with text embeddings;
* at remember-time the subjects' *current* facts are linked to the image
  (by fact id): this is what the frame was supposed to show;
* consistency checks never invent observations: an attribute the describer
  did not mention is reported as ``unverified``.
"""
from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mark.media import consistency, search
from mark.media.embeddings import ImageEmbeddingProvider
from mark.media.files import (
    image_size,
    looks_like_image_path,
    reject_binary,
    require_image,
    sha256_file,
)
from mark.media.records import (
    IMAGE_TAG,
    MODALITY,
    ConsistencyReport,
    ImageMatch,
    ImageRecord,
    VerifyResult,
    image_content,
)
from mark.media.vector_index import ImageVectorIndex

logger = logging.getLogger(__name__)

ImageRef = str | Path | ImageRecord
Describer = Callable[[str], Mapping[str, Any] | str | None]
_RESERVED = ("modality", "path", "sha256", "width", "height", "subjects", "has_caption", "linked_facts")


class ImageMemory:
    """Image memories for one agent memory (``memory.images`` or ``ImageMemory(memory)``)."""

    def __init__(
        self,
        memory: Any,
        *,
        embedder: ImageEmbeddingProvider | None = None,
        describer: Describer | None = None,
    ) -> None:
        self._memory = memory
        self.embedder = embedder
        self.describer = describer
        self._index = ImageVectorIndex(memory._store, memory.agent_id)

    # ── writing ───────────────────────────────────────────────────────────────

    def remember(
        self,
        path: str | Path,
        *,
        caption: str | None = None,
        subjects: Iterable[str] = (),
        session_id: str | None = None,
        occurred_at: datetime | str | None = None,
        metadata: Mapping[str, Any] | None = None,
        tags: Sequence[str] = (),
        link_facts: bool = True,
    ) -> ImageRecord:
        """Remember an image file by path + hash; returns its record.

        The subjects' current facts are linked automatically unless
        ``link_facts=False``. With an embedder set, the image vector is
        computed now (a failure is logged and retried lazily by ``find``).
        """
        file = Path(path).expanduser().resolve()
        if not file.is_file():
            raise FileNotFoundError(f"image not found: {file}")
        require_image(file)  # content, not the name, decides: a PDF renamed .png is refused
        names = tuple(dict.fromkeys(str(s).strip() for s in subjects if str(s).strip()))
        extra = dict(metadata or {})
        reject_binary(extra)
        clash = sorted(set(extra) & set(_RESERVED))
        if clash:
            raise ValueError(f"metadata keys reserved for image memory: {clash}")
        size = image_size(file)
        meta: dict[str, Any] = {
            **extra,
            "modality": MODALITY,
            "path": str(file),
            "sha256": sha256_file(file),
            "width": size[0] if size else None,
            "height": size[1] if size else None,
            "subjects": list(names),
            "has_caption": bool(caption and caption.strip()),
            "linked_facts": self._current_fact_ids(names) if link_facts else {},
        }
        fragment_id = self._memory.store_sync(
            image_content(caption, names, file.name),
            tags=[IMAGE_TAG, *tags],
            source="image",
            session_id=session_id,
            metadata=meta,
            occurred_at=occurred_at,
        )
        record = self.get(fragment_id)
        if self.embedder is not None:
            try:
                search.index_missing([record], self.embedder, self._index)
            except Exception:
                logger.warning("image embedding failed for %s; will retry on find()", file, exc_info=True)
        return record

    def link(self, image: ImageRef, facts: Iterable[Any]) -> ImageRecord:
        """Link facts to an image as what it is expected to show.

        ``facts`` may contain :class:`~mark.memory.facts.Fact` objects or
        subject names (meaning: that subject's current facts). Later links
        for the same (subject, attribute) replace earlier ones.
        """
        fragment = self._fragment(image)
        meta = dict(fragment.metadata)
        linked = {s: dict(a) for s, a in (meta.get("linked_facts") or {}).items()}
        for item in facts:
            if isinstance(item, str):
                for subject, attrs in self._current_fact_ids([item]).items():
                    linked.setdefault(subject, {}).update(attrs)
            else:
                linked.setdefault(item.subject, {})[item.attribute] = item.id
        meta["linked_facts"] = linked
        updated = fragment.model_copy(update={"metadata": meta, "updated_at": datetime.now(timezone.utc)})
        self._memory._store.store(updated)
        return ImageRecord.from_fragment(updated)

    def forget(self, image: ImageRef) -> bool:
        """Delete the image memory and its vectors (the file is left alone)."""
        record = self.get(image)
        self._index.delete(record.id)
        return bool(self._memory.forget(record.id))

    # ── reading ───────────────────────────────────────────────────────────────

    def get(self, image: ImageRef) -> ImageRecord:
        """Resolve a record, fragment id, or file path (latest memory of it)."""
        return ImageRecord.from_fragment(self._fragment(image))

    def list(self, *, subject: str | None = None, session_id: str | None = None) -> list[ImageRecord]:
        """Image records, oldest first, optionally for one subject or session."""
        fragments = self._memory._store.list_by_agent(
            self._memory.agent_id, tags=[IMAGE_TAG], session_id=session_id, limit=None,
        )
        fragments = sorted(
            (f for f in fragments if (f.metadata or {}).get("modality") == MODALITY),
            key=lambda f: f.created_at,
        )
        records = [ImageRecord.from_fragment(f) for f in fragments]
        if subject is not None:
            want = consistency.normalize(subject)
            records = [r for r in records if want in {consistency.normalize(s) for s in r.subjects}]
        return records

    def verify(self, image: ImageRef | None = None) -> VerifyResult | list[VerifyResult]:
        """Re-hash files: ``ok``, ``changed`` or ``missing`` (all images when None)."""
        if image is None:
            return [self._verify(record) for record in self.list()]
        return self._verify(self.get(image))

    def expected(self, image: ImageRef, subject: str | None = None) -> dict[str, Any]:
        """Values of the facts linked to an image.

        Returns ``{subject: {attribute: value}}``, or just ``{attribute: value}``
        when ``subject`` is given (empty when nothing is linked for it).
        """
        record = self.get(image)
        facts = self._memory.facts
        out: dict[str, dict[str, str]] = {}
        for name, attrs in record.linked_facts.items():
            if subject is not None and consistency.normalize(name) != consistency.normalize(subject):
                continue
            for attribute, fact_id in attrs.items():
                # Resolve by id through the history, so a later value of the
                # same attribute never rewrites what this frame expected.
                value = next((f.value for f in facts.history(name, attribute) if f.id == fact_id), None)
                if value is not None:
                    out.setdefault(name, {})[attribute] = value
        if subject is not None:
            return next(iter(out.values()), {})
        return out

    def find(self, query: str | Path, k: int = 5, *, session_id: str | None = None) -> list[ImageMatch]:
        """Rank images for a text query or an image file (Path or existing image path).

        Uses CLIP similarity when an embedder is set; otherwise text recall
        over captions, or an exact-hash match for image queries.
        """
        is_image = looks_like_image_path(query)
        if is_image and not Path(query).is_file():
            raise FileNotFoundError(f"query image not found: {query}")
        if self.embedder is not None:
            records = self.list(session_id=session_id)
            return search.clip_search(query, is_image, records, self.embedder, self._index, k)
        if is_image:
            return search.sha_search(query, self.list(session_id=session_id), k)
        return search.text_search(self._memory, str(query), k, session_id)

    # ── continuity ────────────────────────────────────────────────────────────

    def check_consistency(
        self,
        image: ImageRef,
        subject: str,
        describe: Describer | None = None,
        *,
        synonyms: consistency.Synonyms = None,
        attributes: Iterable[str] | None = None,
        against: str = "current",
    ) -> ConsistencyReport:
        """Compare what ``describe`` sees in the image with the subject's facts.

        against   ``"current"`` (default): the subject's facts now;
                  ``"linked"``: the facts linked when the image was made.
        Returns matches, mismatches (``{"attribute", "expected", "observed"}``)
        and unverified attributes the describer did not mention.
        """
        describe = describe or self.describer
        if describe is None:
            raise ValueError("check_consistency needs a describer (describe=... or ImageMemory(describer=...))")
        record = self.get(image)
        if against == "current":
            expected = {f.attribute: f.value for f in self._memory.facts.current(subject)}
        elif against == "linked":
            expected = self.expected(record, subject)
        else:
            raise ValueError("against must be 'current' or 'linked'")
        if attributes is not None:
            wanted = {consistency.normalize(a) for a in attributes}
            expected = {a: v for a, v in expected.items() if consistency.normalize(a) in wanted}
        if not Path(record.path).is_file():
            raise FileNotFoundError(f"image file is missing: {record.path}")
        observed = describe(record.path) if expected else {}
        matches, mismatches, unverified = consistency.compare(expected, observed, subject, synonyms=synonyms)
        return ConsistencyReport(record.id, subject, matches, mismatches, unverified)

    # ── internals ─────────────────────────────────────────────────────────────

    def _fragment(self, image: ImageRef) -> Any:
        if isinstance(image, ImageRecord):
            image = image.id
        fragment = self._memory._store.get(str(image))
        if (
            fragment is not None
            and (fragment.metadata or {}).get("modality") == MODALITY
            and fragment.agent_id == self._memory.agent_id
        ):
            return fragment
        # Not an id of ours: treat it as a path and take the latest memory of it.
        target = str(Path(image).expanduser().resolve())
        for record in reversed(self.list()):
            if record.path == target:
                return self._memory._store.get(record.id)
        raise KeyError(f"no image memory for {image!r}")

    def _current_fact_ids(self, subjects: Iterable[str]) -> dict[str, dict[str, str]]:
        try:
            facts = self._memory.facts
        except TypeError:  # store without fact support
            return {}
        linked: dict[str, dict[str, str]] = {}
        for subject in subjects:
            current = facts.current(subject)
            if current:
                linked[subject] = {f.attribute: f.id for f in current}
        return linked

    @staticmethod
    def _verify(record: ImageRecord) -> VerifyResult:
        try:
            actual = sha256_file(record.path)
        except OSError:
            return VerifyResult(record.id, record.path, "missing", record.sha256)
        status = "ok" if actual == record.sha256 else "changed"
        return VerifyResult(record.id, record.path, status, record.sha256, actual)

