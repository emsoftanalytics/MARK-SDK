# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 MARK Contributors
"""``remember_many``: a batch write that sets each memory's kind like ``remember`` does.

``store_many_sync`` takes one tag list for the whole batch, so texts are
grouped by kind (explicit ``kind``, else each text's leading label) and each
group is written with its ``kind:<k>`` tag; the reports are merged back into
input order. A batch with a single kind is one ``store_many_sync`` call with
the caller's ``idempotency_key`` unchanged, so its ids match a plain
``store_many_sync`` of the same texts.
"""
from __future__ import annotations

from typing import Any

from mark.kinds import kind_from_text, kind_tag, normalize_kind


def remember_many(memory: Any, texts: Any, *, kind: str | None = None, tags: list[str] | None = None,
                  idempotency_key: str | None = None, **options: Any) -> Any:
    """See :meth:`mark.memory.facade.verbs.VerbsMixin.remember_many`."""
    from mark.memory.facade.write_validation import SkippedItem, StoreReport

    if isinstance(texts, (str, bytes)):
        from mark.errors import InvalidMemoryError

        raise InvalidMemoryError("remember_many expects a list of texts; use remember for one",
                                 reason="not_text")
    items = list(texts)
    base = list(dict.fromkeys(tags or []))
    fixed = normalize_kind(kind)
    if fixed is None and any(t.startswith("kind:") for t in base if isinstance(t, str)):
        fixed = ""  # the caller's own kind tag wins for every item
    groups: dict[str | None, list[int]] = {}
    for index, text in enumerate(items):
        found = fixed if fixed is not None else (kind_from_text(text) if isinstance(text, str) else None)
        groups.setdefault(found or None, []).append(index)
    if not groups:
        groups[None] = []
    occurred = options.pop("occurred_at", None)
    if occurred is not None and len(occurred) != len(items):
        raise ValueError("occurred_at must have one entry per content item")
    ids: list[str | None] = [None] * len(items)
    skipped: list[Any] = []
    existing = 0
    for group_kind, positions in groups.items():
        group_tags = [*base, *([kind_tag(group_kind)] if group_kind and kind_tag(group_kind) not in base else [])]
        key = idempotency_key
        if key is not None and len(groups) > 1:
            key = f"{key}#{group_kind or '-'}"  # each group is its own idempotent batch
        report = memory.store_many_sync(
            [items[i] for i in positions], tags=group_tags or None, idempotency_key=key,
            occurred_at=None if occurred is None else [occurred[i] for i in positions], **options,
        )
        for local, fragment_id in enumerate(report):
            ids[positions[local]] = fragment_id
        skipped += [SkippedItem(positions[item.index], item.reason, item.message) for item in report.skipped]
        existing += report.existing
    return StoreReport(ids, sorted(skipped, key=lambda item: item.index), existing=existing)
