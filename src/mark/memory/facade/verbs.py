# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 MARK Contributors
"""The canonical memory verbs: remember / recall (+ their ``a``-prefixed async twins).

Naming convention (the same on ``mark.memory`` and ``runtime.memory(id)``):
plain names are synchronous, an ``a`` prefix marks the awaitable twin —
``remember``/``aremember``, ``recall``/``arecall``, ``forget``/``aforget``.
The async twins run the sync path on the runtime executor, so middleware,
governance and redaction are identical. The historical names keep working:
``store_sync`` = ``remember``, ``retrieve_sync`` = ``recall`` (without the
new filters), and ``store``/``retrieve`` stay the async originals.
"""
from __future__ import annotations

import asyncio
import functools
from dataclasses import replace
from datetime import datetime
from typing import Any

from mark.kinds import DEFAULT_CONFIDENCE, DEFAULT_IMPORTANCE, kind_tag, normalize_kind
from mark.memory.facade.core import MemoryCore


def _invalid(exc: Exception, what: str = "memory") -> Exception:
    """A MARK error for a pydantic ValidationError; other errors pass through unchanged."""
    from pydantic import ValidationError

    from mark.errors import InvalidInputError

    if isinstance(exc, ValidationError):
        return InvalidInputError.from_validation(exc, what)
    return exc


def _when(value: datetime | str | None, name: str) -> datetime | None:
    """Parse a since/until bound (ISO date or datetime); naive values are UTC."""
    if value is None or isinstance(value, datetime):
        stamp = value
    else:
        try:
            stamp = datetime.fromisoformat(str(value).strip().replace("Z", "+00:00"))
        except ValueError:
            from mark.errors import InvalidInputError

            raise InvalidInputError(f"{name} must be an ISO date or datetime, got {value!r}") from None
    if stamp is not None and stamp.tzinfo is None:
        from datetime import timezone

        stamp = stamp.replace(tzinfo=timezone.utc)
    return stamp


class VerbsMixin(MemoryCore):
    """remember / recall and the async twins of every lifecycle verb."""

    def remember(
        self,
        text: str,
        *,
        kind: str | None = None,
        tags: list[str] | None = None,
        importance: float = DEFAULT_IMPORTANCE,
        confidence: float = DEFAULT_CONFIDENCE,
        session_id: str | None = None,
        occurred_at: datetime | str | None = None,
        source: str | None = None,
        metadata: dict[str, Any] | None = None,
        extract_facts: bool | None = None,
        **options: Any,
    ) -> str:
        """Store one memory and return its id.

        ``kind`` is one of :data:`mark.kinds.KINDS` (``"decision"``,
        ``"todo"``, ``"bug"``, ...) and is stored as the tag ``kind:<kind>``.
        ``occurred_at`` dates the event (shown in recall, used for "latest"
        questions); ``extract_facts=True`` records facts stated in the text.
        Remaining keywords (``scope``, ``state``, ``ttl_seconds``) are passed
        to :meth:`store_sync`. Invalid input raises
        :class:`mark.errors.InvalidInputError`.
        """
        from mark.errors import InvalidInputError

        if not isinstance(text, str) or not text.strip():
            raise InvalidInputError("memory text must be a non-empty string")
        all_tags = list(dict.fromkeys(tags or []))
        kind = normalize_kind(kind)
        if kind and kind_tag(kind) not in all_tags:
            all_tags.append(kind_tag(kind))
        try:
            return self.store_sync(  # type: ignore[attr-defined]
                text, tags=all_tags or None, importance=importance, confidence=confidence,
                session_id=session_id, occurred_at=occurred_at, source=source,
                metadata=metadata, extract_facts=extract_facts, **options,
            )
        except Exception as exc:
            raise _invalid(exc) from exc

    def recall(
        self,
        query: str,
        *,
        k: int | None = None,
        kind: str | None = None,
        tags: list[str] | None = None,
        session_id: str | None = None,
        session_prefix: str | None = None,
        since: datetime | str | None = None,
        until: datetime | str | None = None,
        policy: Any = None,
        order: str = "relevance",
        **options: Any,
    ) -> Any:
        """Search memory; returns a :class:`~mark.intelligence.retrieval.RetrievalResult`.

        ``k`` caps the memories returned (default: the policy's top-k; a
        ``k`` above 10 switches to the deep policy). ``kind``/``tags`` and
        ``session_id``/``session_prefix`` pre-filter the candidates;
        ``since``/``until`` keep memories whose event time (``occurred_at``,
        a leading ``[YYYY-MM-DD]`` stamp, else the stored time) falls inside
        the bounds. ``order="chronological"`` lists the kept results oldest
        first. Other keywords go to :meth:`retrieve_sync`.
        """
        from mark.intelligence.retrieval import RetrievalPolicy
        from mark.memory.timeline_order import ORDERS, apply_order

        if order not in ORDERS:
            from mark.errors import InvalidInputError

            raise InvalidInputError(f"order must be one of {ORDERS}, got {order!r}")
        if k is not None and int(k) < 1:
            from mark.errors import InvalidInputError

            raise InvalidInputError(f"k must be at least 1, got {k!r}")
        start, end = _when(since, "since"), _when(until, "until")
        wanted = list(tags or [])
        kind = normalize_kind(kind)
        if kind:
            wanted.append(kind_tag(kind))
        if policy is None:
            policy = RetrievalPolicy.DEEP if k is not None and k > 10 else RetrievalPolicy.BALANCED
        result = self.retrieve_sync(  # type: ignore[attr-defined]
            query, policy, session_id=session_id, session_prefix=session_prefix,
            tags=wanted or None, **options,
        )
        keep = list(range(len(result.fragments)))
        if start is not None or end is not None:
            # Facade-side time window: exact for what the pipeline returned.
            # TODO(a10 integration): prefer the pipeline's since/until (W4)
            # so the window filters candidates before top-k is taken.
            from mark.intelligence.temporal_order import event_time

            keep = [i for i in keep
                    if (start is None or event_time(result.fragments[i]) >= start)
                    and (end is None or event_time(result.fragments[i]) <= end)]
        if k is not None:
            keep = keep[: int(k)]
        if len(keep) != len(result.fragments):
            result = _select(result, keep)
        return apply_order(result, order)

    # ── async twins (a-prefix) ────────────────────────────────────────────────

    async def _in_executor(self, fn: Any, *args: Any, **kwargs: Any) -> Any:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, functools.partial(fn, *args, **kwargs))

    async def aremember(self, text: str, **kwargs: Any) -> str:
        """Async :meth:`remember` (same arguments)."""
        return await self._in_executor(self.remember, text, **kwargs)

    async def arecall(self, query: str, **kwargs: Any) -> Any:
        """Async :meth:`recall` (same arguments)."""
        return await self._in_executor(self.recall, query, **kwargs)

    async def aforget(self, memory_id: str | None = None, **kwargs: Any) -> int:
        """Async :meth:`forget` (same arguments)."""
        return await self._in_executor(self.forget, memory_id, **kwargs)  # type: ignore[attr-defined]

    async def asupersede(self, memory_id: str, text: str, **kwargs: Any) -> str:
        """Async :meth:`supersede` (same arguments)."""
        return await self._in_executor(self.supersede, memory_id, text, **kwargs)  # type: ignore[attr-defined]

    async def aupdate(self, memory_id: str, text: str, **kwargs: Any) -> str:
        """Async :meth:`update` (same arguments)."""
        return await self._in_executor(self.update, memory_id, text, **kwargs)  # type: ignore[attr-defined]

    async def aresolve(self, memory_id: str, **kwargs: Any) -> str:
        """Async :meth:`resolve` (same arguments)."""
        return await self._in_executor(self.resolve, memory_id, **kwargs)  # type: ignore[attr-defined]


def _select(result: Any, keep: list[int]) -> Any:
    """Keep only the fragments at *keep* (scores and relevances stay aligned)."""

    def pick(values: list) -> list:
        return [values[i] for i in keep] if len(values) == len(result.fragments) else list(values)

    return replace(
        result,
        fragments=[result.fragments[i] for i in keep],
        scores=pick(result.scores),
        relevance=pick(result.relevance),
    )
