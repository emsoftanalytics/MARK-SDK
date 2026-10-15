# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 MARK Contributors
"""The canonical verbs on the simple API (``mark.memory``).

``mark.memory`` offers exactly the verbs of ``runtime.memory(agent_id)``,
with the same names, arguments, defaults and sync/async convention (plain =
sync, ``a`` prefix = async), so code moves between the two unchanged. Every
verb forwards to the reserved simple agent's
:class:`~mark.memory.mark_memory.MarkMemory`; the only additions are
block-awareness (``remember(block=...)``, ``recall(blocks=[...])``) and the
known-block check that turns a misspelt block name into an error instead of
silently returning nothing.
"""
from __future__ import annotations

import functools
import inspect
from dataclasses import replace
from typing import Any, Callable

from mark.kinds import DEFAULT_CONFIDENCE, DEFAULT_IMPORTANCE

_BLOCK_TAG = "block:"


def _forward(name: str) -> Callable[..., Any]:
    """A method that calls the same-named MarkMemory method (docs and signature copied)."""
    from mark.memory.mark_memory import MarkMemory

    target = getattr(MarkMemory, name)
    if inspect.iscoroutinefunction(target):
        # Async twins stay awaitable functions (inspect/IDEs/frameworks check this).
        @functools.wraps(target)
        async def async_method(self: Any, *args: Any, **kwargs: Any) -> Any:
            return await getattr(self._mark_memory, name)(*args, **kwargs)

        return async_method

    @functools.wraps(target)
    def method(self: Any, *args: Any, **kwargs: Any) -> Any:
        return getattr(self._mark_memory, name)(*args, **kwargs)

    return method


class SimpleVerbsMixin:
    """remember / recall / forget / get / list / ... forwarded to the simple agent."""

    _mark_memory: Any
    _known_blocks: set[str]

    # Forwarded unchanged: identical behaviour on both APIs.
    get = _forward("get")
    list = _forward("list")
    count = _forward("count")
    forget = _forward("forget")
    aforget = _forward("aforget")
    supersede = _forward("supersede")
    asupersede = _forward("asupersede")
    update = _forward("update")
    brief = _forward("brief")
    aupdate = _forward("aupdate")
    resolve = _forward("resolve")
    aresolve = _forward("aresolve")
    timeline = _forward("timeline")
    vacuum = _forward("vacuum")
    export_jsonl = _forward("export_jsonl")
    import_jsonl = _forward("import_jsonl")
    embedder_info = _forward("embedder_info")

    @property
    def facts(self) -> Any:
        """Bi-temporal facts (current values with full history) of the simple agent."""
        return self._mark_memory.facts

    @property
    def documents(self) -> Any:
        """Long documents (chunked, with parent links) of the simple agent."""
        try:
            return self._mark_memory.documents
        except AttributeError:  # pragma: no cover - documents arrive with the a10 integration
            from mark.errors import MarkError

            raise MarkError("document memory is not available in this build of mark-sdk") from None

    def remember(
        self,
        text: str,
        *,
        block: str | None = None,
        kind: str | None = None,
        tags: list[str] | None = None,
        importance: float = DEFAULT_IMPORTANCE,
        confidence: float = DEFAULT_CONFIDENCE,
        **options: Any,
    ) -> str:
        """Store one memory; returns its id. ``block`` files it in a named block.

        Same arguments as :meth:`mark.memory.mark_memory.MarkMemory.remember`
        (``kind``, ``tags``, ``session_id``, ``occurred_at``, ``extract_facts``...).
        """
        all_tags = list(tags or [])
        if block:
            self._known_blocks.add(block)
            all_tags.append(_BLOCK_TAG + block)
        return self._mark_memory.remember(
            text, kind=kind, tags=all_tags or None, importance=importance,
            confidence=confidence, **options,
        )

    async def aremember(self, text: str, **kwargs: Any) -> str:
        """Async :meth:`remember` (same arguments)."""
        return await self._mark_memory._in_executor(self.remember, text, **kwargs)

    def recall(self, query: str, *, blocks: list[str] | None = None, k: int | None = None, **options: Any) -> Any:
        """Search memory; returns a :class:`~mark.intelligence.retrieval.RetrievalResult`.

        ``blocks`` restricts the search to named blocks (each block is searched
        on its own and the results merged by score, so one busy block cannot
        take every slot). Unknown block names raise
        :class:`mark.errors.UnknownBlockError`. Other arguments as for
        :meth:`mark.memory.mark_memory.MarkMemory.recall`.
        """
        if not blocks:
            return self._mark_memory.recall(query, k=k, **options)
        self._check_blocks(blocks)
        tags = list(options.pop("tags", None) or [])
        results = [self._mark_memory.recall(query, k=k, tags=[*tags, _BLOCK_TAG + b], **options)
                   for b in dict.fromkeys(blocks)]
        best: dict[str, tuple[Any, float, float]] = {}
        for result in results:
            relevance = result.relevance if len(result.relevance) == len(result.fragments) else result.scores
            for fragment, score, rel in zip(result.fragments, result.scores, relevance):
                if fragment.id not in best or score > best[fragment.id][1]:
                    best[fragment.id] = (fragment, score, rel)
        ranked = sorted(best.values(), key=lambda item: item[1], reverse=True)[: k or None]
        return replace(results[0], fragments=[f for f, _, _ in ranked],
                       scores=[s for _, s, _ in ranked], relevance=[r for _, _, r in ranked])

    async def arecall(self, query: str, **kwargs: Any) -> Any:
        """Async :meth:`recall` (same arguments)."""
        return await self._mark_memory._in_executor(self.recall, query, **kwargs)

    # ── blocks ────────────────────────────────────────────────────────────────

    def _check_blocks(self, blocks: list[str]) -> None:
        """Raise UnknownBlockError for names never created (``block()``) nor written."""
        unknown = [b for b in blocks if b not in self._known_blocks]
        if not unknown:
            return
        stored = {tag[len(_BLOCK_TAG):] for tag in
                  self._mark_memory._store.tag_values(self._mark_memory.agent_id, _BLOCK_TAG)}
        self._known_blocks |= stored
        unknown = [b for b in unknown if b not in self._known_blocks]
        if unknown:
            from mark.errors import UnknownBlockError

            raise UnknownBlockError(unknown, sorted(self._known_blocks))
