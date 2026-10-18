"""Background side of the context watch: anchor detection, prefetch and claim checks.

Threading model: the watch owns at most one worker thread
(``WatchWorker``, ``watch_worker``), so prefetches run one at a time and
in submission order. Shared state (anchors, results, corrections, pending
futures) is only touched under ``_lock``; the retrieval itself runs outside
the lock so ``observe()`` and ``briefing()`` never wait on search. A failed
prefetch is logged and dropped: it must never surface in the agent's turn.

Expects the host class (ContextWatch) to provide ``_memory``, ``_lock``,
``_anchors``, ``_results``, ``_task_results``, ``_corrections``, ``_turn``,
``_pending``, ``_executor``, ``_subjects_cache``, ``ttl_turns`` and
``follow_links``.
"""
from __future__ import annotations

import logging
import re
import time
from concurrent.futures import Future
from dataclasses import replace
from typing import Any

from mark.context.provenance import USER_TAG
from mark.context.watch_changes import _CHANGE_STATEMENT, _STATE_QUESTION, _change_lane, _previous_values  # noqa: F401
from mark.context.watch_lines import trusted
from mark.context.watch_support import _TOKEN_RE, _Anchor, _label_for, _mentions_value

# Same logger name as before the split, so log filtering keeps working.
logger = logging.getLogger("mark.context.watch")


def _confirmed(result: Any, key: str) -> bool:
    """Memory confirms a subject only through a fact or a trusted record naming it.

    The model's own stored replies do not count: otherwise a phrase it
    invented ("Caldo Verde", "mark_memory_middleware") would be "known" as
    soon as its reply was stored, and anchor the next briefings (M7).
    """
    from mark.intelligence.query_understanding import _anchor_norm, _contains_anchor

    if any(_anchor_norm(str(f.subject)) == key for f in getattr(result, "facts", None) or []):
        return True
    return any(trusted(f) and _contains_anchor(_anchor_norm(f.content), key)
               for f in result.fragments)


#: Untrusted results (assistant replies, bare questions) in one top-k that
#: trigger a second, user-statements-only retrieval for that query.
_CROWDED = 3


def _with_user_lane(memory: Any, batch: list[str], results: list[Any]) -> list[Any]:
    """Top up crowded results with user statements (tag ``mark:user``).

    Retrieval ranks by relevance, not provenance: long assistant replies
    repeat the user's words and fill the top-k, pushing the user's own
    statement out (a10 small-model replay: "I moved from Porto to Braga"
    missing behind five replies). For each query whose results hold
    ``_CROWDED`` or more untrusted records, user statements are retrieved
    separately and appended after the trusted results of the first pass.
    """
    crowded = [i for i, r in enumerate(results)
               if sum(not trusted(f) for f in r.fragments) >= _CROWDED]
    if not crowded:
        return results
    try:
        extra = memory.retrieve_many_sync([batch[i] for i in crowded], tags=[USER_TAG])
    except Exception as exc:   # the first pass is still a valid answer
        logger.warning("user-statement retrieval failed: %s", exc)
        return results
    merged = list(results)
    for index, lane in zip(crowded, extra):
        first = merged[index]
        ids = {f.id for f in first.fragments}
        pairs = list(zip(first.fragments, first.scores)) + [
            (f, s) for f, s in zip(lane.fragments, lane.scores) if f.id not in ids]
        merged[index] = replace(first, fragments=[f for f, _ in pairs], scores=[s for _, s in pairs])
    return merged


#: Hyphenated describing words ("4-year-old", "paper-lined", "gluten-free") are
#: not subjects; anchoring them produced "Not in memory: paper-lined" (a10).
_DESCRIPTIVE = re.compile(r"^(\d+|[a-z]+)(-[a-z]+)*-(old|lined|based|free|like|style|sized|long|time|"
                          r"made|aged|year|day|week|month|ish|\w+ed)$")


class _WatchPrefetchMixin:
    """Anchor extraction, background prefetch, link anticipation and stale-value checks."""

    def _extract_anchors(self, text: str) -> list[tuple[str, str]]:
        """(normalised key, label as written) for every anchor and known fact subject in *text*."""
        from mark.intelligence.query_understanding import _anchor_norm, _contains_anchor, query_anchors

        found: dict[str, str] = {}
        for key in query_anchors(text):
            label = _label_for(text, key)
            if not _DESCRIPTIVE.match(label):
                found.setdefault(key, label)
        # Known fact subjects are anchors even when written plainly ("pulse metrics").
        normalised = _anchor_norm(text)
        for subject in self._known_subjects():
            if subject not in found and _contains_anchor(normalised, subject):
                found[subject] = _label_for(text, subject)
        return list(found.items())

    def _known_subjects(self) -> set[str]:
        """Fact subjects in memory, cached for 5 s so observe() stays cheap."""
        stamp, subjects = self._subjects_cache
        if time.monotonic() - stamp < 5.0:
            return subjects
        try:
            subjects = set(self._memory.facts.subjects())
        except Exception:  # stores without facts
            subjects = set()
        self._subjects_cache = (time.monotonic(), subjects)
        return subjects

    def _expire(self, turn: int) -> None:
        """Drop anchors not mentioned for more than ttl_turns turns (caller holds lock)."""
        for key in [k for k, a in self._anchors.items() if turn - a.last_turn > self.ttl_turns]:
            self._anchors.pop(key, None)
            self._results.pop(key, None)

    def refresh(self, *, unknown_only: bool = False) -> None:
        """Re-fetch the active subjects, e.g. after new memories were stored.

        A subject first seen before anything about it was stored stays "Not in
        memory" until it is re-fetched; loops that store each turn call this
        after storing. ``unknown_only`` limits it to subjects reported unknown.
        """
        with self._lock:
            labels = [a.label for a in self._anchors.values()
                      if not unknown_only or a.known is False]
        if labels:
            self._schedule(labels, None)

    def _schedule(self, queries: list[str], task: str | None) -> None:
        """Run a prefetch inline (background=False) or queue it on the single worker."""
        if getattr(self, "_closed", False):
            return
        if self._executor is None:
            self._prefetch(queries, task)
            return
        future = self._executor.submit(self._prefetch, queries, task)
        with self._lock:
            self._pending.add(future)
        future.add_done_callback(self._done)

    def _done(self, future: Future) -> None:
        """Worker-thread callback: forget the future; log (never raise) its failure."""
        with self._lock:
            self._pending.discard(future)
        if future.cancelled():   # dropped by close()
            return
        error = future.exception()
        if error is None:
            return
        from mark.errors import StoreClosedError

        if isinstance(error, StoreClosedError):
            # The host closed the store first (shutdown): not a failure worth a
            # warning, but nothing will work again, so the watch stops itself
            # and its worker thread exits (integration N8, agent N5).
            logger.debug("context watch stopped: %s", error)
            self.close()
            return
        logger.warning("context watch prefetch failed: %s", error)

    def _prefetch(self, queries: list[str], task: str | None) -> None:
        """Retrieve anchors (and the current task) in one batch, then store results under the lock.

        Anchors that turn out to be known may pull in linked subjects, which
        are prefetched right after on the same thread.
        """
        batch = list(queries) + ([task] if task else [])
        if not batch:
            return
        results = _with_user_lane(self._memory, batch, self._memory.retrieve_many_sync(batch))
        from mark.intelligence.query_understanding import _anchor_norm

        links: list[str] = []
        with self._lock:
            for query, result in zip(queries, results):
                key = _anchor_norm(query)
                self._results[key] = result
                anchor = self._anchors.get(key)
                if anchor is not None:
                    anchor.known = key not in result.missing_anchors and _confirmed(result, key)
                    if anchor.known and not anchor.linked and self.follow_links:
                        links.extend(self._links_from(anchor, result))
            if task:
                self._task_results = {task: results[-1]}
        previous = _previous_values(self._memory, results)
        if task:
            changes, pool = _change_lane(self._memory, task, getattr(self, "change_lines", 0))
            with self._lock:
                self._task_changes = {task: changes}
                self._task_pool = {task: pool}
        if previous and hasattr(self, "_previous"):
            with self._lock:
                self._previous.update(previous)
        if links:
            self._prefetch(links, None)

    def _links_from(self, anchor: _Anchor, result: Any) -> list[str]:
        """Subjects named by the anchor's best records: likely next questions (caller holds lock)."""
        from mark.intelligence.query_understanding import _anchor_norm, _contains_anchor

        labels: list[str] = []
        for fragment in [f for f in result.fragments if trusted(f)][:3]:
            if not _contains_anchor(_anchor_norm(fragment.content), anchor.key):
                continue
            for key, label in self._extract_anchors(fragment.content):
                # Only multi-part identifiers (atlas-core, HAL-4433, Mei Lin) are
                # worth anticipating; short codes and numbers (SEV1, 502) are not.
                if " " not in key or not re.search(r"[a-z]", key) or key == anchor.key or key in self._anchors:
                    continue
                if len(labels) >= self.follow_links:
                    break
                self._anchors[key] = _Anchor(key, label, self._turn, self._turn,
                                             last_seq=anchor.last_seq, linked=True)
                labels.append(label)
        return labels

    def _check_claims(self, text: str, keys: list[str]) -> None:
        """Flag values the agent states that memory knows were superseded.

        Not when the user asked about earlier values ("What was Joao's old
        number?"): naming the old value is then the right answer.
        """
        from mark.context.watch_render import _PAST

        if _PAST.search(getattr(self, "_task", "") or ""):
            return
        lowered = text.lower()
        tokens = set(_TOKEN_RE.findall(lowered))
        try:
            facts_view = self._memory.facts
        except Exception:
            return
        for key in keys:
            try:
                current = facts_view.current(key)
            except Exception:
                return
            for fact in current:
                if str(fact.value).lower() in lowered:  # values may be typed (int, ...)
                    self._corrections.pop(f"{key}|{fact.attribute}", None)
                    continue
                history = facts_view.history(fact.subject, fact.attribute)
                stale = [old for old in history[:-1] if _mentions_value(str(old.value), lowered, tokens)]
                if stale:
                    getattr(self, "_correction_turn", {})[f"{key}|{fact.attribute}"] = self._turn
                    self._corrections[f"{key}|{fact.attribute}"] = (
                        f"{fact.subject} {fact.attribute} is {fact.value} now "
                        f"(since {fact.valid_from.date().isoformat()}); "
                        f"{stale[-1].value} is outdated."
                    )
