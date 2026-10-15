"""Context watch: anticipate what an agent needs and have it ready before it asks.

A :class:`ContextWatch` follows the agent's stream — user messages, its own
outputs, tool calls and tool results — and, without the agent issuing any
search, keeps a small working set of relevant facts and memories warm:

1. **Observe.** Every event is scanned for anchors (service names, ticket ids,
   people, known fact subjects). New anchors trigger a background prefetch,
   so retrieval happens while the agent is still thinking or a tool is
   running, not on the critical path of the next model call.
2. **Anticipate.** The working set is ranked by how recently and how often
   each anchor came up; anchors that fall out of the conversation decay away.
3. **Brief.** :meth:`ContextWatch.briefing` returns a compact, LLM-friendly
   block grouped by subject — current facts first, then the most relevant
   records — within a character budget. It never blocks on search unless the
   caller asks it to wait.
4. **Correct.** When the agent states a value that memory knows was
   superseded ("atlas-core listens on 9239" after it moved to 4444), the next
   briefing leads with a correction. Only values found in a fact's history
   trigger this, so the agent is never "corrected" on a guess.

Unknown anchors are reported as "Not in memory", so the agent can say it
doesn't know instead of improvising.

    watch = ContextWatch(mark.runtime.memory("__mark__"))
    watch.observe("Can you check why atlas-core is timing out?", role="user")
    ...
    system_prompt += watch.briefing().text   # before every model call
    watch.observe(model_output, role="assistant")
"""
from __future__ import annotations

import logging
import re
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("mark.context.watch")

_ROLES = {"user", "assistant", "tool", "system"}
_TOKEN_RE = re.compile(r"\w+")


@dataclass
class _Anchor:
    key: str                    # normalised ("atlas core")
    label: str                  # as first seen ("atlas-core")
    first_turn: int
    last_turn: int
    last_seq: int = 0           # event sequence of the latest mention (recency order)
    mentions: int = 1
    known: bool | None = None   # None until a prefetch has answered
    linked: bool = False        # anticipated via a link, not mentioned by anyone yet


@dataclass(frozen=True)
class Briefing:
    """What the watch would put in front of the model right now."""
    text: str
    anchors: list[str] = field(default_factory=list)
    corrections: list[str] = field(default_factory=list)
    unknown: list[str] = field(default_factory=list)
    memory_ids: list[str] = field(default_factory=list)
    pending: int = 0            # prefetches still running (briefing may be incomplete)
    latency_ms: float = 0.0

    def __bool__(self) -> bool:
        return bool(self.text)


class ContextWatch:
    """Follows an agent's stream and keeps anticipated context ready.

    memory          a ``MarkMemory`` (``runtime.memory(agent_id)``)
    budget_chars    maximum briefing size
    max_anchors     how many active subjects a briefing covers
    per_anchor      memories shown per subject
    ttl_turns       turns an anchor stays active without being mentioned
    background      prefetch on a worker thread (False: inline, deterministic)
    """

    def __init__(
        self,
        memory: Any,
        *,
        budget_chars: int = 1800,
        max_anchors: int = 6,
        per_anchor: int = 3,
        ttl_turns: int = 6,
        background: bool = True,
    ) -> None:
        self._memory = memory
        self.budget_chars = budget_chars
        self.max_anchors = max_anchors
        self.per_anchor = per_anchor
        self.ttl_turns = ttl_turns
        self._lock = threading.RLock()
        self._anchors: dict[str, _Anchor] = {}
        self._results: dict[str, Any] = {}          # anchor key -> RetrievalResult
        self._task_results: dict[str, Any] = {}     # task text -> RetrievalResult
        self._task: str = ""
        self._corrections: dict[str, str] = {}      # correction key -> text
        self._turn = 0
        self._seq = 0
        self.follow_links = 3   # linked subjects prefetched per new anchor (0 disables)
        self._pending: set[Future] = set()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mark-watch") if background else None
        self._subjects_cache: tuple[float, set[str]] = (0.0, set())

    # ── stream input ──────────────────────────────────────────────────────────

    def observe(self, text: str, *, role: str = "user") -> list[str]:
        """Feed one event from the agent's stream. Returns newly seen anchors."""
        if role not in _ROLES:
            raise ValueError(f"role must be one of {sorted(_ROLES)}")
        text = (text or "").strip()
        if not text:
            return []
        found = self._extract_anchors(text)
        with self._lock:
            if role == "user":
                self._turn += 1
                self._task = text
            turn = self._turn
            self._seq += 1
            new: list[str] = []
            for key, label in found:
                anchor = self._anchors.get(key)
                if anchor is None:
                    self._anchors[key] = _Anchor(key, label, turn, turn, last_seq=self._seq)
                    new.append(key)
                else:
                    if anchor.linked:  # an anticipated subject is now actually discussed
                        anchor.linked = False
                        anchor.known = None if anchor.known is False else anchor.known
                    anchor.last_turn = turn
                    anchor.last_seq = self._seq
                    anchor.mentions += 1
            self._expire(turn)
            queries = [self._anchors[key].label for key in new]
            task = self._task if role == "user" else None
        if role == "assistant":
            self._check_claims(text, [key for key, _ in found])
        if queries or task:
            self._schedule(queries, task)
        return new

    def observe_tool_call(self, name: str, arguments: Any = None) -> list[str]:
        """Feed a tool call: its arguments often name exactly what is needed next.

        Only the arguments are scanned; the tool's own name is not a subject.
        """
        del name
        if arguments is None:
            return []
        if isinstance(arguments, dict):
            text = " ".join(str(value) for value in arguments.values())
        else:
            text = str(arguments)
        return self.observe(text, role="tool")

    # ── output ────────────────────────────────────────────────────────────────

    def briefing(self, *, wait: float = 0.0) -> Briefing:
        """Assemble the current briefing. ``wait`` > 0 lets running prefetches finish first."""
        started = time.perf_counter()
        if wait > 0:
            self.flush(timeout=wait)
        with self._lock:
            # Mentioned subjects first (most recent first), anticipated links after.
            active = sorted(
                (a for a in self._anchors.values() if a.known is not False),
                key=lambda a: (not a.linked, a.last_seq, a.mentions), reverse=True,
            )[: self.max_anchors]
            unknown = [a.label for a in self._anchors.values()
                       if a.known is False and not a.linked and self._turn - a.last_turn <= 1]
            results = {a.key: self._results.get(a.key) for a in active}
            task_result = self._task_results.get(self._task)
            corrections = list(self._corrections.values())
            pending = len(self._pending)

        lines: list[str] = []
        used = 0
        shown: list[str] = []

        def add(line: str) -> bool:
            nonlocal used
            if used + len(line) + 1 > self.budget_chars:
                return False
            lines.append(line)
            used += len(line) + 1
            return True

        if corrections:
            add("Corrections (memory has newer values):")
            for text in corrections:
                add(f"- {text}")
        seen_ids: set[str] = set()
        for anchor in active:
            result = results.get(anchor.key)
            if result is None:
                continue
            facts = [f for f in getattr(result, "facts", []) if f.is_current]
            records = [
                f for f in result.fragments
                if f.id not in seen_ids and _contains_anchor(_anchor_norm(f.content), anchor.key)
            ][: self.per_anchor]
            records.sort(key=_timeline_key, reverse=True)  # newest first, undated last
            if not facts and not records:
                continue
            if not add(f"{anchor.label}{' (linked)' if anchor.linked else ''}:"):
                break
            for fact in facts:
                add(f"- {fact.attribute}: {fact.value} (since {fact.valid_from.date().isoformat()})")
            for fragment in records:
                if add(f"- {_one_line(fragment.content)}"):
                    seen_ids.add(fragment.id)
                    shown.append(fragment.id)
        if task_result is not None:
            extra = [f for f in task_result.fragments if f.id not in seen_ids][: self.per_anchor]
            if extra:
                add("Related to the current request:")
                for fragment in extra:
                    if add(f"- {_one_line(fragment.content)}"):
                        seen_ids.add(fragment.id)
                        shown.append(fragment.id)
        if unknown:
            add("Not in memory: " + ", ".join(unknown) + " (say so rather than guessing).")
        body = "\n".join(lines)
        text = f"<mark_briefing>\n{body}\n</mark_briefing>" if body else ""
        return Briefing(
            text=text,
            anchors=[a.label for a in active],
            corrections=corrections,
            unknown=unknown,
            memory_ids=shown,
            pending=pending,
            latency_ms=round((time.perf_counter() - started) * 1000, 2),
        )

    def flush(self, timeout: float | None = None) -> bool:
        """Wait for running prefetches. Returns True when none are left."""
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            with self._lock:
                pending = list(self._pending)
            if not pending:
                return True
            remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
            if remaining == 0.0:
                return False
            _wait(pending[0], remaining)

    def reset(self) -> None:
        """Forget the conversation (keep memory)."""
        self.flush(timeout=5)
        with self._lock:
            self._anchors.clear()
            self._results.clear()
            self._task_results.clear()
            self._corrections.clear()
            self._task = ""
            self._turn = 0

    def close(self) -> None:
        """Stop the background worker."""
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None

    def __enter__(self) -> "ContextWatch":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    # ── internals ─────────────────────────────────────────────────────────────

    def _extract_anchors(self, text: str) -> list[tuple[str, str]]:
        from mark.intelligence.retrieval import _anchor_norm, _contains_anchor, query_anchors

        found: dict[str, str] = {}
        for key in query_anchors(text):
            found.setdefault(key, _label_for(text, key))
        # Known fact subjects are anchors even when written plainly ("pulse metrics").
        normalised = _anchor_norm(text)
        for subject in self._known_subjects():
            if subject not in found and _contains_anchor(normalised, subject):
                found[subject] = _label_for(text, subject)
        return list(found.items())

    def _known_subjects(self) -> set[str]:
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
        for key in [k for k, a in self._anchors.items() if turn - a.last_turn > self.ttl_turns]:
            self._anchors.pop(key, None)
            self._results.pop(key, None)

    def _schedule(self, queries: list[str], task: str | None) -> None:
        if self._executor is None:
            self._prefetch(queries, task)
            return
        future = self._executor.submit(self._prefetch, queries, task)
        with self._lock:
            self._pending.add(future)
        future.add_done_callback(self._done)

    def _done(self, future: Future) -> None:
        with self._lock:
            self._pending.discard(future)
        error = future.exception()
        if error is not None:
            logger.warning("context watch prefetch failed: %s", error)

    def _prefetch(self, queries: list[str], task: str | None) -> None:
        batch = list(queries) + ([task] if task else [])
        if not batch:
            return
        results = self._memory.retrieve_many_sync(batch)
        from mark.intelligence.retrieval import _anchor_norm

        links: list[str] = []
        with self._lock:
            for query, result in zip(queries, results):
                key = _anchor_norm(query)
                self._results[key] = result
                anchor = self._anchors.get(key)
                if anchor is not None:
                    anchor.known = key not in result.missing_anchors
                    if anchor.known and not anchor.linked and self.follow_links:
                        links.extend(self._links_from(anchor, result))
            if task:
                self._task_results = {task: results[-1]}
        if links:
            self._prefetch(links, None)

    def _links_from(self, anchor: _Anchor, result: Any) -> list[str]:
        """Subjects named by the anchor's best records: likely next questions (caller holds lock)."""
        from mark.intelligence.retrieval import _anchor_norm, _contains_anchor

        labels: list[str] = []
        for fragment in result.fragments[:3]:
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
        """Flag values the agent states that memory knows were superseded."""
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
                if fact.value.lower() in lowered:
                    self._corrections.pop(f"{key}|{fact.attribute}", None)
                    continue
                history = facts_view.history(fact.subject, fact.attribute)
                stale = [old for old in history[:-1] if _mentions_value(old.value, lowered, tokens)]
                if stale:
                    self._corrections[f"{key}|{fact.attribute}"] = (
                        f"{fact.subject} {fact.attribute} is {fact.value} now "
                        f"(since {fact.valid_from.date().isoformat()}); "
                        f"{stale[-1].value} is outdated."
                    )


def _anchor_norm(text: str) -> str:
    from mark.intelligence.retrieval import _anchor_norm as norm

    return norm(text)


def _contains_anchor(content_norm: str, anchor: str) -> bool:
    from mark.intelligence.retrieval import _contains_anchor as contains

    return contains(content_norm, anchor)


def _timeline_key(fragment: Any) -> tuple[int, str]:
    from mark.intelligence.retrieval import declared_event_time

    when = declared_event_time(fragment)
    return (1, when.isoformat()) if when is not None else (0, "")


def _mentions_value(value: str, lowered: str, tokens: set[str]) -> bool:
    value = value.lower().strip()
    if not value:
        return False
    if re.fullmatch(r"\w+", value):
        return value in tokens
    return value in lowered


def _label_for(text: str, key: str) -> str:
    """The anchor as written in the text (e.g. 'atlas-core'), falling back to the key."""
    pattern = r"(?<!\w)" + r"[\W_]+".join(re.escape(part) for part in key.split()) + r"(?!\w)"
    match = re.search(pattern, text, re.IGNORECASE)
    return match.group(0) if match else key


def _one_line(text: str, limit: int = 240) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _wait(future: Future, timeout: float | None) -> None:
    try:
        future.exception(timeout=timeout)
    except Exception:
        pass
