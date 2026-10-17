"""ContextWatch for plain agent loops: OpenAI-style message lists in, briefing out.

Most agent loops keep the conversation as a list of chat messages and resend
the whole list on every model call::

    messages.append({"role": "user", "content": question})
    watch.observe_messages(messages)                    # feeds only what is new
    reply = client.chat(inject(messages, watch.briefing(wait=0.25)))

``observe_messages`` understands ``role``/``content`` dicts (content as a
string or a list of parts), assistant ``tool_calls`` (function name plus JSON
``arguments``), legacy ``function_call`` and ``tool``/``function`` results.
Objects with the same attributes (SDK message classes) work as well.

Invariants:

* **Only new messages are fed.** The watch remembers how long the list was
  and fingerprints of its last few messages. When the list was only appended
  to, the new tail is fed. When it was truncated or rewritten (sliding
  window, summarisation), the last seen message is located again and only
  what follows it is fed; with no overlap at all, only the latest user turn
  onwards is fed. History is never replayed wholesale, and at most
  ``_BACKFILL_LIMIT`` messages are fed per call.
* **System messages are never observed.** They usually carry the injected
  briefing; observing it would make the watch reinforce its own output. Any
  ``<mark_briefing>`` block is stripped from other messages for the same reason.
* **Never raises.** A malformed message is skipped and logged; the agent's
  turn must not fail because of anticipation.
* **inject() never mutates** the caller's list or message dicts.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
from collections.abc import Sequence
from typing import Any

logger = logging.getLogger("mark.context.watch")

#: At most this many unseen messages are fed per call (long histories on first use).
_BACKFILL_LIMIT = 20
#: Fingerprints kept to re-find our position after the list was rewritten.
_TAIL = 8
#: Tool results can be huge; only their head is scanned for subjects.
_TOOL_TEXT_LIMIT = 4000
_BRIEFING_RE = re.compile(r"\s*<mark_briefing>.*?</mark_briefing>\s*", re.DOTALL)
_SKIP_ROLES = {"system", "developer"}


def _get(message: Any, key: str, default: Any = None) -> Any:
    """Field access for dicts and message objects alike."""
    if isinstance(message, dict):
        return message.get(key, default)
    return getattr(message, key, default)


def _text(content: Any) -> str:
    """Plain text of a message content: a string or a list of text/image parts."""
    if content is None:
        return ""
    if isinstance(content, str):
        text = content
    elif isinstance(content, (list, tuple)):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            else:  # {"type": "text"|"input_text"|"output_text", "text": ...}; images are skipped
                value = _get(part, "text")
                if isinstance(value, str):
                    parts.append(value)
        text = "\n".join(parts)
    else:
        text = str(content)
    return _BRIEFING_RE.sub("\n", text).strip()


def _arguments(raw: Any) -> Any:
    """Tool-call arguments: parsed JSON when possible, the raw value otherwise."""
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except ValueError:
            return raw
    return raw


def _tool_calls(message: Any) -> list[tuple[str, Any]]:
    """(name, arguments) for every tool call an assistant message makes."""
    calls: list[tuple[str, Any]] = []
    for call in _get(message, "tool_calls") or []:
        function = _get(call, "function") or call  # OpenAI nests; some SDKs flatten
        calls.append((str(_get(function, "name", "") or ""),
                      _arguments(_get(function, "arguments", _get(call, "args")))))
    legacy = _get(message, "function_call")
    if legacy:
        calls.append((str(_get(legacy, "name", "") or ""), _arguments(_get(legacy, "arguments"))))
    return calls


def _fingerprint(message: Any) -> str:
    """Stable identity of a message's observable content (not of the object)."""
    try:
        payload = json.dumps(
            [_get(message, "role"), _get(message, "content"), _get(message, "tool_calls"),
             _get(message, "tool_call_id"), _get(message, "function_call")],
            sort_keys=True, default=str,
        )
    except Exception:  # pragma: no cover - default=str makes this very unlikely
        payload = repr(message)
    return hashlib.blake2b(payload.encode("utf-8", "replace"), digest_size=12).hexdigest()


def _unseen_start(prints: list[str], count: int, tail: list[str]) -> int:
    """Index of the first message not fed yet (see the module invariants)."""
    if not tail:
        return 0
    k = min(len(tail), count)
    # Fast path: the list only grew (or stayed the same).
    if count <= len(prints) and prints[count - k:count] == tail[-k:]:
        return count
    # Rewritten: find the latest position preceded by our last few messages.
    # A short window (3) still matches when older messages were dropped from
    # the middle (system prompt kept, history trimmed).
    for end in range(len(prints), 0, -1):
        k = min(3, len(tail), end)
        if prints[end - k:end] == tail[-k:]:
            return end
    # No overlap: a new or fully rewritten conversation; start at its latest user turn.
    return -1


def _latest_user_turn(messages: Sequence[Any]) -> int:
    for index in range(len(messages) - 1, -1, -1):
        if _get(messages[index], "role") == "user":
            return index
    return max(0, len(messages) - 1)


class _WatchMessagesMixin:
    """``observe_messages`` and the small-model preset for :class:`ContextWatch`.

    Expects ``observe``, ``observe_tool_call`` and ``_lock`` on the host class.
    Cursor state (``_msg_count``, ``_msg_tail``) is created lazily here so
    the host's constructor stays unchanged.
    """

    @classmethod
    def for_small_models(cls, memory: Any, **overrides: Any) -> Any:
        """A watch tuned for small local models with short context windows.

        Differences from the defaults: ``budget_chars`` 700 (vs 1800),
        ``max_anchors`` 3 (vs 6), ``per_anchor`` 1 record per subject (vs 3),
        ``line_chars`` 120 per bullet (vs 240) and ``follow_links`` 1 (vs 3).
        It also sets ``budget_tokens=175`` (about 700 characters with the
        default 4-chars-per-token estimate, so the same size as the character
        budget) to get priority fill: corrections and "Not in memory" notices
        are kept first, since they matter most for models that would
        otherwise guess. Any keyword overrides a value (``budget_tokens=None``
        restores the plain character budget).
        """
        settings = {"budget_chars": 700, "max_anchors": 3, "per_anchor": 1, "line_chars": 120,
                    "budget_tokens": 175}
        follow_links = overrides.pop("follow_links", 1)
        settings.update(overrides)
        watch = cls(memory, **settings)
        watch.follow_links = follow_links
        return watch

    def observe_messages(self, messages: Sequence[Any]) -> list[str]:
        """Feed the messages of *messages* not seen by a previous call. Returns new anchors.

        Safe to call with the full conversation before every model call. Never raises.
        """
        try:
            items = list(messages or [])
            prints = [_fingerprint(message) for message in items]
            with self._lock:  # concurrent callers must not feed the same tail twice
                start = _unseen_start(prints, getattr(self, "_msg_count", 0), getattr(self, "_msg_tail", []))
                if start < 0:
                    start = _latest_user_turn(items)
                self._msg_count = len(items)
                self._msg_tail = prints[-_TAIL:]
            fresh = items[start:][-_BACKFILL_LIMIT:]
        except Exception as exc:
            logger.warning("observe_messages could not read the message list: %s", exc)
            return []
        new: list[str] = []
        for message in fresh:
            try:
                new.extend(self._observe_message(message))
            except Exception as exc:  # one bad message never breaks the turn
                logger.warning("observe_messages skipped a message: %s", exc)
        return new

    def _observe_message(self, message: Any) -> list[str]:
        """Route one chat message to observe() / observe_tool_call()."""
        role = _get(message, "role")
        if role in _SKIP_ROLES or role is None:
            return []
        text = _text(_get(message, "content"))
        if role in ("tool", "function"):
            return self.observe(text[:_TOOL_TEXT_LIMIT], role="tool")
        if role == "assistant":
            found = self.observe(text, role="assistant") if text else []
            for name, arguments in _tool_calls(message):
                found.extend(self.observe_tool_call(name, arguments))
            return found
        return self.observe(text, role="user")  # "user" and unknown roles


def inject(messages: Sequence[Any], briefing: Any, *, budget_tokens: Any = None,
           token_counter: Any = None) -> list[Any]:
    """A new message list with *briefing* placed in the system message.

    *briefing* is a :class:`Briefing` or its text. ``budget_tokens`` (an int
    or a ContextBudget) caps the injected block: a budgeted Briefing is
    re-fitted by section priority, other text keeps whole lines from the top
    until the cap (see :mod:`mark.context.budget`). Any earlier
    ``<mark_briefing>`` block in the system message is replaced, so calling
    this before every model call never accumulates briefings. Without a
    system message, one is inserted at the front. An empty briefing still
    removes a stale block. The input list and its messages are not mutated.
    """
    text = (getattr(briefing, "text", briefing) or "").strip()
    if budget_tokens is not None:
        text = _fit_text(briefing, text, budget_tokens, token_counter)
    out = list(messages or [])
    for index, message in enumerate(out):
        if _get(message, "role") != "system":
            continue
        content = _get(message, "content")
        if isinstance(content, (list, tuple)):
            parts = [p for p in content if "<mark_briefing>" not in str(_get(p, "text", p) or "")]
            new_content: Any = parts + ([{"type": "text", "text": text}] if text else [])
        else:
            base = _BRIEFING_RE.sub("\n", str(content or "")).strip()
            new_content = "\n\n".join(part for part in (base, text) if part)
        updated = dict(message) if isinstance(message, dict) else {"role": "system"}
        updated["content"] = new_content
        out[index] = updated
        return out
    if text:
        out.insert(0, {"role": "system", "content": text})
    return out


def _fit_text(briefing: Any, text: str, budget_tokens: Any, token_counter: Any) -> str:
    """*text* under a token budget (used by :func:`inject`)."""
    from mark.context.budget import BudgetItem, as_budget, render_lines
    from mark.context.watch_render import CLOSE, OPEN, render_briefing

    budget = as_budget(budget_tokens, token_counter)
    items = getattr(briefing, "items", None)
    if items:
        return budget.fit(items, render_briefing).text
    # Plain text: whole lines, earlier lines first (all the same priority). A
    # <mark_briefing> wrapper is kept intact around whatever lines fit.
    wrapped = text.startswith(OPEN) and text.endswith(CLOSE)
    body = text[len(OPEN):-len(CLOSE)].strip() if wrapped else text
    lines = [BudgetItem("records", line) for line in body.splitlines() if line.strip()]
    return budget.fit(lines, render_briefing if wrapped else render_lines).text
