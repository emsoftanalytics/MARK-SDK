"""Sliding conversation window with tool-call safety ("MARK as the context window").

When MARK replaces the conversation history, the model sees only the system
message(s), the last ``keep_last`` messages and a MARK briefing; everything
older is stored in memory and comes back through the briefing when relevant.

:func:`split_history` decides what is kept. It accepts OpenAI-style dicts
(``role``: system/user/assistant/tool) and LangChain message objects
(``type``: system/human/ai/tool) alike, and guarantees:

* **System messages are always kept** (they are not counted in ``keep_last``).
* **The current user turn is always kept.** If the latest user message is
  older than the window (a long tool loop), it is kept in front of the window,
  so the model still sees the question it is answering.
* **Tool calls are never split from their results.** If the window starts in
  the middle of a tool exchange, it is extended backwards to the assistant
  message that made the calls; tool results with no assistant message before
  them (orphans) are dropped. Providers reject both broken shapes.
* Order is preserved; nothing is mutated.

:func:`turn_text` renders one message as a line for storage in memory
(tool-call-only assistant messages render as nothing);
:func:`message_key` identifies a message across store paths.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Sequence

_ROLE_ALIASES = {
    "human": "user", "user": "user",
    "ai": "assistant", "assistant": "assistant",
    "tool": "tool", "function": "tool",
    "system": "system", "developer": "system",
}


def _get(message: Any, key: str, default: Any = None) -> Any:
    if isinstance(message, dict):
        return message.get(key, default)
    return getattr(message, key, default)


def message_role(message: Any) -> str:
    """Normalised role: user / assistant / tool / system (or the raw value)."""
    raw = _get(message, "role") if isinstance(message, dict) else None
    raw = raw or _get(message, "type") or _get(message, "role") or ""
    return _ROLE_ALIASES.get(str(raw), str(raw))


def message_text(message: Any) -> str:
    """Plain text of a message's content (string or list of text parts)."""
    content = _get(message, "content")
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, (list, tuple)):
        parts = []
        for part in content:
            text = part if isinstance(part, str) else _get(part, "text")
            if isinstance(text, str) and text:
                parts.append(text)
        return "\n".join(parts)
    return str(content)


def has_tool_calls(message: Any) -> bool:
    return bool(_get(message, "tool_calls") or _get(message, "function_call"))


def _status(message: Any) -> str:
    return str(_get(message, "status") or "")


def turn_text(message: Any, limit: int = 2000) -> str:
    """One stored line per message, e.g. ``"User said: ..."`` or ``"Tool [lookup] returned: ..."``.

    An assistant message that only dispatches tool calls has no words of its
    own and renders as ``""`` (nothing is stored; the calls' results are). A
    failed tool call (``status="error"``) renders as ``"Tool [x] failed: ..."``.
    """
    role = message_role(message)
    text = " ".join(message_text(message).split())[:limit]
    if not text:
        return ""
    if role == "user":
        return f"User said: {text}"
    if role == "assistant":
        return f"Assistant replied: {text}"
    if role == "tool":
        name = _get(message, "name") or "tool"
        verb = "failed" if _status(message) == "error" else "returned"
        return f"Tool [{name}] {verb}: {text}"
    return text


def message_key(message: Any) -> str:
    """Stable identity for de-duplicating storage.

    Tool results are keyed by their ``tool_call_id`` (plus content), not by
    ``id``: LangGraph assigns the id only when the message enters the state,
    so the result a tool wrapper sees and the one later read from history
    must map to the same key. Other messages use id plus content.
    """
    call_id = _get(message, "tool_call_id")
    ident = None if (call_id and message_role(message) == "tool") else _get(message, "id")
    payload = json.dumps([message_role(message), message_text(message), ident, call_id],
                         default=str, sort_keys=True)
    return hashlib.blake2b(payload.encode("utf-8", "replace"), digest_size=12).hexdigest()


@dataclass(frozen=True)
class HistorySplit:
    """``kept`` is what the model sees (system messages included, in order);
    ``dropped`` is what MARK must hold instead (non-system, in order)."""

    kept: list[Any]
    dropped: list[Any]


def split_history(messages: Sequence[Any], keep_last: int) -> HistorySplit:
    """Keep system messages + the last *keep_last* others (see module invariants)."""
    if keep_last < 1:
        raise ValueError("keep_last must be >= 1 (the current turn is always kept)")
    items = list(messages or [])
    body = [i for i, m in enumerate(items) if message_role(m) != "system"]
    if len(body) <= keep_last:
        return HistorySplit(items, [])
    start = len(body) - keep_last
    # Never start inside a tool exchange: walk back to the assistant call.
    while start > 0 and message_role(items[body[start]]) == "tool":
        start -= 1
    window = body[start:]
    # Orphaned results at the very front (no calling message left): drop them.
    while window and message_role(items[window[0]]) == "tool":
        window = window[1:]
    users = [i for i in body if message_role(items[i]) == "user"]
    if users and users[-1] not in window:
        window = [users[-1], *window]   # the question being answered stays visible
    keep = set(window) | {i for i, m in enumerate(items) if message_role(m) == "system"}
    return HistorySplit([m for i, m in enumerate(items) if i in keep],
                        [items[i] for i in body if i not in keep])
