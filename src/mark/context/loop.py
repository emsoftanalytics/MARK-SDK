"""A minimal chat loop where MARK is the context window (no agent framework).

:class:`MarkChat` talks to any OpenAI-compatible ``/chat/completions``
endpoint (Ollama, LM Studio, llama.cpp server, vLLM, OpenAI) with the
standard library only. Each call sends:

1. the system prompt, with a token-budgeted MARK briefing appended,
2. the last ``keep_last`` messages of the conversation (tool-safe window,
   see :func:`mark.context.history.split_history`),

and nothing else. Every completed exchange is stored in MARK *before* it can
fall out of the window, so an early fact ("my hive is called Colmeia Azul")
reaches the model through the briefing twenty turns later.

    chat = MarkChat(mark.runtime.memory("chat"), model="llama3.2:3b",
                    base_url="http://localhost:11434/v1", keep_last=4, budget_tokens=300)
    print(chat.send("What did I call my hive?"))

:meth:`MarkChat.prepare` and :meth:`MarkChat.commit` split :meth:`send` in
two, for callers that run the model themselves (streaming, tools, retries).
"""
from __future__ import annotations

import json
from dataclasses import replace
import urllib.error
import urllib.request
from typing import Any

from mark.context.budget import ContextBudget, as_budget
from mark.context.history import split_history
from mark.context.watch import Briefing, ContextWatch
from mark.context.watch_messages import inject
from mark.context.watch_render import render_briefing


class MarkChatError(RuntimeError):
    """The model server could not be reached or answered with an error."""


def chat_completion(base_url: str, model: str, messages: list[dict[str, Any]], *,
                    api_key: str | None = None, timeout: float = 120.0,
                    options: dict[str, Any] | None = None) -> str:
    """One OpenAI-compatible chat completion; returns the reply text."""
    body = json.dumps({"model": model, "messages": messages, **(options or {})}).encode()
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(base_url.rstrip("/") + "/chat/completions",
                                     data=body, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read())
    except urllib.error.HTTPError as error:   # surface the server's reason (e.g. unknown model)
        detail = error.read().decode(errors="replace")[:300]
        raise MarkChatError(f"model server error {error.code}: {detail}") from None
    except (urllib.error.URLError, OSError, ValueError) as error:
        raise MarkChatError(f"model server unreachable at {base_url}: {error}") from None
    try:
        return payload["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError, TypeError):
        raise MarkChatError(f"unexpected model response: {str(payload)[:300]}") from None


class MarkChat:
    """Conversation whose long-term context lives in MARK, not in the prompt.

    memory          a ``MarkMemory`` (``mark.runtime.memory(agent_id)``)
    model           model name on the server (e.g. ``"llama3.2:3b"``)
    base_url        OpenAI-compatible base URL (default local Ollama)
    system          system prompt
    keep_last       messages kept verbatim (>= 1; default 6 = three exchanges)
    budget_tokens   cap on the injected briefing (int or ContextBudget; default
                    400, filled: as many relevant memories as fit)
    token_counter   ``str -> int`` for the budget (default ceil(chars / 4))
    """

    def __init__(self, memory: Any, *, model: str = "llama3.2:3b",
                 base_url: str = "http://localhost:11434/v1", system: str = "",
                 keep_last: int = 6, budget_tokens: int | ContextBudget = 400,
                 token_counter: Any = None, api_key: str | None = None,
                 timeout: float = 120.0, options: dict[str, Any] | None = None) -> None:
        if keep_last < 1:
            raise ValueError("keep_last must be >= 1")
        self.memory = memory
        self.model = model
        self.base_url = base_url
        self.system = system
        self.keep_last = keep_last
        self.api_key = api_key
        self.timeout = timeout
        self.options = dict(options or {})
        self.budget = as_budget(budget_tokens, token_counter, fill=True)
        # Inline prefetch (background=False): this loop is synchronous anyway,
        # and the briefing must reflect the turn just stored.
        self.watch = ContextWatch(memory, background=False, budget_tokens=self.budget)
        self.window: list[dict[str, Any]] = []      # the last keep_last messages, nothing older
        self.last_prompt: list[dict[str, Any]] = []
        self.last_briefing: Briefing | None = None
        self.turns = 0

    def prepare(self, text: str) -> list[dict[str, Any]]:
        """The exact message list the model gets for user message *text* (no I/O to the model)."""
        self.watch.observe(text, role="user")
        pending = [*self.window, {"role": "user", "content": text}]
        kept = split_history(pending, self.keep_last).kept
        self.last_briefing = self._briefing(kept)
        base = [{"role": "system", "content": self.system}] if self.system else []
        self.last_prompt = inject(base + kept, self.last_briefing)
        return self.last_prompt

    def _briefing(self, kept: list[dict[str, Any]]) -> Briefing:
        """The budgeted briefing, minus memories the window already shows verbatim."""
        brief = self.watch.briefing()
        # A memory repeats a window message when it contains the message's start
        # (stored lines are capped, so the whole message may not be in it).
        recent = [" ".join(str(m.get("content") or "").split())[:80] for m in kept]
        recent = [text for text in recent if len(text) >= 12]
        items = [i for i in brief.items if not any(text in i.text for text in recent)]
        if len(items) == len(brief.items):
            return brief
        fit = self.budget.fit(items, render_briefing)   # re-fit: freed tokens go to other memories
        return replace(brief, text=fit.text, usage=fit.usage, items=tuple(items),
                       memory_ids=[i.ref for i in fit.items if i.ref])

    def commit(self, text: str, reply: str | None) -> None:
        """Record a finished exchange: store it in MARK, then slide the window.

        *reply* may be None when no model was called; only the user turn is kept then.
        """
        self.turns += 1
        exchange = [{"role": "user", "content": text}]
        stored = f"User said: {' '.join(text.split())}"
        if reply is not None:
            exchange.append({"role": "assistant", "content": reply})
            stored += f"\nAssistant replied: {' '.join(reply.split())}"
            self.watch.observe(reply, role="assistant")
        # Stored first, trimmed second: nothing leaves the window unless MARK has it.
        self.memory.store_sync(stored, source="mark-chat", tags=["conversation"],
                               metadata={"turn": self.turns})
        self.watch.refresh()   # subjects first seen in this turn are in memory now
        self.window = split_history([*self.window, *exchange], self.keep_last).kept

    def send(self, text: str) -> str:
        """Ask the model; returns its reply. Raises :class:`MarkChatError` on server errors."""
        messages = self.prepare(text)
        reply = chat_completion(self.base_url, self.model, messages, api_key=self.api_key,
                                timeout=self.timeout, options=self.options)
        self.commit(text, reply)
        return reply

    def close(self) -> None:
        self.watch.close()

    def __enter__(self) -> "MarkChat":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()
