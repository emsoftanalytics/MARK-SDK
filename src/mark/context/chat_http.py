"""Standard-library client for OpenAI-compatible ``/chat/completions`` (used by MarkChat).

No HTTP dependency on purpose: MarkChat must run anywhere Python runs, next
to a local Ollama / llama.cpp / LM Studio server.

``on_token`` switches the request to streaming (``"stream": true``, server
sent events) and is called with each piece of text as it arrives. A 3B
model on a laptop CPU takes 30-100 s per answer; without streaming a chat
UI shows nothing for that long (a10 small-model run).
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any, Callable


class MarkChatError(RuntimeError):
    """The model server could not be reached or answered with an error."""


def _request(base_url: str, body: dict[str, Any], api_key: str | None) -> urllib.request.Request:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return urllib.request.Request(base_url.rstrip("/") + "/chat/completions",
                                  data=json.dumps(body).encode(), headers=headers)


def _read_stream(response: Any, on_token: Callable[[str], None]) -> str:
    """Concatenate the ``delta.content`` pieces of an SSE stream, reporting each one."""
    parts: list[str] = []
    for raw in response:
        line = raw.decode("utf-8", "replace").strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            break
        try:
            delta = json.loads(data)["choices"][0].get("delta") or {}
        except (ValueError, KeyError, IndexError, TypeError):
            raise MarkChatError(f"unexpected stream chunk: {data[:300]}") from None
        piece = delta.get("content") or ""
        if piece:
            parts.append(piece)
            on_token(piece)
    return "".join(parts)


def chat_completion(base_url: str, model: str, messages: list[dict[str, Any]], *,
                    api_key: str | None = None, timeout: float = 120.0,
                    options: dict[str, Any] | None = None,
                    on_token: Callable[[str], None] | None = None) -> str:
    """One OpenAI-compatible chat completion; returns the reply text.

    With *on_token* the reply is streamed and each piece passed to it as it
    arrives; the return value is the whole reply either way.
    """
    body = {"model": model, "messages": messages, **(options or {})}
    if on_token is not None:
        body["stream"] = True
    try:
        with urllib.request.urlopen(_request(base_url, body, api_key), timeout=timeout) as response:
            if on_token is not None:
                return _read_stream(response, on_token)
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
