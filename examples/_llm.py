"""Tiny LLM helper shared by the examples (standard library only).

Set MARK_LLM_URL to any OpenAI-compatible server to get real answers, e.g.
  Ollama:     MARK_LLM_URL=http://localhost:11434/v1  MARK_LLM_MODEL=llama3.2:3b
  LM Studio:  MARK_LLM_URL=http://localhost:1234/v1
Without it, ask() returns None and the examples print the prompts instead.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request


def ask(prompt: str, system: str = "") -> str | None:
    """Send one chat request; return the reply text, or None when no LLM is configured."""
    url = os.environ.get("MARK_LLM_URL")
    if not url:
        return None
    messages = ([{"role": "system", "content": system}] if system else []) + [
        {"role": "user", "content": prompt}
    ]
    body = json.dumps({"model": os.environ.get("MARK_LLM_MODEL", "llama3.2:3b"), "messages": messages})
    request = urllib.request.Request(
        url.rstrip("/") + "/chat/completions",
        data=body.encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            return json.loads(response.read())["choices"][0]["message"]["content"]
    except urllib.error.HTTPError as error:  # show the server's reason, e.g. a retired model
        detail = error.read().decode(errors="replace")[:300]
        raise SystemExit(f"LLM server error {error.code}: {detail}") from None
