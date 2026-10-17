# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 MARK Contributors
"""Describers: turn an image file into observed attributes.

A describer is any ``callable(path) -> dict | str``. This module ships one
for vision models behind an OpenAI-compatible ``/chat/completions`` endpoint
(LM Studio, vLLM, llama.cpp server, Ollama's OpenAI API, hosted APIs). It
uses only the standard library; the image is sent inline as a base64 data
URL and is never persisted by MARK.

The model is asked for a flat JSON object and told to *omit* anything it
cannot see, which is what lets continuity checks report ``unverified``
instead of a guessed match. Non-JSON replies are returned as plain text and
compared conservatively.
"""
from __future__ import annotations

import base64
import json
import mimetypes
import re
import urllib.request
from collections.abc import Sequence
from pathlib import Path
from typing import Any

_PROMPT = (
    "Describe the visible attributes of {who} in this image as one flat JSON object "
    "mapping attribute name to a short value (for example colours, clothing, props, "
    "hair). {focus}Only include what is clearly visible; omit anything you cannot "
    "see. Reply with JSON only."
)
_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


class OpenAIVisionDescriber:
    """Describe images with a vision LLM over an OpenAI-compatible HTTP API.

    Example::

        describer = OpenAIVisionDescriber("http://localhost:1234/v1", "qwen2.5-vl-7b")
        report = memory.images.check_consistency(frame, "Elena", describe=describer)
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        api_key: str | None = None,
        subject: str | None = None,
        attributes: Sequence[str] = (),
        timeout: float = 120.0,
        max_tokens: int = 400,
    ) -> None:
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model = model
        self.api_key = api_key
        self.subject = subject
        self.attributes = tuple(attributes)
        self.timeout = timeout
        self.max_tokens = max_tokens

    def prompt(self) -> str:
        """The instruction sent with each image."""
        who = self.subject or "the main subject"
        focus = f"Cover these attributes if visible: {', '.join(self.attributes)}. " if self.attributes else ""
        return _PROMPT.format(who=who, focus=focus)

    def request_body(self, path: str | Path) -> dict[str, Any]:
        """The chat-completions payload (exposed for inspection and tests)."""
        file = Path(path)
        mime = mimetypes.guess_type(file.name)[0] or "image/png"
        data = base64.b64encode(file.read_bytes()).decode("ascii")
        return {
            "model": self.model,
            "temperature": 0,
            "max_tokens": self.max_tokens,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": self.prompt()},
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{data}"}},
                ],
            }],
        }

    def __call__(self, path: str) -> dict[str, Any] | str:
        """Send the image and return the parsed attributes (or raw text)."""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(
            self.url, data=json.dumps(self.request_body(path)).encode("utf-8"), headers=headers, method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            reply = json.loads(response.read().decode("utf-8"))
        return parse_reply(reply["choices"][0]["message"]["content"] or "")


def parse_reply(text: str) -> dict[str, Any] | str:
    """Parse a model reply: a JSON object if there is one, else the text."""
    cleaned = _FENCE.sub("", text.strip())
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start != -1 and end > start:
        try:
            value = json.loads(cleaned[start:end + 1])
        except json.JSONDecodeError:
            return text.strip()
        if isinstance(value, dict):
            return value
    return text.strip()
