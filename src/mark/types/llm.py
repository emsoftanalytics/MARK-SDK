# SPDX-License-Identifier: MIT
# Copyright (c) 2025 MARK Contributors
"""Protocol for developer-supplied LLM providers."""
from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class LLMProvider(Protocol):
    """Minimal protocol for an LLM callable used by local MARK features.

    Implementations only need to satisfy complete(). Streaming, tool-use,
    and structured-output are out of scope at this boundary — they are
    implementation details of the caller or of cloud plugins.

    Example adapter for any OpenAI-compatible client::

        class MyLLM:
            def complete(self, prompt: str) -> str:
                response = openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                )
                return response.choices[0].message.content or ""
    """

    def complete(self, prompt: str) -> str:
        """Send prompt and return the model's text response."""
        ...
