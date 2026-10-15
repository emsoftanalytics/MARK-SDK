# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 MARK Contributors
"""Configuration for :class:`mark.runtime.Mark`.

* :class:`MarkConfig` is the immutable record of where a local Mark lives.
* :class:`MarkConfigureMixin` holds the late-binding ``configure_*`` hooks
  that swap LLM, compressor, or query expander on a running instance. Each
  hook is a no-op when the underlying engine does not support it.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class MarkConfig:
    """Configuration for a local Mark runtime."""
    project_path: Path
    store_path:   Path


class MarkConfigureMixin:
    """Late-binding configuration hooks for a running :class:`Mark`."""

    memory: Any
    _runtime_engine: Any

    def configure_query_expander(self, expander: object) -> None:
        """Replace the active query expander at runtime (late-binding)."""
        if hasattr(self._runtime_engine, "configure_query_expander"):
            self._runtime_engine.configure_query_expander(expander)

    def session_log(self, agent_id: str, session_id: str | None = None) -> object:
        """Return a SessionActivityLog scoped to agent_id (and optionally session_id)."""
        if hasattr(self._runtime_engine, "session_log"):
            return self._runtime_engine.session_log(agent_id, session_id)
        raise AttributeError("session_log not available on this runtime")

    def configure_compressor(self, compressor: object) -> None:
        """Replace the active contextual compressor at runtime (late-binding).

        Example::

            mark = Mark.local(project_path=".")
            mark.configure_compressor(LLMContextualCompressor(llm=my_llm))
            result = mark.memory.retrieve("What must stay consistent for Elena?",
                                          compress=True)
        """
        if hasattr(self._runtime_engine, "configure_compressor"):
            self._runtime_engine.configure_compressor(compressor)

    def configure_llm(self, llm: object) -> None:
        """Bind an LLM provider to enable local entity extraction in observe().

        Can be called at any point after Mark.local() — the LLM will be used
        on the next observe() call. Existing fragments are not re-processed.

        Example::

            mark = Mark.local(project_path=".")
            mark.configure_llm(MyOpenAIAdapter())
            result = mark.memory.observe("Elena enters the North Warehouse.")
        """
        if hasattr(self._runtime_engine, "configure_llm"):
            self._runtime_engine.configure_llm(llm)
        # The SimpleMemory's MarkMemory was created before this call and holds
        # its own reference, so propagate to it explicitly.
        if hasattr(self.memory, "_mark_memory") and hasattr(self.memory._mark_memory, "_llm"):
            self.memory._mark_memory._llm = llm  # type: ignore[assignment]
