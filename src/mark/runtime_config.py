# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 MARK Contributors
"""Configuration for :class:`mark.runtime.Mark`.

* :class:`MarkConfig` is the immutable record of where a local Mark lives.
* :class:`MarkConfigureMixin` holds the late-binding ``configure_*`` hooks
  that swap LLM, compressor, or query expander on a running instance. Each
  hook is a no-op when the underlying engine does not support it.
* :func:`apply_user_subject` stores the user subject (who "I" is) with the
  store and applies it on open, for ``Mark.local``, the CLI and MCP alike.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


#: schema_info key holding the user subject, so it travels with the store.
USER_SUBJECT_KEY = "user_subject"
#: Subject of first-person facts when none was ever configured.
DEFAULT_USER_SUBJECT = "user"


def apply_user_subject(runtime: Any, subject: str | None = None) -> str:
    """Apply the user subject to *runtime*; returns the subject in effect.

    With *subject*, it is validated, stored in the store's ``schema_info``
    and applied. Without, the stored value (if any) is applied. Why stored:
    it used to be in-process only, so "Where do I live?" lost its subject
    after a reopen and the CLI/MCP never knew it (a10 report M-a).
    """
    store = getattr(runtime, "store", None)
    if subject is not None:
        name = str(subject).strip()
        if not name:
            from mark.errors import InvalidInputError

            raise InvalidInputError("user subject must be a non-empty name")
        runtime.configure_user_subject(name)
        store.set_meta(USER_SUBJECT_KEY, name)
        return name
    get_meta = getattr(store, "get_meta", None)
    stored = get_meta(USER_SUBJECT_KEY) if callable(get_meta) else None
    if stored:
        runtime.configure_user_subject(stored)
        return stored
    return DEFAULT_USER_SUBJECT


@dataclass(frozen=True)
class MarkConfig:
    """Configuration for a local Mark runtime."""
    project_path: Path
    store_path:   Path


class MarkConfigureMixin:
    """Late-binding configuration hooks for a running :class:`Mark`."""

    memory: Any
    _runtime_engine: Any

    @property
    def user_subject(self) -> str:
        """Who "I" is in this store's memories (default ``"user"``)."""
        stored = self._runtime_engine.store.get_meta(USER_SUBJECT_KEY)
        return stored or DEFAULT_USER_SUBJECT

    def configure_user_subject(self, subject: str) -> None:
        """Name the user ("I", "my") and store it, so every later open uses it too.

        Example: ``mark.configure_user_subject("ana")`` then "I moved to Porto"
        records ``ana city = Porto`` and "Where do I live?" finds it — also
        from ``mark recall`` and MCP. Same as ``mark config user-subject ana``.
        """
        apply_user_subject(self._runtime_engine, subject)

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
