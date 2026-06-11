"""Minimal shared state helpers for LangGraph-style integrations."""

from __future__ import annotations

from typing import Any, TypedDict


class MarkState(TypedDict, total=False):
    """LangGraph state slice carrying MARK context between graph nodes."""
    agent_id: str
    session_id: str
    session_prefix: str
    tags: list[str]
    mark_context: str
    mark_metadata: dict[str, Any]
