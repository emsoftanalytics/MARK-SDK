"""Lifecycle and data-management commands of the CLI engine.

``forget``, ``resolve``, ``supersede``, ``list``, ``export``, ``import``,
``brief``, ``agents`` and ``config``, mixed into :class:`mark._cli.engine._Engine` (so
they are reachable in-process and through ``mark serve`` alike, and part of
the protocol fingerprint). Commands that take a memory id accept the short
8-character id that ``recall``/``list`` print, as long as it is unambiguous.

Standard library only at import time; MARK is loaded by the engine.
"""
from __future__ import annotations

from typing import Any

#: How many characters of a memory id the CLI prints.
SHORT_ID = 8


def memory_json(fragment: Any) -> dict[str, Any]:
    """The JSON shape of one memory in ``recall``/``list`` output."""
    from mark.intelligence.temporal_order import declared_event_time, event_time
    from mark.kinds import TAG_RESOLVED, TAG_SUPERSEDED, kind_of

    declared = declared_event_time(fragment)
    lifecycle = ("resolved" if TAG_RESOLVED in fragment.tags
                 else "superseded" if TAG_SUPERSEDED in fragment.tags else None)
    return {
        "id": fragment.id,
        "short_id": fragment.id[:SHORT_ID],
        "content": fragment.content,
        # Event date when the memory declares one, else the day it was stored.
        "date": event_time(fragment).date().isoformat(),
        "occurred_at": declared.isoformat() if declared else None,
        "stored_at": fragment.created_at.isoformat(),
        "kind": kind_of(fragment.tags),
        "tags": [t for t in fragment.tags if not t.startswith(("kind:", "mark:"))],
        "lifecycle": lifecycle,
    }


class EngineOpsMixin:
    """forget / resolve / supersede / list / export / import / brief / agents."""

    runtime: Any

    def _memory(self, agent: str) -> Any:  # provided by _Engine
        raise NotImplementedError

    def _full_id(self, agent: str, memory_id: str) -> str:
        """Expand a short id; ``ValueError`` when it matches nothing or several memories."""
        memory_id = memory_id.strip()
        matches = self.runtime.store.ids_with_prefix(agent, memory_id) if memory_id else []
        if len(matches) == 1:
            return matches[0]
        if not matches:
            raise ValueError(f"no memory with id {memory_id!r} in agent {agent!r}")
        raise ValueError(f"id {memory_id!r} is ambiguous; give more characters")

    def cmd_forget(self, agent: str, id: str | None = None, match: str | None = None,  # noqa: A002
                   secure: bool = False) -> dict[str, Any]:
        memory = self._memory(agent)
        if id is not None:
            forgotten = memory.forget(self._full_id(agent, id), secure=secure)
        else:
            forgotten = memory.forget(match=match, secure=secure)
        if secure and forgotten:
            memory.vacuum()  # the CLI promise: the text is gone from disk when this returns
        return {"forgotten": forgotten, "secure": secure}

    def cmd_resolve(self, agent: str, id: str, note: str | None = None) -> dict[str, Any]:  # noqa: A002
        return {"id": self._memory(agent).resolve(self._full_id(agent, id), note=note)}

    def cmd_supersede(self, agent: str, id: str, text: str) -> dict[str, Any]:  # noqa: A002
        old = self._full_id(agent, id)
        return {"id": self._memory(agent).supersede(old, text), "superseded": old}

    def cmd_list(self, agent: str, kind: str | None = None, open: bool = False,  # noqa: A002
                 limit: int = 20, offset: int = 0, tags: list[str] | None = None) -> dict[str, Any]:
        memory = self._memory(agent)
        filters = {"kind": kind, "open": True if open else None, "tags": tags}
        page = memory.list(limit=max(1, int(limit)), offset=max(0, int(offset)), **filters)
        return {"memories": [memory_json(f) for f in page], "total": memory.count(**filters),
                "offset": max(0, int(offset))}

    def cmd_config_show(self) -> dict[str, Any]:
        from mark.embeddings.info import embedder_info
        from mark.runtime_config import DEFAULT_USER_SUBJECT, USER_SUBJECT_KEY

        store = self.runtime.store
        return {"db": str(getattr(self, "db", "")),
                "user_subject": store.get_meta(USER_SUBJECT_KEY) or DEFAULT_USER_SUBJECT,
                "schema_version": store.schema_version(),
                "embedder": embedder_info(self.runtime.embedder).to_dict()}

    def cmd_config_user_subject(self, name: str) -> dict[str, Any]:
        from mark.runtime_config import apply_user_subject

        # Stored with the store and applied to this (possibly long-lived server) runtime.
        return {"user_subject": apply_user_subject(self.runtime, name)}

    def cmd_agents(self) -> dict[str, Any]:
        store = self.runtime.store
        return {"agents": [{"agent": a, "memories": store.count(a)} for a in store.agent_ids()]}

    def cmd_export(self, path: str, agent: str | None = None, decrypt: bool = False) -> dict[str, Any]:
        from mark.memory.facade.portability import export_jsonl

        store = self.runtime.store
        agents = [agent] if agent else store.agent_ids()
        return {"path": path, **export_jsonl(store, path, agents, decrypt=decrypt)}

    def cmd_import(self, path: str, agent: str | None = None) -> dict[str, Any]:
        from mark.memory.facade.portability import import_jsonl

        return {"path": path, **import_jsonl(path, self._memory, agent_id=agent)}

    def cmd_brief(self, agent: str, budget: int = 400) -> dict[str, Any]:
        memory = self._memory(agent)
        try:
            from mark.context.session_brief import brief
        except ImportError:
            return {"text": fallback_brief(memory, int(budget)), "source": "fallback"}
        result = brief(memory, budget_tokens=int(budget))
        payload = {**result.to_dict(), "source": "brief"}
        if not str(payload.get("text") or "").strip():
            payload["text"] = "(memory is empty)"  # say so rather than print nothing
        return payload


def fallback_brief(memory: Any, budget_tokens: int) -> str:
    """A plain session brief used until :mod:`mark.context.brief` is available.

    Current facts, then open TODOs, recent decisions and recent bugs, newest
    first, cut at roughly ``budget_tokens`` (4 characters per token).
    """
    budget = max(40, budget_tokens) * 4
    sections: list[tuple[str, list[str]]] = [
        ("Facts", [fact.as_text() for fact in memory.facts.current()]),
        ("Open TODOs", _texts(memory.list(kind="todo", open=True, limit=10))),
        ("Recent decisions", _texts(memory.list(kind="decision", open=True, limit=5))),
        ("Recent bugs", _texts(memory.list(kind="bug", open=True, limit=5))),
    ]
    lines: list[str] = []
    used = 0
    for title, items in sections:
        if not items:
            continue
        for line in (f"{title}:", *(f"- {item}" for item in items)):
            if used + len(line) + 1 > budget:
                return "\n".join(lines) if lines else "(nothing to brief yet)"
            lines.append(line)
            used += len(line) + 1
    return "\n".join(lines) if lines else "(nothing to brief yet)"


def _texts(fragments: list[Any]) -> list[str]:
    newest = sorted(fragments, key=lambda f: f.created_at, reverse=True)
    return [f"{f.content} ({f.id[:SHORT_ID]})" for f in newest]
