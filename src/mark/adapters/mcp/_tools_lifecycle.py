"""MCP tools that list and revise memories: ``mark_list``, ``mark_resolve``, ``mark_supersede``, ``mark_forget``.

The same lifecycle verbs as the Python API and the ``mark`` CLI (``list
--kind/--open``, ``resolve``, ``supersede``, ``forget [--match] [--secure]``),
so an MCP client can close a TODO or replace a stale decision instead of
piling up contradicting notes. Every id argument takes the 8-character
short id the other tools print (any unambiguous prefix of 4+ characters).

Ids are resolved inside the caller's namespace only (``ids_with_prefix`` is
agent-scoped): a client limited by ``allowed_agent_ids`` can never touch
another namespace's memory by guessing its id.
"""
from __future__ import annotations

from typing import Annotated, Any, Callable

from mark.adapters.mcp._common import MAX_CONTENT, AgentId, bounded, fail, guard, require_text, short
from mark.kinds import KINDS, TAG_RESOLVED, TAG_SUPERSEDED, kind_of

MemoryId = Annotated[str, bounded("Id of the memory: the 8-character short id shown by mark_retrieve, "
                                  "mark_list or mark_write (or the full id).", maxLength=100)]
#: Shortest id prefix accepted (shorter ones match too much by accident).
MIN_PREFIX = 4
_LIST_CHARS = 300


def own_id(memory: Any, target: str) -> str:
    """The full id of *target* (a full id or a unique prefix) in this namespace, else a tool error."""
    prefix = (target or "").strip()
    if len(prefix) < MIN_PREFIX:
        fail(f"memory_id must have at least {MIN_PREFIX} characters (the short id is 8)")
    matches = memory._store.ids_with_prefix(memory.agent_id, prefix, limit=2)
    if not matches:
        fail(f"no memory with id {prefix!r} in this namespace")
    if len(matches) > 1:
        fail(f"id {prefix!r} matches several memories; give more characters")
    return matches[0]


def _line(fragment: Any) -> str:
    from mark.context.watch_kinds import when_of

    when = when_of(fragment)
    kind = kind_of(fragment)
    state = (" (resolved)" if TAG_RESOLVED in fragment.tags
             else " (superseded)" if TAG_SUPERSEDED in fragment.tags else "")
    text = " ".join(guard(fragment.content).split())
    if len(text) > _LIST_CHARS:
        text = text[: _LIST_CHARS - 1] + "…"
    date = f"{when.date().isoformat()} " if when is not None else ""
    return f"- [{short(fragment.id)}] {date}{f'[{kind}] ' if kind else ''}{text}{state}"


def register_lifecycle_tools(server: Any, run: Callable[..., Any], *, enabled: set[str]) -> None:
    """Add the enabled lifecycle tools; *run(name, agent_id, work)* runs *work(memory)* off-loop."""

    if "mark_list" in enabled:
        @server.tool()
        async def mark_list(
            kind: Annotated[str, bounded("Only this kind: " + ", ".join(KINDS) + " (empty = all).",
                                         enum=["", *KINDS], maxLength=20)] = "",
            open: Annotated[bool, bounded("Hide resolved and superseded memories.")] = True,  # noqa: A002
            limit: Annotated[int, bounded("Memories to show (1-100).", minimum=1, maximum=100)] = 20,
            offset: Annotated[int, bounded("Skip this many first (paging).", minimum=0)] = 0,
            agent_id: AgentId = "",
        ) -> str:
            """List memories, most important first, optionally only one kind (e.g. open TODOs).

            Each line shows the short id to pass to mark_resolve / mark_supersede /
            mark_forget, the date, the kind and the text.
            """
            if not 1 <= limit <= 100 or offset < 0:
                fail("limit must be between 1 and 100 and offset at least 0")

            def work(memory: Any) -> str:
                filters = {"kind": kind.strip() or None, "open": True if open else None}
                page = memory.list(limit=limit, offset=offset, **filters)
                total = memory.count(**filters)
                if not page:
                    return "No memories match." if not offset else f"No memories after offset {offset}."
                more = total - offset - len(page)
                footer = [f"[{more} more: use offset={offset + len(page)}]"] if more > 0 else []
                return "\n".join([_line(f) for f in page] + footer)

            return await run("mark_list", agent_id, work)

    if "mark_resolve" in enabled:
        @server.tool()
        async def mark_resolve(
            memory_id: MemoryId,
            note: Annotated[str, bounded("Optional note on how it was resolved.", maxLength=2000)] = "",
            agent_id: AgentId = "",
        ) -> str:
            """Mark a TODO or bug as done; it leaves open lists and briefs but keeps its history."""

            def work(memory: Any) -> str:
                full = own_id(memory, memory_id)
                memory.resolve(full, note=note.strip() or None)
                return f"Resolved memory {short(full)}."

            return await run("mark_resolve", agent_id, work)

    if "mark_supersede" in enabled:
        @server.tool()
        async def mark_supersede(
            memory_id: MemoryId,
            content: Annotated[str, bounded("The new, correct text.", minLength=1, maxLength=MAX_CONTENT)],
            agent_id: AgentId = "",
        ) -> str:
            """Replace an outdated memory (e.g. a changed decision) with new text, keeping the old as history."""
            text = require_text("content", content, MAX_CONTENT)

            def work(memory: Any) -> str:
                full = own_id(memory, memory_id)
                new_id = memory.supersede(full, text)
                return f"Superseded memory {short(full)} with {short(new_id)}."

            return await run("mark_supersede", agent_id, work)

    if "mark_forget" in enabled:
        @server.tool()
        async def mark_forget(
            memory_id: Annotated[str, bounded("Id of the memory to delete (short id is enough); "
                                              "or use match.", maxLength=100)] = "",
            match: Annotated[str, bounded("Forget every memory containing this exact text "
                                          "(case-insensitive) instead of one id.", maxLength=500)] = "",
            secure: Annotated[bool, bounded("Also erase the text from the database file.")] = False,
            agent_id: AgentId = "",
        ) -> str:
            """Delete memories for good (stale, wrong or sensitive): one by id, or all containing a text.

            Facts taken from them are retracted. Cannot be undone; prefer
            mark_supersede when a memory is merely outdated.
            """
            if bool(memory_id.strip()) == bool(match.strip()):
                fail("give exactly one of memory_id or match")

            def work(memory: Any) -> str:
                if memory_id.strip():
                    forgotten = memory.forget(own_id(memory, memory_id), secure=secure)
                else:
                    forgotten = memory.forget(match=match.strip(), secure=secure)
                if secure and forgotten:
                    memory.vacuum()  # the promise: the text is gone from disk when this returns
                erased = " and erased them from disk" if secure and forgotten else ""
                return f"Forgot {forgotten} memor{'y' if forgotten == 1 else 'ies'}{erased}."

            return await run("mark_forget", agent_id, work)
