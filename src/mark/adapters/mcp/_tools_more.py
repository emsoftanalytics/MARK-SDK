"""MCP tools over local memory: ``mark_fact``, ``mark_timeline``, ``mark_brief``, ``mark_forget``.

They need direct memory access (``backend.memory_for(agent_id)``, which
``LocalMarkBackend`` provides); with another backend they fail with a clear
tool error. All blocking work runs off the event loop.

* ``mark_fact``      get / set / history / list of bi-temporal facts — the
  right tool for values that change ("deploy port", "user city").
* ``mark_timeline``  every event about one subject, oldest first.
* ``mark_brief``     the one-call session brief (``mark.context.brief``).
* ``mark_forget``    delete one memory by id (the facade's ``forget``; its
  cascade to derived facts is the facade's job).
"""
from __future__ import annotations

from typing import Annotated, Any, Callable, Literal

from mark.adapters.backend import run_off_loop
from mark.adapters.mcp._common import AgentId, bounded, clean_error, fail, guard, require_text

Subject = Annotated[str, bounded("What the fact is about, e.g. 'user' or 'checkout-api'.",
                                 minLength=1, maxLength=200)]
Attribute = Annotated[str, bounded("The property, e.g. 'city' or 'port'; empty = all.", maxLength=200)]


def register_memory_tools(server: Any, backend: Any, resolve: Callable[[str], str], *,
                          enabled: set[str]) -> None:
    """Add the enabled memory tools to *server*."""

    async def run(name: str, agent_id: str, work: Callable[[Any], str]) -> str:
        memory_for = getattr(backend, "memory_for", None)
        if memory_for is None:
            fail(f"{name} needs a local MARK backend (memory_for); this backend has none")
        resolved = resolve(agent_id)
        try:
            return await run_off_loop(lambda: work(memory_for(resolved)))
        except Exception as exc:
            if isinstance(exc, _tool_error()):
                raise
            fail(f"{name} failed: {clean_error(str(exc))}")

    if "mark_fact" in enabled:
        @server.tool()
        async def mark_fact(
            action: Annotated[Literal["get", "set", "history", "list"],
                              bounded("get: current value(s); set: record a new value; "
                                      "history: every value over time; list: subjects with facts.")],
            subject: Annotated[str, bounded("What the fact is about (not needed for list).", maxLength=200)] = "",
            attribute: Attribute = "",
            value: Annotated[str, bounded("New value (set only).", maxLength=2000)] = "",
            valid_from: Annotated[str, bounded("When the value became true, ISO date (set only; "
                                               "default now).", maxLength=40)] = "",
            agent_id: AgentId = "",
        ) -> str:
            """Read or record facts: values that change over time, with full history.

            Setting a new value keeps the old one as history, so "where did X
            live before" stays answerable.
            """
            if action != "list":
                require_text("subject", subject, 200)
            if action in ("set", "history") and not attribute.strip():
                fail(f"attribute is required for {action}")
            if action == "set":
                require_text("value", value, 2000)

            def work(memory: Any) -> str:
                facts = memory.facts
                if action == "list":
                    subjects = facts.subjects()
                    return "Subjects with facts: " + (", ".join(subjects) if subjects else "none")
                if action == "set":
                    fact = facts.set(subject.strip(), attribute.strip(), value.strip(),
                                     valid_from=valid_from.strip() or None)
                    return f"Recorded: {guard(fact.as_text())}"
                if action == "history":
                    rows = facts.history(subject.strip(), attribute.strip())
                    if not rows:
                        return f"No facts for {subject} {attribute}."
                    return "\n".join(guard(f.as_text()) for f in rows)
                rows = facts.current(subject.strip(), attribute.strip() or None)
                if not rows:
                    return f"No current facts for {subject}{' ' + attribute if attribute else ''}."
                return "\n".join(guard(f.as_text()) for f in rows)

            return await run("mark_fact", agent_id, work)

    if "mark_timeline" in enabled:
        @server.tool()
        async def mark_timeline(
            subject: Subject,
            attribute: Attribute = "",
            limit: Annotated[int, bounded("Newest events to keep (1-200).", minimum=1, maximum=200)] = 30,
            since: Annotated[str, bounded("Only events on/after this ISO date.", maxLength=40)] = "",
            until: Annotated[str, bounded("Only events on/before this ISO date.", maxLength=40)] = "",
            agent_id: AgentId = "",
        ) -> str:
            """Every event about one subject (facts and memories), oldest first."""
            require_text("subject", subject, 200)
            if not 1 <= limit <= 200:
                fail("limit must be between 1 and 200")

            def work(memory: Any) -> str:
                timeline = memory.timeline(subject.strip(), attribute.strip() or None, limit=limit,
                                           since=since.strip() or None, until=until.strip() or None)
                return guard(timeline.as_text())

            return await run("mark_timeline", agent_id, work)

    if "mark_brief" in enabled:
        @server.tool()
        async def mark_brief(
            budget_tokens: Annotated[int, bounded("Size cap in tokens (chars/4), 100-4000.",
                                                  minimum=100, maximum=4000)] = 400,
            focus: Annotated[str, bounded("Optional topic to put first.", maxLength=500)] = "",
            agent_id: AgentId = "",
        ) -> str:
            """A short session brief: current facts, open TODOs, recent decisions,
            bugs and fixes, and recent changes, in one call. Useful at the start
            of a session."""
            if not 100 <= budget_tokens <= 4000:
                fail("budget_tokens must be between 100 and 4000")

            def work(memory: Any) -> str:
                from mark.context.session_brief import brief

                text = brief(memory, budget_tokens=budget_tokens, focus=focus.strip() or None).text
                return text or "<mark_briefing>\nMemory is empty: nothing to brief yet.\n</mark_briefing>"

            return await run("mark_brief", agent_id, work)

    if "mark_forget" in enabled:
        @server.tool()
        async def mark_forget(
            memory_id: Annotated[str, bounded("Full id of the memory to delete (as shown by "
                                              "mark_retrieve / mark_write).", minLength=1, maxLength=100)],
            agent_id: AgentId = "",
        ) -> str:
            """Delete one memory by id (for stale or wrong memories). Cannot be undone."""
            target = require_text("memory_id", memory_id, 100)

            def work(memory: Any) -> str:
                full = _own_id(memory, target)
                if full is None or not memory.forget(full):
                    fail(f"no memory with id {target!r} in this namespace")
                return f"Forgot memory {full}."

            return await run("mark_forget", agent_id, work)


#: Memories scanned to resolve a short id prefix (ids are shown in full by mark_retrieve).
_PREFIX_SCAN = 5000


def _own_id(memory: Any, target: str) -> str | None:
    """The full id of *target* (an id or a unique prefix of >= 8 chars) in this namespace.

    The store deletes by id alone, so ownership is checked here: an MCP
    client limited by ``allowed_agent_ids`` must not delete another
    namespace's memory by guessing its id.
    """
    store = getattr(memory, "_store", None)
    fragment = store.get(target) if store is not None else None
    if fragment is not None:
        return target if getattr(fragment, "agent_id", None) == memory.agent_id else None
    if len(target) < 8:
        return None
    matches = [f.id for f in memory.list(limit=_PREFIX_SCAN) if str(f.id).startswith(target)]
    return matches[0] if len(matches) == 1 else None


def _tool_error() -> type[Exception]:
    from mark.adapters.mcp._common import tool_error_class

    return tool_error_class()
