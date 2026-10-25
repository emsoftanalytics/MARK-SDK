"""MCP tools over local memory: ``mark_fact``, ``mark_timeline``, ``mark_brief`` (+ lifecycle tools).

They need direct memory access (``backend.memory_for(agent_id)``, which
``LocalMarkBackend`` provides); with another backend they fail with a clear
tool error. All blocking work runs off the event loop.

* ``mark_fact``      get / set / history / list of bi-temporal facts — the
  right tool for values that change ("deploy port", "user city").
* ``mark_timeline``  every event about one subject, oldest first.
* ``mark_brief``     the one-call session brief (``mark.context.brief``).
* ``mark_list`` / ``mark_resolve`` / ``mark_supersede`` / ``mark_forget``
  live in :mod:`mark.adapters.mcp._tools_lifecycle` and share ``run``.
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
                                      "history: every value over time; list: subjects with facts "
                                      "(with subject: that subject's current facts).")],
            subject: Annotated[str, bounded("What the fact is about (optional for list).", maxLength=200)] = "",
            attribute: Attribute = "",
            value: Annotated[str, bounded("New value (set only).", maxLength=2000)] = "",
            valid_from: Annotated[str, bounded("When the value became true (set only; default now): "
                                               "ISO date or words such as 'yesterday'. A future "
                                               "date schedules the value.", maxLength=40)] = "",
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
                if action == "list" and subject.strip():
                    rows = facts.current(subject.strip())   # one subject: its current facts
                    if not rows:
                        return f"No current facts for {guard(subject.strip())}."
                    return "\n".join(guard(f.as_text()) for f in rows)
                if action == "list":
                    # Subjects as written (not the normalised lookup keys).
                    subjects = sorted({f.subject for f in facts.current()}, key=str.casefold)
                    return "Subjects with facts: " + (guard(", ".join(subjects)) if subjects else "none")
                if action == "set":
                    fact = facts.set(subject.strip(), attribute.strip(), value.strip(),
                                     valid_from=valid_from.strip() or None)
                    verb = "Scheduled (not current until then)" if fact.is_scheduled else "Recorded"
                    return f"{verb}: {guard(fact.as_text())}"
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

    from mark.adapters.mcp._tools_lifecycle import register_lifecycle_tools

    register_lifecycle_tools(server, run, enabled=enabled)


def _tool_error() -> type[Exception]:
    from mark.adapters.mcp._common import tool_error_class

    return tool_error_class()
