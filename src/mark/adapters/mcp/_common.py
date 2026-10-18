"""Shared plumbing for the MARK MCP tools: errors, argument checks and plain-text output.

**Errors.** A failure the caller can act on (empty query, unknown memory id,
value out of range, a store error) is raised as the SDK's ``ToolError``, so
the client gets ``isError=True`` and one clear sentence instead of a
"success" string or a pydantic dump (usage test m1). Arguments are checked
here rather than by pydantic constraints for the same reason; the bounds
are still advertised in each tool's JSON schema (``bounded``).

Arguments that fail the SDK's own schema validation (a string for an
integer...) are reported the same way: :func:`install_error_cleaner` turns
the pydantic dump (with its URL) into one MARK-style sentence.

**Output.** Tool results are plain text for a model to read. Memory content
is not HTML-escaped ("->" stays "->", agent report #14); only text that
could forge a context boundary (``<memory_context>``, ``<mark_briefing>``)
is neutralised, so stored text can never close the block it is shown in,
and NUL/control characters are dropped. Ids are the 8-character short ids
the CLI prints; every tool that takes an id accepts them.

**Size.** ``render_retrieval`` is bounded: each memory shows its best
passage within ``max_item_chars``, the whole reply fits ``budget_tokens``
(the first item is cut, never dropped), and when the gap report says memory
does not hold the answer, results below the relevance floor are left out.
"""
from __future__ import annotations

import re
from typing import Annotated, Any, NoReturn

from pydantic import Field

from mark.context.budget import estimate_tokens
from mark.context.provenance import UNTRUSTED_TAG, is_assistant

_BOUNDARY = re.compile(r"<(/?)(memory_context|mark_briefing|mark_memory)", re.IGNORECASE)
_PYDANTIC = re.compile(r"validation error", re.IGNORECASE)
_LEADING_DATE = re.compile(r"^\s*\[?\d{4}-\d{2}-\d{2}")
# Control characters other than newline/tab (NUL included): never sent to a client.
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
MAX_QUERY = 2000
MAX_CONTENT = 16000
#: Characters of an id shown in tool output (the CLI's short id).
SHORT_ID = 8
#: Default per-memory cap in mark_retrieve: a long memory shows its best passage.
ITEM_CHARS = 600
#: Calibrated relevance below which results are dropped when memory lacks the answer.
RELEVANCE_FLOOR = 0.35


def tool_error_class() -> type[Exception]:
    """The installed SDK's ToolError (mcp 2.x ``mcpserver``, 1.x ``fastmcp``)."""
    try:
        from mcp.server.mcpserver.exceptions import ToolError
        return ToolError
    except ImportError:
        pass
    try:
        from mcp.server.fastmcp.exceptions import ToolError  # type: ignore[no-redef]
        return ToolError
    except ImportError:  # pragma: no cover - the adapter requires mcp
        return RuntimeError


def fail(message: str) -> NoReturn:
    """Raise a tool error the client sees as ``isError=True`` with *message*."""
    raise tool_error_class()(message)


def clean_error(text: str) -> str:
    """One readable line from a backend error (pydantic dumps reduced to field + reason)."""
    text = str(text or "unknown error").strip()
    if not _PYDANTIC.search(text):
        return text.splitlines()[0][:300]
    lines = [line.strip() for line in text.splitlines()[1:] if line.strip()]
    if len(lines) >= 2:
        reason = re.sub(r"\s*\[type=.*$", "", lines[1])
        return f"invalid {lines[0]}: {reason}"[:300]
    return "invalid input"


def bounded(description: str, **bounds: Any) -> Any:
    """``Field`` that documents *bounds* (minimum, maximum, maxLength) in the schema only."""
    return Field(description=description, json_schema_extra=bounds or None)


def require_text(name: str, value: str, limit: int) -> str:
    """*value* stripped; empty or over *limit* characters is a tool error."""
    text = (value or "").strip()
    if not text:
        fail(f"{name} must not be empty")
    if len(text) > limit:
        fail(f"{name} is {len(text)} characters; the limit is {limit}")
    return text


def require_range(name: str, value: float, low: float, high: float) -> float:
    if isinstance(value, bool) or not low <= value <= high:
        fail(f"{name} must be between {low:g} and {high:g} (got {value!r})")
    return value


def guard(text: str) -> str:
    """*text* with boundary-tag look-alikes neutralised and control characters removed."""
    return _BOUNDARY.sub(lambda m: f"&lt;{m.group(1)}{m.group(2)}", _CONTROL.sub("", text or ""))


def short(fragment_id: str) -> str:
    """The 8-character id shown to clients (any unambiguous prefix is accepted back)."""
    return str(fragment_id or "")[:SHORT_ID]


def _label(fragment: Any) -> str:
    tags = getattr(fragment, "tags", None) or ()
    if is_assistant(fragment):
        return "(assistant, unverified) "
    if UNTRUSTED_TAG in tags:
        return "(untrusted) "
    return ""


def _date(fragment: Any) -> str:
    """"2026-10-02 " (or "" when the text already starts with its own date)."""
    from mark.context.watch_kinds import when_of

    if _LEADING_DATE.match(getattr(fragment, "content", "") or ""):
        return ""
    when = when_of(fragment)
    return f"{when.date().isoformat()} " if when is not None else ""


def render_retrieval(result: Any, *, max_results: int, budget_tokens: int = 0,
                     max_item_chars: int = ITEM_CHARS, states: dict[str, str] | None = None) -> str:
    """A retrieval result as bounded plain text: facts, then memories with date and short id.

    Each memory is cut to its best passage (``passage_window``, as
    ``RetrievalResult.as_context`` does) within ``max_item_chars``.
    ``budget_tokens`` > 0 caps the reply (characters / 4): the first line
    that does not fit is cut to the remaining room, later ones are counted
    in a final line. When the gap report is high/critical, memories below
    :data:`RELEVANCE_FLOOR` are left out (they are not the answer).
    Each memory shows its relevance ("score 0.82") and, when it is not
    current, "(superseded)", "(resolved)" or "(outdated)" (*states*, see
    :mod:`mark.memory.result_states`; lifecycle tags are read when omitted).
    """
    from mark.memory.result_states import states as states_of

    from mark.intelligence.passage_window import window

    query_text = str(getattr(result, "query", ""))
    query = guard(" ".join(query_text.split()))[:200].replace('"', "'")
    head, tail = f'<memory_context query="{query}">', "</memory_context>"
    lines: list[str] = []
    facts = [f for f in getattr(result, "facts", None) or [] if getattr(f, "is_current", True)]
    if facts:
        lines.append("Known facts (current):")
        lines += [f"- {guard(f.as_text())}" for f in facts]
    gap = getattr(result, "gap_report", None)
    severity = getattr(getattr(gap, "severity", None), "value", "none")
    fragments = _relevant(result, severity)[:max_results]
    labels = states if states is not None else states_of(result)
    relevance = dict(zip((f.id for f in getattr(result, "fragments", None) or []),
                         getattr(result, "relevance", None) or getattr(result, "scores", None) or []))
    if fragments:
        lines.append("Memories (most relevant first):")
        for fragment in fragments:
            text, _ = window(" ".join(guard(fragment.content).split()), query_text, max_item_chars)
            score = f"score {relevance[fragment.id]:.2f} " if fragment.id in relevance else ""
            state = f"({labels[fragment.id]}) " if fragment.id in labels else ""
            lines.append(f"- {_date(fragment)}[{short(fragment.id)}] {score}{state}{_label(fragment)}{text}")
    if not lines:
        lines.append("Nothing in memory matches this query.")
    if severity in ("high", "critical"):
        lines.append(f"Memory gap ({severity}): {guard(gap.reason)}")
    if budget_tokens <= 0:
        return "\n".join([head, *lines, tail])
    return "\n".join([head, *_fit(lines, budget_tokens - estimate_tokens(head + tail) - 2, budget_tokens), tail])


def _relevant(result: Any, severity: str) -> list[Any]:
    """Fragments to show: all, or (memory lacks the answer) only those above the floor."""
    fragments = list(getattr(result, "fragments", None) or [])
    relevance = list(getattr(result, "relevance", None) or [])
    if severity not in ("high", "critical") or len(relevance) != len(fragments):
        return fragments
    return [f for f, score in zip(fragments, relevance) if score >= RELEVANCE_FLOOR]


def _fit(lines: list[str], room: int, budget_tokens: int) -> list[str]:
    """The lines that fit *room* tokens; a line that does not fit is cut, never silently dropped."""
    kept: list[str] = []
    used = 0
    for index, line in enumerate(lines):
        cost = estimate_tokens(line) + 1
        if used + cost <= room:
            kept.append(line)
            used += cost
            continue
        left_chars = (room - used - 1) * 4
        if left_chars >= 80:  # worth showing a cut line (e.g. one huge first memory)
            kept.append(line[: left_chars - 1].rstrip() + "…")
            index += 1
        dropped = len(lines) - index
        if dropped:
            kept.append(f"[{dropped} more line(s) not shown: budget_tokens={budget_tokens}]")
        break
    return kept


def validation_sentence(tool: str, exc: BaseException) -> str | None:
    """One MARK-style sentence for an SDK schema-validation failure, else None."""
    cause = exc.__cause__ if exc.__cause__ is not None else exc
    errors = getattr(cause, "errors", None)
    if not callable(errors) or type(cause).__name__ != "ValidationError":
        return None
    parts = []
    for item in errors():
        field = ".".join(str(loc) for loc in item.get("loc", ()) if loc != f"{tool}Arguments") or "arguments"
        parts.append(f"{field}: {str(item.get('msg', 'invalid value')).lower()}")
    return f"{tool} got invalid arguments: " + "; ".join(parts) + "."


def install_error_cleaner(server: Any) -> None:
    """Report argument problems as one sentence: unknown names, and schema failures.

    The SDK validates arguments against the schema before a tool runs and
    returns ``str(ValidationError)``; wrapping ``call_tool`` (which the
    protocol handler calls) rewrites only those errors. Arguments the tool
    does not have are rejected by name instead of silently ignored (a
    misspelt ``extract_fact`` used to store the memory without facts).
    """
    original = server.call_tool
    error_class = tool_error_class()
    known: dict[str, set[str]] = {}

    async def call_tool(name: str, arguments: dict[str, Any], *args: Any, **kwargs: Any) -> Any:
        unknown = await _unknown_arguments(server, known, name, arguments)
        if unknown:
            valid = ", ".join(sorted(known[name])) or "none"
            raise error_class(f"{name} got unknown argument(s) {', '.join(map(repr, unknown))}; "
                              f"valid arguments: {valid}.")
        try:
            return await original(name, arguments, *args, **kwargs)
        except error_class as exc:
            sentence = validation_sentence(name, exc)
            if sentence is None:
                raise
            raise error_class(sentence) from None

    server.call_tool = call_tool


async def _unknown_arguments(server: Any, known: dict[str, set[str]], name: str, arguments: Any) -> list[str]:
    """Names in *arguments* that tool *name* does not declare ([] when the tool is unknown)."""
    if not isinstance(arguments, dict) or not arguments:
        return []
    if name not in known:
        for tool in await server.list_tools():
            schema = getattr(tool, "inputSchema", None) or getattr(tool, "input_schema", None) or {}
            known[tool.name] = set((schema.get("properties") or {}) if isinstance(schema, dict) else ())
    if name not in known:
        return []  # the SDK reports unknown tools itself
    return sorted(set(arguments) - known[name])


#: Parameter annotations shared by several tools.
AgentId = Annotated[str, Field(description="Memory namespace; empty = the server default.")]
