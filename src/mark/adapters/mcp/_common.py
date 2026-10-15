"""Shared plumbing for the MARK MCP tools: errors, argument checks and plain-text output.

**Errors.** A failure the caller can act on (empty query, unknown memory id,
value out of range, a store error) is raised as the SDK's ``ToolError``, so
the client gets ``isError=True`` and one clear sentence instead of a
"success" string or a pydantic dump (usage test m1). Arguments are checked
here rather than by pydantic constraints for the same reason; the bounds
are still advertised in each tool's JSON schema (``bounded``).

**Output.** Tool results are plain text for a model to read. Memory content
is not HTML-escaped ("->" stays "->", agent report #14); only text that
could forge a context boundary (``<memory_context>``, ``<mark_briefing>``)
is neutralised, so stored text can never close the block it is shown in.
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
MAX_QUERY = 2000
MAX_CONTENT = 16000


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
    """*text* with boundary-tag look-alikes neutralised (the only escaping done)."""
    return _BOUNDARY.sub(lambda m: f"&lt;{m.group(1)}{m.group(2)}", text or "")


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


def render_retrieval(result: Any, *, max_results: int, budget_tokens: int = 0) -> str:
    """A retrieval result as plain text: facts, then memories with date and short id.

    ``budget_tokens`` > 0 caps the output (estimated as characters / 4); what
    did not fit is counted in a final line instead of disappearing silently.
    """
    query = guard(" ".join(str(getattr(result, "query", "")).split()))[:200].replace('"', "'")
    head, tail = f'<memory_context query="{query}">', "</memory_context>"
    lines: list[str] = []
    facts = [f for f in getattr(result, "facts", None) or [] if getattr(f, "is_current", True)]
    if facts:
        lines.append("Known facts (current):")
        lines += [f"- {guard(f.as_text())}" for f in facts]
    fragments = list(getattr(result, "fragments", None) or [])[:max_results]
    if fragments:
        lines.append("Memories (most relevant first):")
        lines += [f"- {_date(f)}[{f.id}] {_label(f)}{guard(' '.join(f.content.split()))}"
                  for f in fragments]
    if not lines:
        lines.append("Nothing in memory matches this query.")
    gap = getattr(result, "gap_report", None)
    severity = getattr(getattr(gap, "severity", None), "value", "none")
    if severity in ("high", "critical"):
        lines.append(f"Memory gap ({severity}): {guard(gap.reason)}")
    if budget_tokens <= 0:
        return "\n".join([head, *lines, tail])
    kept: list[str] = []
    used = estimate_tokens(head + tail) + 2
    for line in lines:
        cost = estimate_tokens(line) + 1
        if used + cost > budget_tokens:
            break
        kept.append(line)
        used += cost
    dropped = len(lines) - len(kept)
    if dropped:
        kept.append(f"[{dropped} more line(s) not shown: budget_tokens={budget_tokens}]")
    return "\n".join([head, *kept, tail])


#: Parameter annotations shared by several tools.
AgentId = Annotated[str, Field(description="Memory namespace; empty = the server default.")]
