"""Record lines of the session brief: retired records, bug -> fix pairs, labels.

Split from ``session_brief`` (a10, agent test N2): the brief listed a
superseded decision as current, showed resolved bugs like open ones and put
a bug and its fix on separate, unrelated-looking lines. The rules:

* **Retired** records (``supersede``/``resolve``, see ``mark.context.outdated``)
  and records replaced by a later record of the same kind ("switched to
  Postgres instead of SQLite") never appear as current TODOs, decisions or
  changes.
* A **bug** and the fix that closes it share one line:
  ``- 2026-10-03 BUG: dates shift in UTC-5 -> FIX: store dates as UTC``. A fix
  pairs with a bug through ``metadata`` (``resolves``/``closes``/``supersedes``),
  by naming its id, or by repeating most of its words.
* A bug closed by ``resolve()`` without a fix record is marked "(resolved)".
* Untrusted memories are labelled "(untrusted)" and boundary-tag lookalikes
  are escaped, as in every briefing renderer.
"""
from __future__ import annotations

import re
from typing import Any

from mark.context.outdated import replaced_by, resolved, retired
from mark.context.relevance import content_words
from mark.context.watch_kinds import kind_of, when_of
from mark.context.watch_lines import escape_tags, is_untrusted
from mark.security.boundary import defuse_boundaries
from mark.context.watch_support import _one_line

_LEADING_DATE = re.compile(r"^\s*\[?\d{4}-\d{2}-\d{2}\]?\s*")
#: Share of a bug's content words a fix must repeat to pair with it (no explicit link).
_PAIR_OVERLAP = 0.4
_KIND_WORDS = frozenset({"bug", "fix", "fixed", "error"})


def _meta(fragment: Any) -> dict[str, Any]:
    meta = getattr(fragment, "metadata", None)
    return meta if isinstance(meta, dict) else {}


def _links(fragment: Any) -> set[str]:
    out: set[str] = set()
    for key in ("resolves", "closes", "supersedes", "fixes"):
        value = _meta(fragment).get(key)
        out.update(str(v) for v in (value if isinstance(value, (list, tuple)) else [value]) if v)
    return out


def current_records(fragments: list[Any]) -> list[Any]:
    """*fragments* without retired records and records a later one of their kind replaced.

    Resolved bugs stay (they are shown marked); only their superseded copies go.
    """
    keep = [f for f in fragments if not retired(f) or (kind_of(f) == "bug" and resolved(f))]
    by_kind: dict[str | None, list[Any]] = {}
    for fragment in keep:
        by_kind.setdefault(kind_of(fragment), []).append(fragment)
    replaced: set[str] = set()
    for kind, group in by_kind.items():
        if kind in ("decision", "change", "todo"):
            replaced |= set(replaced_by(group, _when_key))
    return [f for f in keep if f.id not in replaced]


def _when_key(fragment: Any) -> tuple[int, float]:
    when = when_of(fragment)
    return (1, when.timestamp()) if when is not None else (0, 0.0)


def fix_pairs(bugs: list[Any], fixes: list[Any]) -> dict[str, Any]:
    """``{bug id: fix}`` for every bug a fix record closes (see module rules)."""
    pairs: dict[str, Any] = {}
    used: set[str] = set()
    for bug in bugs:
        words = content_words(bug.content) - _KIND_WORDS
        for fix in fixes:
            if fix.id in used:
                continue
            linked = bug.id in _links(fix) or (bug.id[:8] and bug.id[:8] in (fix.content or ""))
            overlap = len(words & content_words(fix.content)) / len(words) if words else 0.0
            if linked or overlap >= _PAIR_OVERLAP:
                pairs[bug.id] = fix
                used.add(fix.id)
                break
    return pairs


def record_line(fragment: Any, line_chars: int, *, with_id: bool = False, fix: Any = None) -> str:
    """``- 2026-10-03 content`` with labels, a short id for TODOs and an optional fix."""
    when = when_of(fragment)
    content = getattr(fragment, "content", "") or ""
    dated = _LEADING_DATE.match(content) is not None      # the text already starts with its date
    date = f"{when.date().isoformat()} " if when is not None and not dated else ""
    label = "(untrusted) " if is_untrusted(fragment) else ""
    if fix is None and kind_of(fragment) == "bug" and resolved(fragment):
        label += "(resolved) "
    body = escape_tags(content)
    if fix is not None:
        body = f"{body.rstrip()} -> {_LEADING_DATE.sub('', escape_tags(fix.content or ''), count=1)}"
    elif _meta(fragment).get("resolution"):
        body = f"{body.rstrip()} -> fix: {escape_tags(str(_meta(fragment)['resolution']))}"
    suffix = f" (id {fragment.id[:8]})" if with_id and fragment.id else ""
    return f"- {label}{date}{_one_line(defuse_boundaries(body), line_chars)}{suffix}"
