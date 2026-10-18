"""``mark forget --match TEXT``: show what would go, then ask before deleting.

The match is a whole-word/phrase match (see
:mod:`mark.memory.facade.forget_match`). The command first runs a dry run and
prints the count and one preview line per memory, then:

* ``--dry-run``: stops there (exit 0);
* ``--yes``: forgets them without asking;
* on a terminal: asks ``Forget N memories? [y/N]``;
* otherwise (an agent's shell, a pipe): refuses with the count, exit 1, and
  says to add ``--yes``.

More than the cap (50) also needs ``--all``. Standard library only.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

#: Preview lines printed before the question (the rest are counted).
_SHOWN = 10


def _plural(count: int) -> str:
    return "1 memory" if count == 1 else f"{count} memories"


def forget_matching(args: Any, db: Path, payload: dict[str, Any]) -> int:
    """Run the dry run / confirm / forget sequence; returns the exit code."""
    from mark._cli.client import execute
    from mark._cli.output import print_result

    use_server = not args.no_server
    found = execute(db, "forget", {**payload, "dry_run": True}, use_server=use_server)
    count = int(found["count"])
    if count == 0:
        print(f"no memory matches {found['match']!r} (whole words); nothing forgotten")
        return 0
    print(f"{_plural(count)} match{'es' if count == 1 else ''} {found['match']!r}:")
    for line in found["previews"][:_SHOWN]:
        print(f"  {line}")
    if count > _SHOWN:
        print(f"  ... and {count - _SHOWN} more")
    if getattr(args, "dry_run", False):
        return 0
    if found.get("over_cap") and not payload.get("all"):
        print(f"mark: {_plural(count)} match, more than the safety cap of {found['cap']}; "
              "add --all to forget them all", file=sys.stderr)
        return 1
    if not getattr(args, "yes", False) and not _confirmed(count):
        return 1
    result = execute(db, "forget", payload, use_server=use_server)
    print_result(args, "forget", result)
    return 0


def _confirmed(count: int) -> bool:
    """Ask on a terminal; refuse (with how to proceed) when nobody can answer."""
    stdin = sys.stdin
    if stdin is None or not stdin.isatty():
        print(f"mark: not forgetting {_plural(count)} without confirmation; "
              "rerun with --yes to forget them (or --dry-run to only list them)", file=sys.stderr)
        return False
    print(f"Forget {_plural(count)}? [y/N] ", end="", flush=True)
    answer = stdin.readline().strip().lower()
    if answer in {"y", "yes"}:
        return True
    print("mark: nothing forgotten", file=sys.stderr)
    return False
