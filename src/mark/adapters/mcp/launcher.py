"""``python -m mark.adapters.mcp``: start a stdio MCP server over a MARK store.

    python -m mark.adapters.mcp --db .mark/memory.db --briefing --preset small

Path handling is the one rule of :func:`mark.runtime_data.store_db_file`,
shared with ``Mark.local`` and ``mark --db``: a ``.db`` file is the file, a
``.mark`` folder (or a folder already holding ``memory.db``) is the store
folder, and any other folder is a project whose store is
``<dir>/.mark/memory.db``; default ``$MARK_DB``, else ``./.mark/memory.db``.
The CLI's ``mark mcp`` verb is a thin wrapper over :func:`main` / :func:`run_mcp`.

Nothing is printed on stdout: stdio MCP uses it for the protocol.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Sequence


def resolve_db(path: str | Path | None) -> Path:
    """The SQLite file for *path* (file, folder or None), exactly as the CLI resolves it."""
    from mark.runtime_data import store_db_file

    return store_db_file(str(path) if path else os.environ.get("MARK_DB") or str(Path(".mark") / "memory.db"))


def build_server(db: str | Path | None = None, *, agent_id: str = "__mark__", briefing: bool = False,
                 preset: str = "default", allowed_agent_ids: Sequence[str] | None = None,
                 tools: Sequence[str] | None = None, name: str = "mark-memory") -> tuple[Any, Any]:
    """``(server, runtime)`` over the store at *db*; the caller shuts the runtime down."""
    from mark.adapters.backend import LocalMarkBackend
    from mark.adapters.mcp import create_mark_mcp_server
    from mark.memory.runtime import MarkRuntime

    from mark._private_fs import ensure_private_dir

    file = resolve_db(db)
    ensure_private_dir(file.parent)  # 0700 when created here, as Mark.local and the CLI do
    runtime = MarkRuntime.local(store_path=file)
    server = create_mark_mcp_server(
        LocalMarkBackend(runtime, default_agent_id=agent_id), name=name, default_agent_id=agent_id,
        allowed_agent_ids=allowed_agent_ids, briefing=briefing, briefing_preset=preset, tools=tools,
    )
    return server, runtime


def run_mcp(db: str | Path | None = None, *, transport: str = "stdio", **options: Any) -> None:
    """Serve MCP over *transport* until the client disconnects; the store is closed after."""
    server, runtime = build_server(db, **options)
    try:
        server.run(transport=transport)
    finally:
        runtime.shutdown()


def parser() -> argparse.ArgumentParser:
    """Arguments of ``python -m mark.adapters.mcp`` (and the CLI's ``mark mcp``)."""
    from mark.adapters.mcp import ALL_TOOLS

    ap = argparse.ArgumentParser(prog="mark mcp", description="Serve MARK memory over MCP (stdio).")
    ap.add_argument("--db", help="SQLite store file or .mark directory (default: $MARK_DB or ./.mark/memory.db)")
    ap.add_argument("--agent", default="__mark__", help="default memory namespace")
    ap.add_argument("--briefing", action="store_true", help="also expose mark_briefing (push context)")
    ap.add_argument("--preset", default="default", choices=("default", "small", "tiny"),
                    help="briefing size for the client's model")
    ap.add_argument("--allow-agent", action="append", dest="allowed", metavar="ID",
                    help="restrict clients to these namespaces (repeatable)")
    ap.add_argument("--tools", help=f"comma-separated subset of: {', '.join(ALL_TOOLS)}")
    return ap


def main(argv: Sequence[str] | None = None) -> None:
    args = parser().parse_args(argv)
    tools = [t.strip() for t in args.tools.split(",") if t.strip()] if args.tools else None
    run_mcp(args.db, agent_id=args.agent, briefing=args.briefing, preset=args.preset,
            allowed_agent_ids=args.allowed, tools=tools)
