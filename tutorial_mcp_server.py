"""
MARK MCP server entry-point for the tutorial.

Spawned as a subprocess by tutorial.py. Reads the store path from argv[1].

    python tutorial_mcp_server.py /path/to/.mark
"""

from __future__ import annotations

import sys
from pathlib import Path

from mark.runtime import Mark
from mark.adapters.mcp import create_mark_mcp_server_from_local


def main() -> None:
    store_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".mark")
    project_path = store_path.parent

    mark = Mark.local(project_path=project_path, store_path=store_path)
    server = create_mark_mcp_server_from_local(
        mark,
        name="mark-tutorial-memory",
        agent_id="coding-agent",
    )
    server.run()


if __name__ == "__main__":
    main()
