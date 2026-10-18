# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 MARK Contributors
"""Unknown graph block ids are an error, not an empty result.

``retrieve_sync(block_id=...)`` with a typo used to return nothing (and MCP
reported a "critical memory gap"); now it raises
:class:`mark.errors.UnknownBlockError` naming the known blocks, as
``mark.memory.recall(blocks=[...])`` already did for named blocks.
"""
from __future__ import annotations

from typing import Any


def check_block_ids(store: Any, agent_id: str, block_id: str | None, block_ids: list[str] | None) -> None:
    """Raise UnknownBlockError for block ids *agent_id* never created nor wrote to."""
    wanted = [b for b in [block_id, *(block_ids or [])] if b]
    unknown_of = getattr(store, "unknown_block_ids", None)
    if not wanted or unknown_of is None:
        return
    unknown = unknown_of(agent_id, wanted)
    if unknown:
        from mark.errors import UnknownBlockError

        raise UnknownBlockError(unknown, store.known_block_ids(agent_id, limit=20))
