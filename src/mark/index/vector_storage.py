"""On-disk format of a VectorIndex: versioned JSON written atomically.

The file is written to a temporary sibling, fsynced, then moved into place
with ``os.replace``, so a crash mid-save never leaves a truncated index.
Legacy v1 files (no "agents" map) load with every vector in the shared
``None`` partition.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterator

_FORMAT = "mark.vector-index"
_FORMAT_VERSION = 2


def _write_json_atomic(target: Path, data: dict[str, Any]) -> None:
    """Write *data* as JSON to *target* atomically (temp file + fsync + os.replace)."""
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(data, handle, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, target)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def _index_payload(dim: int | None, owner: dict, content: dict, partitions: dict) -> dict[str, Any]:
    """Serialisable form of an index (caller holds the index lock)."""
    return {
        "format": _FORMAT,
        "version": _FORMAT_VERSION,
        "dim": dim,
        "vectors": {fragment_id: partitions[agent].get(fragment_id) for fragment_id, agent in owner.items()},
        "content": dict(content),
        "agents": dict(owner),
    }


def _read_index_file(path: Path) -> Iterator[tuple[str, list[float], str, str | None]]:
    """Yield (fragment id, vector, content, agent id) from a saved index file."""
    data = json.loads(path.read_text(encoding="utf-8"))
    content = {str(key): str(value) for key, value in data.get("content", {}).items()}
    agents = data.get("agents", {}) or {}
    for fragment_id, vector in data.get("vectors", {}).items():
        owner = agents.get(fragment_id)
        yield (
            str(fragment_id),
            [float(value) for value in vector],
            content.get(str(fragment_id), ""),
            None if owner is None else str(owner),
        )
