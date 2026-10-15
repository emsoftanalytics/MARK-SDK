"""On-disk format of a VectorIndex: versioned JSON written atomically.

The file is written to a temporary sibling, fsynced, then moved into place
with ``os.replace``, so a crash mid-save never leaves a truncated index.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

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
