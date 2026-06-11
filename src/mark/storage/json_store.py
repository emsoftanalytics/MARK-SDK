"""Legacy JSON file store for simple records."""
from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from mark.memory.block import MemoryBlock
from mark.memory.record import MemoryRecord


class JsonMemoryStore:
    """Legacy local JSON persistence for block/record compatibility.

    Current MARK local runtime uses SQLite LocalMemoryStore. This store remains
    as a small MIT fallback/migration surface, not as cloud storage logic.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self._save({"blocks": [], "records": []})

    def create_block(self, block: MemoryBlock) -> MemoryBlock:
        """Create and persist a new block."""
        data = self._load()
        existing = self.get_block_by_label(block.label)
        if existing:
            return existing
        data["blocks"].append(block.to_dict())
        self._save(data)
        return block

    def get_block_by_label(self, label: str) -> MemoryBlock | None:
        """Find a block by label, or None."""
        for block in self.list_blocks():
            if block.label == label:
                return block
        return None

    def list_blocks(self) -> list[MemoryBlock]:
        """Return all blocks."""
        data = self._load()
        return [MemoryBlock.from_dict(item, store=self) for item in data["blocks"]]

    def write_record(self, record: MemoryRecord) -> MemoryRecord:
        """Persist a record."""
        data = self._load()
        data["records"].append(record.to_dict())
        self._save(data)
        return record

    def list_records(self, block_id: str | None = None) -> list[MemoryRecord]:
        """Return records, optionally filtered by block."""
        data = self._load()
        records = [MemoryRecord.from_dict(item) for item in data["records"]]
        if block_id is None:
            return records
        return [record for record in records if record.block_id == block_id]

    def update_record(self, record: MemoryRecord) -> MemoryRecord:
        """Persist changes to an existing record."""
        data = self._load()
        for index, item in enumerate(data["records"]):
            if item["id"] == record.id:
                data["records"][index] = replace(record, last_accessed_at=datetime.now(timezone.utc)).to_dict()
                self._save(data)
                return MemoryRecord.from_dict(data["records"][index])
        raise KeyError(f"Memory record not found: {record.id}")

    def _load(self) -> dict[str, list[dict[str, object]]]:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid MARK memory store: {self.path}") from exc
        return {
            "blocks": list(data.get("blocks", [])),
            "records": list(data.get("records", [])),
        }

    def _save(self, data: dict[str, list[dict[str, object]]]) -> None:
        self.path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
