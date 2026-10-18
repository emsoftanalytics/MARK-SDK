"""Paging and counting an agent's fragments at the storage layer (no silent caps).

``page_fragments``/``count_fragments`` back ``memory.list()``/``count()``;
``ids_with_prefix`` expands the 8-character short ids the CLI and MCP print;
``tag_values`` lists block labels. Mixed into the store through
:class:`mark.store._sqlite.lifecycle.LifecycleStoreMixin`.
"""
from __future__ import annotations

from typing import Any

from mark.store._sqlite.codec import _chunks, _SQL_CHUNK


class PagingStoreMixin:
    """List, count and look up fragments by filter or id prefix."""

    def _page_where(
        self, agent_id: str, *, states: Any = None, tags: list[str] | None = None,
        exclude_tags: list[str] | None = None, session_id: str | None = None,
    ) -> tuple[str, list[object]]:
        where, params = self._fragment_filters(
            agent_id, scope=None, states=states, tier=None, session_id=session_id,
            session_prefix=None, tags=tags, block_id=None, block_ids=None,
        )
        for tag in exclude_tags or ():
            where += " AND NOT EXISTS (SELECT 1 FROM json_each(tags) WHERE value = ?)"
            params.append(tag)
        return where, params

    def page_fragments(
        self, agent_id: str, *, limit: int | None, offset: int = 0, include_unreadable: bool = False,
        **filters: Any,
    ) -> list[Any]:
        """One page of an agent's fragments in ``list_by_agent`` order (rowid breaks ties).

        Memories an encrypted store could not decrypt (``mark:unreadable``,
        see ``unreadable.py``) are left out unless ``include_unreadable``; then
        their stored text is returned as it is and a warning is logged.
        """
        from mark.store._sqlite.unreadable import UNREADABLE_TAG

        if not include_unreadable:
            filters["exclude_tags"] = [*(filters.get("exclude_tags") or ()), UNREADABLE_TAG]
        where, params = self._page_where(agent_id, **filters)
        sql = f"SELECT * FROM fragments WHERE {where} ORDER BY importance DESC, created_at DESC, rowid DESC"
        if limit is not None or offset:
            sql += " LIMIT ? OFFSET ?"
            params += [-1 if limit is None else int(limit), max(0, int(offset))]
        out = [self._row_to_fragment(dict(row)) for row in self._fetchall(sql, params)]
        return self._raw_unreadable(out) if include_unreadable else out

    def _raw_unreadable(self, fragments: list[Any]) -> list[Any]:
        """Show unreadable memories as stored (they were never encrypted), with a warning."""
        import logging

        from mark.store._sqlite.codec import _as_bytes
        from mark.store._sqlite.unreadable import UNREADABLE_TAG

        bad = {f.id for f in fragments if UNREADABLE_TAG in f.tags}
        if not bad:
            return fragments
        logging.getLogger("mark.store.sqlite").warning(
            "including %d unreadable memories as stored (not encrypted with this store's key)", len(bad))
        raw = {str(r[0]): _as_bytes(r[1]).decode("utf-8", "replace") for chunk in _chunks(list(bad), _SQL_CHUNK)
               for r in self._fetchall(f"SELECT id, content FROM fragments WHERE id IN ({','.join('?' * len(chunk))})", chunk)}
        return [f.model_copy(update={"content": raw[f.id]}) if f.id in raw else f for f in fragments]

    def count_fragments(self, agent_id: str, **filters: Any) -> int:
        """How many fragments ``page_fragments`` would return without a limit."""
        from mark.store._sqlite.unreadable import UNREADABLE_TAG

        if not filters.pop("include_unreadable", False):
            filters["exclude_tags"] = [*(filters.get("exclude_tags") or ()), UNREADABLE_TAG]
        where, params = self._page_where(agent_id, **filters)
        row = self._fetchone(f"SELECT COUNT(*) FROM fragments WHERE {where}", params)
        return int(row[0]) if row else 0

    def ids_with_prefix(self, agent_id: str, prefix: str, limit: int = 2) -> list[str]:
        """Fragment ids of an agent starting with *prefix* (CLI short ids); at most *limit*."""
        rows = self._fetchall(
            "SELECT id FROM fragments WHERE agent_id = ? AND substr(id, 1, ?) = ? LIMIT ?",
            (agent_id, len(prefix), prefix, int(limit)),
        )
        return [str(row[0]) for row in rows]

    def tag_values(self, agent_id: str, prefix: str) -> list[str]:
        """Distinct tags of an agent starting with *prefix* (e.g. every ``block:`` label)."""
        rows = self._fetchall(
            "SELECT DISTINCT j.value FROM fragments AS f, json_each(f.tags) AS j"
            " WHERE f.agent_id = ? AND j.value LIKE ? ESCAPE '\\'",
            (agent_id, prefix.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"),
        )
        return sorted(str(row[0]) for row in rows)

    def known_block_ids(self, agent_id: str, limit: int | None = None) -> list[str]:
        """Graph block ids of an agent: blocks it created and blocks its fragments belong to."""
        sql = ("SELECT id FROM blocks WHERE agent_id = ? UNION "
               "SELECT DISTINCT block_id FROM fragments WHERE agent_id = ? AND block_id IS NOT NULL ORDER BY 1")
        params: list[object] = [agent_id, agent_id]
        if limit is not None:
            sql += " LIMIT ?"
            params.append(int(limit))
        return [str(row[0]) for row in self._fetchall(sql, params)]

    def unknown_block_ids(self, agent_id: str, block_ids: list[str]) -> list[str]:
        """The ids in *block_ids* that are no block of *agent_id* (see :meth:`known_block_ids`)."""
        wanted = list(dict.fromkeys(str(b) for b in block_ids if b))
        found: set[str] = set()
        for chunk in _chunks(wanted, _SQL_CHUNK):
            marks = ",".join("?" for _ in chunk)
            rows = self._fetchall(
                f"SELECT id FROM blocks WHERE agent_id = ? AND id IN ({marks}) UNION "
                f"SELECT DISTINCT block_id FROM fragments WHERE agent_id = ? AND block_id IN ({marks})",
                [agent_id, *chunk, agent_id, *chunk])
            found |= {str(row[0]) for row in rows}
        return [b for b in wanted if b not in found]
