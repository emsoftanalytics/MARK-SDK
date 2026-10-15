"""Store side of keyed token sketches (see :mod:`mark.store._sqlite.sketches`).

``_sketch_candidates`` returns, in rowid order, the retrievable rows whose
sketch can contain every given word. Sketches are brought up to date first:
rows that are new, or whose ciphertext changed, are decrypted once and
sketched; nothing else is decrypted. Rows that fail to decrypt are always
candidates, so the normal read path isolates them exactly as before.

Sketches are kept in process memory unless ``persist_token_sketches()`` was
called for this database (a ``schema_info`` flag), in which case they are
also stored in ``fragment_sketches`` and survive restarts. The flag and the
table only exist for encrypted stores.
"""
from __future__ import annotations

from typing import Iterable

from mark.security import NoOpEncryptionProvider

from mark.store._sqlite.codec import _SQL_CHUNK, _TTL_LIVE_SQL, _as_bytes, _chunks
from mark.store._sqlite.sketches import (
    DDL_SKETCH_TRIGGER,
    DDL_SKETCHES,
    SKETCH_BITS,
    SKETCH_TABLE,
    STAMP_SQL,
    TokenSketcher,
    from_blob,
    to_blob,
)

_PERSIST_KEY = "token_sketches"
_ALL_BITS = (1 << SKETCH_BITS) - 1


class SketchMixin:
    """Keyed token sketches for encrypted stores (in memory, or persisted on request)."""

    def persist_token_sketches(self, enabled: bool = True) -> None:
        """Keep timeline sketches of this encrypted store on disk (opt-in; see sketches.py).

        Persisted sketches make the first timeline after a restart fast, at
        the cost of revealing which memories probably share words. Disabling
        drops the table. No effect on unencrypted stores (they use FTS5).
        """
        if isinstance(self.encryption, NoOpEncryptionProvider):
            return
        with self._writing(structural=False) as conn:
            if enabled:
                conn.execute(DDL_SKETCHES)
                conn.execute(DDL_SKETCH_TRIGGER)
                conn.execute(
                    "INSERT INTO schema_info (key, value) VALUES (?, '1')"
                    " ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                    (_PERSIST_KEY,),
                )
            else:
                conn.execute(f"DROP TRIGGER IF EXISTS {SKETCH_TABLE}_delete")
                conn.execute(f"DROP TABLE IF EXISTS {SKETCH_TABLE}")
                conn.execute("DELETE FROM schema_info WHERE key = ?", (_PERSIST_KEY,))

    def _sketch_candidates(self, where: str, params: list[object], tokens: Iterable[str]) -> list[int]:
        """Rowids (ascending) of retrievable rows whose sketch has every bit of *tokens*.

        The row list is re-read only when ``write_version`` changed (every
        content write bumps it); otherwise this is one bit test per row.
        TTL expiry is applied again when the candidates are fetched.
        """
        sketcher = self.__dict__.get("_token_sketcher")
        if sketcher is None:
            sketcher = self.__dict__["_token_sketcher"] = TokenSketcher.for_store(self.encryption.content_digest)
        known: dict[int, tuple[bytes, int]] = self.__dict__.setdefault("_token_sketches", {})
        scans = self.__dict__.setdefault("_sketch_scans", {})
        key = (where, tuple(params))
        version = self._write_version()
        hit = scans.get(key)
        if hit is not None and hit[0] == version:
            current = hit[1]
        else:
            current = self._sync_sketches(sketcher, known, where, params)
            if len(scans) >= 16:
                scans.clear()
            scans[key] = (version, current)
        wanted = sketcher.query(tokens)
        return [rowid for rowid in current if known[rowid][1] & wanted == wanted]

    def _sync_sketches(
        self, sketcher: TokenSketcher, known: dict[int, tuple[bytes, int]], where: str, params: list[object],
    ) -> list[int]:
        """Bring sketches of the matching rows up to date; return their rowids in order."""
        persisted = self.get_meta(_PERSIST_KEY) == "1"
        stored: dict[int, tuple[bytes, int]] = {}
        if persisted:
            rows = self._fetchall(
                f"SELECT f.rowid, {STAMP_SQL.replace('content', 'f.content')}, s.stamp, s.bits"
                f" FROM fragments f LEFT JOIN {SKETCH_TABLE} s ON s.rowid = f.rowid"
                f" WHERE {where}{_TTL_LIVE_SQL} ORDER BY f.rowid",
                params,
            )
            stored = {int(r[0]): (bytes(r[2]), from_blob(r[3])) for r in rows if r[2] is not None}
        else:
            rows = self._fetchall(
                f"SELECT rowid, {STAMP_SQL} FROM fragments WHERE {where}{_TTL_LIVE_SQL} ORDER BY rowid", params,
            )
        current = [(int(r[0]), bytes(r[1] or b"")) for r in rows]
        stale = []
        for rowid, stamp in current:
            if known.get(rowid, (None,))[0] == stamp:
                continue
            if stored.get(rowid, (None,))[0] == stamp:
                known[rowid] = stored[rowid]
            else:
                stale.append(rowid)
        if stale:
            fresh = self._sketch_rows(sketcher, stale)
            known.update(fresh)
            if persisted:
                with self._writing(structural=False) as conn:
                    conn.executemany(
                        f"INSERT OR REPLACE INTO {SKETCH_TABLE} (rowid, stamp, bits) VALUES (?, ?, ?)",
                        [(rowid, stamp, to_blob(bits)) for rowid, (stamp, bits) in fresh.items()],
                    )
        return [rowid for rowid, _ in current]

    def _sketch_rows(self, sketcher: TokenSketcher, rowids: list[int]) -> dict[int, tuple[bytes, int]]:
        """Decrypt and sketch the given rows: rowid -> (stamp, bits)."""
        out: dict[int, tuple[bytes, int]] = {}
        for chunk in _chunks(rowids, _SQL_CHUNK):  # type: ignore[arg-type]
            marks = ",".join("?" for _ in chunk)
            for rowid, content, stamp in self._fetchall(
                f"SELECT rowid, content, {STAMP_SQL} FROM fragments WHERE rowid IN ({marks})", chunk,
            ):
                try:
                    bits = sketcher.sketch(self.encryption.decrypt_str(_as_bytes(content)))
                except Exception:  # noqa: BLE001 - always a candidate; the read path isolates it
                    bits = _ALL_BITS
                out[int(rowid)] = (bytes(stamp or b""), bits)
        return out
