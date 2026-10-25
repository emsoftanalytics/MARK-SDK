# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 Somtochukwu Owen Ikeanyi
"""Upgrade benchmark: backup + verify + migrate time for a store at schema 5.

Takes a store built by ``bench.py --keep --dir DIR`` (or any MARK store),
copies it, turns the copy into a schema-5 (0.2.0a9) store the way the a9
layout differs (no multilingual FTS table, TTL index or vector snapshots),
then opens it with this build and reports the upgrade's phases, wall and
CPU time, load average, and for comparison a plain file copy and
``VACUUM INTO`` of the same store::

    python benchmarks/scale/upgrade_bench.py /data/mark-1m/.mark/memory.db

The copies are written next to the store and removed afterwards (``--keep``
keeps them). Page-cache state is not controlled: numbers are warm-cache.
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import shutil
import sqlite3
import sys
import time
from pathlib import Path


def _downgrade(db: Path) -> None:
    conn = sqlite3.connect(db)
    for sql in ("DROP TRIGGER IF EXISTS fragments_fts_ml_delete", "DROP TABLE IF EXISTS fragments_fts_ml",
                "DROP INDEX IF EXISTS idx_frag_ttl", "DROP TABLE IF EXISTS vector_snapshot_meta",
                "DROP TABLE IF EXISTS vector_snapshots", "DROP TABLE IF EXISTS vector_dirty",
                "DELETE FROM schema_info WHERE key IN ('fts_ml_backfilled', 'last_upgrade', 'last_upgrade_backup')",
                "UPDATE schema_info SET value = '5' WHERE key = 'schema_version'"):
        conn.execute(sql)
    conn.commit()
    conn.execute("VACUUM")  # a compact file, like a real a9 store
    conn.close()


def _cpu() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return usage.ru_utime + usage.ru_stime


def _timed(func) -> tuple[float, float]:
    wall, cpu = time.perf_counter(), _cpu()
    func()
    return time.perf_counter() - wall, _cpu() - cpu


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("db")
    parser.add_argument("--keep", action="store_true")
    parser.add_argument("--no-compare", action="store_true", help="skip the file-copy / VACUUM INTO runs")
    args = parser.parse_args(argv)
    from mark.store.sqlite import LocalMemoryStore

    source = Path(args.db).resolve()
    work = source.with_name("upgrade-bench.db")
    for leftover in source.parent.glob("upgrade-bench.db*"):
        leftover.unlink()
    shutil.copyfile(source, work)
    _downgrade(work)
    size = work.stat().st_size
    report: dict = {"db": str(source), "bytes": size, "cpus": os.cpu_count()}
    conn = sqlite3.connect(work)
    report["memories"] = conn.execute("SELECT count(*) FROM fragments").fetchone()[0]
    conn.close()
    if not args.no_compare:
        copy = work.with_name("upgrade-bench.copy.db")
        report["file_copy_s"] = round(_timed(lambda: shutil.copyfile(work, copy))[0], 3)
        copy.unlink()
        vac = work.with_name("upgrade-bench.vacuum.db")

        def vacuum_into() -> None:
            c = sqlite3.connect(work)
            c.execute("VACUUM INTO ?", (str(vac),))
            c.close()
        report["vacuum_into_s"] = round(_timed(vacuum_into)[0], 3)
        vac.unlink()
    report["load_before"] = os.getloadavg()
    store = None

    def upgrade() -> None:
        nonlocal store
        store = LocalMemoryStore(work)
    wall, cpu = _timed(upgrade)
    record = dict(store.last_upgrade or {})
    store.close()
    report.update(upgrade_wall_s=round(wall, 3), upgrade_cpu_s=round(cpu, 3), load_after=os.getloadavg(),
                  backup_s=record.get("backup_seconds"), verify_s=record.get("verify_seconds"),
                  migrate_s=record.get("migrate_seconds"), check=record.get("check"),
                  backup_bytes=record.get("backup_bytes"),
                  backup_mb_per_s=round(record["backup_bytes"] / record["backup_seconds"] / 1e6, 1)
                  if record.get("backup_seconds") else None, message=record.get("message"))
    if not args.keep:
        for item in source.parent.glob("upgrade-bench.db*"):
            item.unlink()
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
