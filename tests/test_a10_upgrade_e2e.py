# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 Somtochukwu Owen Ikeanyi
"""End to end with the real releases: stores written by mark-sdk 0.2.0a8 / 0.2.0a9.

The old releases run in their own virtualenvs (``MARK_A8_PYTHON`` /
``MARK_A9_PYTHON``, or the default paths below); the tests are skipped when
those are missing. Each store is written by the old release, opened by this
build (upgraded, backed up), and the backup is then opened again by the old
release, which must find every memory and fact exactly as before.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from _helpers_upgrade import backups, counts, stamp

_VENVS = Path("/home/emsoft-analytics/mark-dev/retest/integ/venv")
PYTHONS = {4: os.environ.get("MARK_A8_PYTHON", str(_VENVS / "va8/bin/python")),
           5: os.environ.get("MARK_A9_PYTHON", str(_VENVS / "va9/bin/python"))}
TEXTS = ["Ana moved to Lisbon in 2024.", "上海仓库的库存周五盘点。", "Deploy finished 🚀 without errors.",
         "TODO: rotate the staging keys."]

# The same toy provider in both interpreters: XOR, prefixed so stored values are not UTF-8.
PROVIDER = '''
from mark.security import EncryptionProvider
class Xor(EncryptionProvider):
    def encrypt(self, data):
        return b"\\xff" + bytes(b ^ 0xA5 for b in data)
    def decrypt(self, data):
        data = bytes(data)
        if data[:1] != b"\\xff":
            raise ValueError("not encrypted")
        return bytes(b ^ 0xA5 for b in data[1:])
'''
WRITE = PROVIDER + '''
import json, sys
from datetime import datetime, timezone
from mark.memory.runtime import MarkRuntime
from mark.types.memory_document import MemoryDocument
db, encrypted, texts = sys.argv[1], sys.argv[2] == "1", json.loads(sys.argv[3])
rt = MarkRuntime.local(store_path=db, encryption=Xor() if encrypted else None)
for text in texts:
    rt.memory("a").store_sync(text)
for year, city in ((2024, "Lisbon"), (2025, "Porto")):
    rt.store.fact_set("a", "ana", "city", city, valid_from=datetime(year, 1, 1, tzinfo=timezone.utc))
rt.store.store_document(MemoryDocument(title="Runbook", content="Restart the API first.", agent_id="a"))
rt.shutdown()
'''
READ = PROVIDER + '''
import json, sys
from mark.memory.runtime import MarkRuntime
db, encrypted = sys.argv[1], sys.argv[2] == "1"
rt = MarkRuntime.local(store_path=db, encryption=Xor() if encrypted else None)
hits = rt.memory("a").retrieve_sync("上海仓库").fragments
history = [f["value"] for f in rt.store.fact_history("a", "ana", "city")]
print(json.dumps({"schema": rt.store.schema_version(), "top": hits[0].content if hits else None,
                  "history": history, "memories": len(rt.store._fetchall("SELECT id FROM fragments"))}))
rt.shutdown()
'''


def _old(version: int, script: str, *args: str) -> str:
    python = PYTHONS[version]
    if not Path(python).exists():
        pytest.skip(f"no mark-sdk virtualenv for schema {version} at {python}")
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}  # the old release, not this tree
    result = subprocess.run([python, "-c", script, *args], capture_output=True, text=True, env=env,
                            cwd=Path(python).parent, timeout=300)
    assert result.returncode == 0, result.stderr[-2000:]
    return result.stdout


def _xor():
    namespace: dict = {}
    exec(PROVIDER, namespace)  # noqa: S102 - the same toy provider the old release used
    return namespace["Xor"]()


def _write_old(version: int, db: Path, *, encrypted: bool = False) -> dict[str, int]:
    _old(version, WRITE, str(db), "1" if encrypted else "0", json.dumps(TEXTS))
    assert stamp(db) == version
    return counts(db)


def _read_old(version: int, db: Path, *, encrypted: bool = False) -> dict:
    copy = db.with_name(db.name + ".opened.db")  # keep the backup itself pristine
    shutil.copy(db, copy)
    return json.loads(_old(version, READ, str(copy), "1" if encrypted else "0").strip().splitlines()[-1])


@pytest.mark.parametrize("version", [4, 5])
def test_real_old_store_is_upgraded_and_its_backup_opens_in_the_old_release(tmp_path: Path, version: int) -> None:
    from mark.memory.runtime import MarkRuntime

    db = tmp_path / "memory.db"
    before = _write_old(version, db)
    assert before["fragments"] == len(TEXTS) and before["facts"] == 2 and before["documents"] == 1
    rt = MarkRuntime.local(store_path=db)
    assert rt.store.schema_version() == 6
    hits = rt.memory("a").retrieve_sync("上海仓库").fragments
    assert hits and hits[0].content == TEXTS[1]
    assert [f["value"] for f in rt.store.fact_history("a", "ana", "city")] == ["Lisbon", "Porto"]
    rt.shutdown()
    (backup,) = backups(tmp_path)
    assert f".v{version}-backup-" in backup.name and counts(backup) == before == counts(db)
    seen = _read_old(version, backup)
    assert seen == {"schema": version, "top": TEXTS[1], "history": ["Lisbon", "Porto"], "memories": len(TEXTS)}


def test_real_encrypted_a9_store_backup_stays_encrypted(tmp_path: Path) -> None:
    from mark.memory.runtime import MarkRuntime

    db = tmp_path / "memory.db"
    before = _write_old(5, db, encrypted=True)
    rt = MarkRuntime.local(store_path=db, encryption=_xor())
    assert rt.store.schema_version() == 6
    assert rt.memory("a").retrieve_sync("Lisbon").fragments
    rt.shutdown()
    (backup,) = backups(tmp_path)
    raw = backup.read_bytes()
    assert b"Lisbon" not in raw and "上海".encode() not in raw and b"staging keys" not in raw
    assert counts(backup) == before
    seen = _read_old(5, backup, encrypted=True)
    assert seen["schema"] == 5 and seen["history"] == ["Lisbon", "Porto"] and seen["memories"] == len(TEXTS)


def test_cli_round_trip_with_a_real_a9_store(tmp_path: Path, capsys) -> None:
    from mark.cli import main

    db = tmp_path / "memory.db"
    before = _write_old(5, db)
    assert main(["--db", str(db), "migrate", "--dry-run"]) == 0
    assert "-> PASS" in capsys.readouterr().out and stamp(db) == 5
    assert main(["--db", str(db), "migrate"]) == 0
    (backup,) = backups(tmp_path)
    assert stamp(db) == 6 and counts(db) == before
    assert main(["--db", str(db), "remember", "--no-server", "written after the upgrade"]) == 0
    assert main(["--db", str(db), "backups", "restore", str(backup), "--yes"]) == 0
    capsys.readouterr()
    assert stamp(db) == 5 and counts(db) == before
    seen = json.loads(_old(5, READ, str(db), "0").strip().splitlines()[-1])  # a9 uses the restored store
    assert seen["schema"] == 5 and seen["memories"] == len(TEXTS) and seen["top"] == TEXTS[1]
    (kept,) = tmp_path.glob("memory.db.pre-restore-*.db")
    assert counts(kept)["fragments"] == len(TEXTS) + 1  # the post-upgrade write is kept too
