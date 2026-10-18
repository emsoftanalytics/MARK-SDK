"""a10 re-test issue 1: closing a store never closes a connection mid-statement.

An unclosed ContextWatch kept querying on its worker thread while
``Mark.local``'s exit closed the store; ``close()`` closed connections other
threads were using and the process died with SIGSEGV. The store now gates
statements (close waits for running ones, refuses new ones) and watches are
closed before their store.
"""
from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from mark.errors import StoreClosedError
from mark.store.sqlite import LocalMemoryStore

SRC = str(Path(__file__).resolve().parents[1] / "src")


def _hold_read(store: LocalMemoryStore, inside: threading.Event, release: threading.Event, out: list) -> None:
    with store._reading() as conn:
        inside.set()
        release.wait(5)
        out.append(conn.execute("SELECT COUNT(*) FROM fragments").fetchone()[0])


def test_close_waits_for_a_running_statement_and_refuses_new_ones(tmp_path) -> None:
    store = LocalMemoryStore(tmp_path / "m.db")
    inside, release, out = threading.Event(), threading.Event(), []
    reader = threading.Thread(target=_hold_read, args=(store, inside, release, out))
    reader.start()
    assert inside.wait(5)
    closer = threading.Thread(target=store.close)
    closer.start()
    time.sleep(0.2)
    assert closer.is_alive()                  # waiting for the reader, not closing under it
    with pytest.raises(StoreClosedError):     # but no new statement is admitted
        store._fetchone("SELECT 1")
    release.set()
    reader.join(5)
    closer.join(5)
    assert out == [0] and not closer.is_alive()
    with pytest.raises(StoreClosedError):
        store._fetchone("SELECT 1")


def test_a_statement_outliving_the_close_timeout_keeps_its_connection(tmp_path) -> None:
    store = LocalMemoryStore(tmp_path / "m.db")
    inside, release, out = threading.Event(), threading.Event(), []
    reader = threading.Thread(target=_hold_read, args=(store, inside, release, out))
    reader.start()
    assert inside.wait(5)
    started = time.monotonic()
    store.close(timeout=0.1)                  # gives up waiting (bounded) ...
    assert time.monotonic() - started < 3
    release.set()
    reader.join(5)
    assert out == [0]                         # ... yet the reader's connection still worked
    assert not store._statement_gate()._orphans   # and was closed when it finished


def test_close_hooks_run_first_newest_first_and_can_unregister(tmp_path) -> None:
    store = LocalMemoryStore(tmp_path / "m.db")
    calls: list[str] = []
    store.add_close_hook(lambda: calls.append("a"))
    undo = store.add_close_hook(lambda: calls.append("b"))
    store.add_close_hook(lambda: calls.append(str(store._fetchone("SELECT 1")[0])))  # store still usable
    store.add_close_hook(lambda: 1 / 0)       # a failing hook does not keep the store open
    undo()
    store.close()
    store.close()
    assert calls == ["1", "a"]
    store.add_close_hook(lambda: calls.append("late"))   # after close: runs at once
    assert calls[-1] == "late"


def test_memory_store_close_waits_too() -> None:
    store = LocalMemoryStore(":memory:")
    store.close()
    with pytest.raises(StoreClosedError):
        store._fetchall("SELECT 1")


def test_an_unclosed_watch_is_closed_with_its_store(tmp_path) -> None:
    from mark import Mark
    from mark.context.watch import ContextWatch

    with Mark.local(tmp_path) as mark:
        memory = mark.runtime.memory("x")
        memory.store_many_sync([f"svc{i} port is {8000 + i}" for i in range(20)])
        watch = ContextWatch.for_preset(memory, "small")
        watch.observe("what is svc2's port and the Atlas-Core rollout?")
    assert watch._closed
    assert not any(t.name == "mark-watch" and t.is_alive() for t in threading.enumerate())


def test_unreferenced_watches_do_not_pin_the_store_hooks(tmp_path) -> None:
    import gc

    from mark import Mark
    from mark.context.watch import ContextWatch

    with Mark.local(tmp_path) as mark:
        memory = mark.runtime.memory("x")
        for _ in range(5):
            ContextWatch.for_preset(memory, "small", background=False)
        gc.collect()
        assert not mark.runtime.store.__dict__.get("_close_hooks")


_LOOP = """
import faulthandler, sys, tempfile
faulthandler.enable()
from mark import Mark
from mark.context.watch import ContextWatch
d = tempfile.mkdtemp(dir=sys.argv[1])
with Mark.local(d) as mark:
    mark.runtime.memory("x").store_many_sync([f"svc{i} port is {8000+i}; owner team{i}" for i in range(60)])
for cycle in range(int(sys.argv[2])):
    with Mark.local(d) as mark:
        m = mark.runtime.memory("x")
        m.facts.set(f"svc{cycle%50}", "owner", f"alice{cycle}")
        w = ContextWatch.for_preset(m, "small")   # never closed
        w.observe(f"what's svc{cycle%50} port? also svc{(cycle+1)%50}, Atlas-Core and Kestrel?")
        if cycle % 2: w.briefing(wait=0.0)
print("done", flush=True)
"""


def test_unclosed_watches_across_store_closes_never_crash(tmp_path) -> None:
    env = {**os.environ, "PYTHONPATH": SRC, "MARK_EMBEDDER": "hash"}
    proc = subprocess.run(
        [sys.executable, "-c", _LOOP, str(tmp_path), "12"],
        env=env, capture_output=True, text=True, timeout=300,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "done" in proc.stdout
    assert "Traceback" not in proc.stderr and "prefetch failed" not in proc.stderr
