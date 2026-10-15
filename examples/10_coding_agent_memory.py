"""Coding-agent memory: kinds, resolving TODOs, superseding notes, and the session brief.

Run:  python examples/10_coding_agent_memory.py
"""
from tempfile import TemporaryDirectory

from mark import Mark

with TemporaryDirectory() as folder, Mark.local(folder) as mark:
    repo = mark.runtime.memory("repo-agent")

    # Session 1: record what was decided, found and left to do, each with a kind.
    repo.remember("DECISION: the job queue uses SQLite in WAL mode.", kind="decision")
    bug = repo.remember("BUG: uploader.py::Uploader.send drops the last chunk on retry.", kind="bug")
    todo = repo.remember("TODO: add jittered backoff to uploader.py::Uploader.send", kind="todo")
    old = repo.remember("Run the tests with `pytest -q`.", kind="snippet")

    # Session 2: the fix lands, the TODO is done, the test command changed.
    repo.remember("FIX: Uploader.send flushes the final chunk before retrying.", kind="fix")
    repo.resolve(bug, note="fixed in 4f2c1e0")
    repo.resolve(todo)
    repo.supersede(old, "Run the tests with `uv run --extra dev pytest -q`.")  # history kept

    # Session 3 starts with one call instead of several searches.
    print(repo.brief().as_text(), "\n")
    print("Open TODOs:", [m.content for m in repo.list(kind="todo", open=True)])
    command = repo.recall("How do I run the tests?", k=1).top().content
    print("Test command:", command)

    assert repo.list(kind="todo", open=True) == []
    assert "uv run" in command
