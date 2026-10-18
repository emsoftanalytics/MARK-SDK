# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 MARK Contributors
"""MarkChat and ContextWatch accept ``mark.memory`` (SimpleMemory), not only MarkMemory.

Release re-test (dev major 1): ``MarkChat(mark.memory)`` crashed and
``ContextWatch(mark.memory)`` briefed nothing because SimpleMemory lacked
``retrieve_many_sync``. The HTTP server is a transport stub with a fixed reply.
"""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from mark import ContextWatch, Mark
from mark.context.loop import MarkChat


@pytest.fixture
def server():
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802 - http.server API
            self.rfile.read(int(self.headers["Content-Length"]))
            payload = json.dumps({"choices": [{"message": {"content": "Noted."}}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args) -> None:
            pass

    httpd = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_port}/v1"
    httpd.shutdown()
    httpd.server_close()


@pytest.fixture
def mark(tmp_path: Path):
    m = Mark.local(tmp_path / "simple-chat")
    yield m
    m.close()


def test_context_watch_briefs_from_simple_memory(mark) -> None:
    mark.memory.remember("The billing service runs on port 4444.")
    with ContextWatch(mark.memory, background=False) as watch:
        watch.observe("Which port does the billing service use?", role="user")
        assert "4444" in watch.briefing().text


def test_mark_chat_runs_over_simple_memory(server, mark) -> None:
    mark.memory.remember("The billing service runs on port 4444.")
    with MarkChat(mark.memory, base_url=server) as chat:
        assert chat.send("I'm allergic to penicillin.") == "Noted."
        chat.send("Which port does the billing service use?")
    contents = [f.content for f in mark.memory.list(limit=None)]
    assert "I'm allergic to penicillin." in contents
    assert any(f.attribute == "allergy" for f in mark.memory.facts.current())


def test_simple_memory_forwards_the_runtime_read_and_write_verbs(mark) -> None:
    memory = mark.memory
    assert memory.agent_id == "__mark__"
    fid = memory.store_sync("Release train leaves on Fridays.")
    assert memory.retrieve_sync("release train").fragments[0].id == fid
    results = memory.retrieve_many_sync(["release train", "fridays"])
    assert [r.fragments[0].id for r in results] == [fid, fid]


def test_documented_parameters_are_in_the_signatures(mark) -> None:
    import inspect

    recall = inspect.signature(mark.memory.recall).parameters
    remember = inspect.signature(mark.memory.remember).parameters
    assert {"k", "kind", "tags", "session_id", "since", "until", "order", "open", "blocks"} <= set(recall)
    assert {"kind", "tags", "session_id", "occurred_at", "source", "metadata", "extract_facts"} <= set(remember)
