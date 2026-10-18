# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 MARK Contributors
"""MarkChat and ContextWatch lifecycle, a10: raw user text, facts from user turns only,
progress/streaming hooks, cold first briefing, and a worker that ends with its store.

The HTTP server is a transport test double (fixed replies, SSE when asked to
stream); no model output is measured here.
"""
from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from mark import ContextWatch
from mark.context.loop import MarkChat
from mark.memory.runtime import MarkRuntime


class _Server:
    def __init__(self, reply: str = "I moved to Lisbon too, said the assistant.") -> None:
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802 - http.server API
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                if body.get("stream"):
                    chunks = [reply[:10], reply[10:]]
                    data = "".join(f"data: {json.dumps({'choices': [{'delta': {'content': c}}]})}\n\n"
                                   for c in chunks) + "data: [DONE]\n\n"
                    payload, kind = data.encode(), "text/event-stream"
                else:
                    payload = json.dumps({"choices": [{"message": {"content": reply}}]}).encode()
                    kind = "application/json"
                self.send_response(200)
                self.send_header("Content-Type", kind)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *args) -> None:
                pass

        self.httpd = HTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_port}/v1"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()


@pytest.fixture
def server():
    srv = _Server()
    yield srv
    srv.httpd.shutdown()
    srv.httpd.server_close()


@pytest.fixture
def runtime(tmp_path: Path):
    rt = MarkRuntime.local(store_path=tmp_path / "chat10.db")
    yield rt
    rt.shutdown()


def test_user_text_is_stored_raw_and_facts_come_from_it(server, runtime) -> None:
    memory = runtime.memory("chat")
    with MarkChat(memory, base_url=server.url) as chat:
        chat.send("I'm allergic to penicillin, by the way.")
        chat.send("What am I allergic to?")
    contents = [f.content for f in memory.list(limit=None)]
    assert "I'm allergic to penicillin, by the way." in contents
    assert not any(c.startswith("User said:") for c in contents)
    facts = {(f.subject, f.attribute): f.value for f in memory.facts.current()}
    assert facts.get(("user", "allergy")) == "penicillin"
    # Nothing is extracted from the assistant's words ("I moved to Lisbon").
    assert all("lisbon" not in str(v).lower() for v in facts.values())


def test_extraction_can_be_turned_off(server, runtime) -> None:
    memory = runtime.memory("chat")
    with MarkChat(memory, base_url=server.url, extract_facts=False) as chat:
        chat.send("I'm allergic to penicillin.")
    assert memory.facts.current() == []


def test_progress_and_streaming_hooks(server, runtime) -> None:
    stages: list[str] = []
    pieces: list[str] = []
    with MarkChat(runtime.memory("chat"), base_url=server.url,
                  on_status=stages.append, on_token=pieces.append) as chat:
        reply = chat.send("Hello there, I keep bees.")
    assert reply == "I moved to Lisbon too, said the assistant." and "".join(pieces) == reply
    assert len(pieces) == 2
    assert stages == ["briefing", "model", "storing", "done"]


def test_cold_first_briefing_waits_for_its_prefetch(runtime) -> None:
    memory = runtime.memory("ops")
    memory.store_sync("atlas-core listens on port 4444.")
    with ContextWatch(memory) as watch:   # background prefetch
        watch.observe("Which port does atlas-core use?", role="user")
        text = watch.briefing().text      # no wait= given
    assert "4444" in text


def test_worker_thread_ends_when_idle_or_closed(runtime) -> None:
    memory = runtime.memory("ops")
    watch = ContextWatch(memory)
    watch.observe("Anything about zephyr-9?", role="user")
    watch.flush(timeout=5)
    watch._executor.idle_seconds = 0.05
    watch.observe("And quokka-77?", role="user")
    watch.flush(timeout=5)
    deadline = time.monotonic() + 3
    while watch._executor.alive and time.monotonic() < deadline:
        time.sleep(0.02)
    assert not watch._executor.alive        # an unclosed watch leaves no thread behind
    watch.close()


def test_store_closed_stops_the_watch_quietly(tmp_path: Path, caplog) -> None:
    rt = MarkRuntime.local(store_path=tmp_path / "closing.db")
    watch = ContextWatch(rt.memory("ops"))
    rt.shutdown()
    with caplog.at_level("WARNING", logger="mark.context.watch"):
        watch.observe("Is atlas-core up?", role="user")
        watch.flush(timeout=5)
    assert "prefetch failed" not in caplog.text
    assert watch._executor is None          # stopped itself
    watch.close()
