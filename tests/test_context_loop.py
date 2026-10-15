"""MarkChat reference loop against an in-process fake OpenAI-compatible server.

The server is a transport test double: it records each request and answers
with a fixed acknowledgement, so the tests can check exactly what the model
would have been sent (window + budgeted briefing) and what MARK stored.
"""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from mark.context.budget import estimate_tokens
from mark.context.loop import MarkChat, MarkChatError, chat_completion
from mark.memory.runtime import MarkRuntime


class _Server:
    def __init__(self, status: int = 200) -> None:
        self.requests: list[dict] = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802 - http.server API
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.requests.append({"path": self.path, "body": body,
                                       "auth": self.headers.get("Authorization")})
                payload = ({"choices": [{"message": {"content": f"ack {len(outer.requests)}"}}]}
                           if status == 200 else {"error": "model not found"})
                data = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *args) -> None:
                pass

        self.httpd = HTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_port}/v1"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture
def server():
    srv = _Server()
    yield srv
    srv.close()


@pytest.fixture
def memory(tmp_path: Path):
    runtime = MarkRuntime.local(store_path=tmp_path / "loop.db")
    yield runtime.memory("chat")
    runtime.shutdown()


TURNS = ["Hi, I'm Ana and I keep bees in Lisbon.", "My hive is called Colmeia Azul.",
         "Which flowers suit spring?", "And summer?", "What is my hive called?"]


def test_window_plus_briefing_replaces_history(server, memory) -> None:
    with MarkChat(memory, base_url=server.url, model="m", system="Be brief.",
                  keep_last=2, budget_tokens=120, api_key="k") as chat:
        replies = [chat.send(text) for text in TURNS]
    assert replies == [f"ack {i}" for i in range(1, 6)]
    last = server.requests[-1]
    assert last["path"] == "/v1/chat/completions" and last["auth"] == "Bearer k"
    sent = last["body"]["messages"]
    assert last["body"]["model"] == "m"
    # system + last two messages (previous answer, current question) only
    assert [m["role"] for m in sent] == ["system", "assistant", "user"]
    assert sent[-1]["content"] == "What is my hive called?"
    assert not any("Colmeia Azul" in m["content"] for m in sent[1:])
    # ...but MARK put the dropped fact back, within the budget.
    system = sent[0]["content"]
    assert system.startswith("Be brief.") and "Colmeia Azul" in system
    briefing = system.split("Be brief.\n\n", 1)[1]
    assert estimate_tokens(briefing) <= 120 and chat.last_briefing.usage.used_tokens <= 120


def test_every_exchange_is_stored_before_it_leaves_the_window(server, memory) -> None:
    with MarkChat(memory, base_url=server.url, keep_last=2) as chat:
        for text in TURNS[:3]:
            chat.send(text)
        assert len(chat.window) == 2
    stored = [f.content for f in memory.retrieve_sync("Ana bees hive Lisbon Colmeia").fragments]
    assert any("User said: My hive is called Colmeia Azul." in s and "Assistant replied: ack 2" in s
               for s in stored)


def test_briefing_skips_what_the_window_already_shows(server, memory) -> None:
    with MarkChat(memory, base_url=server.url, keep_last=4, budget_tokens=400) as chat:
        chat.send("My hive is called Colmeia Azul.")
        prompt = chat.prepare("Tell me about my hive Colmeia Azul again.")
    assert sum("Colmeia Azul" in m["content"] for m in prompt) == 2   # window + question, not briefing
    assert "User said: My hive" not in prompt[0]["content"]


def test_prepare_commit_without_a_model(memory) -> None:
    with MarkChat(memory, base_url="http://127.0.0.1:9/v1", keep_last=1) as chat:
        for text in TURNS:
            chat.prepare(text)
            chat.commit(text, None)
        assert chat.window == [{"role": "user", "content": TURNS[-1]}]
        assert "Colmeia Azul" in chat.prepare("What is my hive called?")[0]["content"]


def test_server_errors_are_reported(memory) -> None:
    bad = _Server(status=404)
    try:
        with pytest.raises(MarkChatError, match="404"):
            chat_completion(bad.url, "missing", [{"role": "user", "content": "hi"}])
    finally:
        bad.close()
    with pytest.raises(MarkChatError, match="unreachable"):
        chat_completion("http://127.0.0.1:9/v1", "m", [], timeout=2)
    with pytest.raises(ValueError):
        MarkChat(memory, keep_last=0)
