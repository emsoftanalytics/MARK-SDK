# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 MARK Contributors
"""LLMFactExtractor against a local OpenAI-compatible stub server (no network).

Checks the request shape, tolerant reply parsing, that transport errors raise
(so the job queue records and retries them), and the end-to-end background
path: write -> queued job -> extract_pending -> facts.
"""
from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from mark.embeddings import HashEmbeddingProvider
from mark.intelligence.fact_extraction import FactExtractionService, LLMFactExtractor
from mark.intelligence.fact_extraction.llm import parse_fact_reply
from mark.memory.runtime import MarkRuntime


class _Stub:
    """Replies with queued message contents (or HTTP status codes) and records requests."""

    def __init__(self) -> None:
        self.replies: list[str | int] = []
        self.requests: list[dict] = []
        self.headers: list[dict] = []


@pytest.fixture()
def stub() -> Iterator[tuple[_Stub, str]]:
    state = _Stub()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802 - http.server API
            body = self.rfile.read(int(self.headers["Content-Length"]))
            state.requests.append({"path": self.path, **json.loads(body)})
            state.headers.append(dict(self.headers))
            reply = state.replies.pop(0) if state.replies else '{"facts": []}'
            if isinstance(reply, int):
                self.send_response(reply)
                self.end_headers()
                return
            payload = json.dumps({"choices": [{"message": {"role": "assistant", "content": reply}}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield state, f"http://127.0.0.1:{server.server_address[1]}/v1"
    finally:
        server.shutdown()
        server.server_close()


def test_request_shape_and_parsed_candidates(stub) -> None:
    state, url = stub
    state.replies.append('{"facts": [{"subject": "atlas-core", "attribute": "Port", "value": 4444,'
                         ' "confidence": 0.9}]}')
    extractor = LLMFactExtractor(model="tiny-model", base_url=url + "/", api_key="k")
    when = datetime(2026, 3, 30, tzinfo=timezone.utc)
    [fact] = extractor.extract("CHANGE: atlas-core moved from port 9239 to port 4444.", occurred_at=when)
    assert (fact.subject, fact.attribute, fact.value, fact.valid_from) == ("atlas-core", "port", "4444", when)
    assert fact.confidence == 0.9 and fact.rule == "llm"
    request = state.requests[0]
    assert request["path"] == "/v1/chat/completions" and request["model"] == "tiny-model"
    assert request["temperature"] == 0.0
    assert "2026-03-30" in request["messages"][1]["content"]
    assert state.headers[0]["Authorization"] == "Bearer k"
    assert extractor.background is True and extractor.name == "llm"


@pytest.mark.parametrize("reply", [
    'Sure! Here are the facts:\n```json\n{"facts": [{"subject": "web", "attribute": "region", "value": "eu-1"}]}\n```',
    '[{"entity": "web", "property": "region", "new_value": "eu-1"}]',
    'Facts: {"subject": "web", "attribute": "region", "value": "eu-1"} hope that helps',
])
def test_tolerant_reply_parsing(reply: str) -> None:
    [fact] = parse_fact_reply(reply)
    assert (fact.subject, fact.attribute, fact.value) == ("web", "region", "eu-1")


def test_invalid_items_are_dropped_one_by_one() -> None:
    reply = json.dumps({"facts": [
        {"subject": "it", "attribute": "port", "value": "1"},           # pronoun subject
        {"subject": "web", "attribute": "owner", "value": "[EMAIL]"},   # redaction placeholder
        {"subject": "web", "attribute": "port", "value": "2", "confidence": 0.2},  # below threshold
        {"subject": "web", "attribute": "", "value": "3"},              # no attribute
        "not an object",
        {"subject": "web", "attribute": "port", "value": "4", "valid_from": "2026-01-02"},
    ]})
    found = parse_fact_reply(reply, min_confidence=0.5)
    assert [(f.value, f.valid_from.date().isoformat()) for f in found] == [("4", "2026-01-02")]
    assert parse_fact_reply("no json here") == [] and parse_fact_reply("") == []


def test_transport_errors_raise(stub) -> None:
    state, url = stub
    state.replies.append(500)
    with pytest.raises(OSError):
        LLMFactExtractor(model="m", base_url=url, timeout=5).extract("web moved to port 1.")
    assert LLMFactExtractor(model="m", base_url=url).extract("   ") == []  # no call for empty text
    assert len(state.requests) == 1


def test_background_path_queues_then_extracts_with_retry(stub) -> None:
    state, url = stub
    state.replies.extend([500, '{"facts": [{"subject": "atlas-core", "attribute": "owner", "value": "Ana"}]}'])
    runtime = MarkRuntime.local(embedder=HashEmbeddingProvider(dim=32))
    runtime.fact_extraction = FactExtractionService(
        runtime.store, LLMFactExtractor(model="m", base_url=url, timeout=5), start_worker=False)
    memory = runtime.memory("a")
    memory.store_sync("[2026-03-12] OWNERSHIP: atlas-core is now owned by Ana.")
    assert state.requests == []  # the write did not call the model
    assert memory.facts.extract_pending() == 1
    [job] = runtime.store.fact_job_list()
    assert job["status"] == "pending" and "HTTPError" in job["error"]
    assert memory.facts.extract_pending() == 1
    owner = memory.facts.get("atlas-core", "owner")
    assert owner.value == "Ana" and owner.valid_from.date().isoformat() == "2026-03-12"
    assert owner.metadata["extracted_by"] == "llm"
    runtime.shutdown()
