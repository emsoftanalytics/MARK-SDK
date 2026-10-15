# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 MARK Contributors
"""MCP ``mark_briefing``: push context per session, isolated, bounded, never blocking.

Runs the real MCP server in-process against the local backend (no network).
"""
from __future__ import annotations

import asyncio
import threading

import pytest

pytest.importorskip("mcp", reason="mcp extra not installed")

from mark.adapters.backend import LocalMarkBackend
from mark.adapters.mcp import create_mark_mcp_server, create_mark_mcp_server_from_local
from mark.runtime import Mark

AGENT = "__mark__"


@pytest.fixture()
def mark(tmp_path):
    with Mark.local(project_path=str(tmp_path)) as m:
        ops = m.runtime.memory(AGENT)
        ops.store_sync("[2026-03-12] TICKET HAL-4433: checkout returns 502 errors; affects checkout-api.")
        ops.store_sync("[2026-03-14] TICKET HAL-5150: payroll export fails for payroll-svc.")
        ops.facts.set("checkout-api", "port", "8080", valid_from="2026-03-01")
        ops.facts.set("checkout-api", "port", "9090", valid_from="2026-03-20")
        m.runtime.memory("other").store_sync("[2026-03-01] SECRET-77: only for the other namespace.")
        yield m


def _server(mark, **options):
    return create_mark_mcp_server_from_local(mark, agent_id=AGENT, briefing=True, **options)


def _text(result) -> str:
    content = getattr(result, "content", result)
    if isinstance(content, (list, tuple)):
        return "\n".join(getattr(part, "text", str(part)) for part in content)
    return str(content)


def _call(server, args):
    return _text(asyncio.run(server.call_tool("mark_briefing", {"wait": 2.0, **args})))


def _read(server, uri):
    async def _run():
        return list(await server.read_resource(uri))
    return "\n".join(getattr(part, "content", "") for part in asyncio.run(_run()))


def test_briefing_is_opt_in(mark) -> None:
    async def names(server):
        return {t.name for t in await server.list_tools()}
    plain = create_mark_mcp_server_from_local(mark, agent_id=AGENT)
    assert "mark_briefing" not in asyncio.run(names(plain))
    assert "mark_briefing" in asyncio.run(names(_server(mark)))


def test_briefing_end_to_end_with_links_and_corrections(mark) -> None:
    server = _server(mark)
    text = _call(server, {"text": "Customers see 502s. Please look into HAL-4433.", "session_id": "s1"})
    assert text.startswith("<mark_briefing>")
    assert "TICKET HAL-4433" in text and "port: 9090" in text
    text = _call(server, {"text": "I'll restart checkout-api on port 8080.", "role": "assistant",
                          "session_id": "s1"})
    assert "checkout-api port is 9090 now" in text and "8080 is outdated" in text


def test_briefing_accepts_message_lists(mark) -> None:
    server = _server(mark)
    messages = [
        {"role": "system", "content": "You are on call."},
        {"role": "user", "content": [{"type": "text", "text": "What about HAL-5150?"}]},
    ]
    text = _call(server, {"messages": messages, "session_id": "m"})
    assert "\nHAL-5150:" in text and "\nHAL-4433:" not in text   # subject headings


def test_sessions_are_isolated(mark) -> None:
    server = _server(mark)
    one = _call(server, {"text": "Look into HAL-4433.", "session_id": "one"})
    two = _call(server, {"text": "Look into HAL-5150.", "session_id": "two"})
    # Subject headings show what each session is about ("Related to the current
    # request" may list other records: the store is tiny).
    assert "\nHAL-4433:" in one and "\nHAL-5150:" not in one
    assert "\nHAL-5150:" in two and "\nHAL-4433:" not in two
    # Session one is unchanged by session two's traffic.
    assert "\nHAL-5150:" not in _call(server, {"session_id": "one"})
    # Same session id in another agent namespace is a different session.
    server = create_mark_mcp_server_from_local(mark, agent_id=AGENT, briefing=True)
    other = _call(server, {"text": "SECRET-77?", "session_id": "one", "agent_id": "other"})
    assert "only for the other namespace" in other
    assert "SECRET-77" not in _call(server, {"text": "SECRET-77?", "session_id": "one"}).split("Not in memory")[0]


def test_lru_eviction_closes_watches(mark) -> None:
    server = _server(mark, briefing_sessions=2)
    sessions = server.mark_briefing_sessions
    _call(server, {"text": "Look into HAL-4433.", "session_id": "a"})
    first = sessions.get(AGENT, "a", create=False)
    _call(server, {"text": "Look into HAL-5150.", "session_id": "b"})
    _call(server, {"text": "hello", "session_id": "c"})
    assert len(sessions) == 2 and (AGENT, "a") not in sessions.keys()
    assert first._executor is None                       # evicted watch was closed
    # A re-created session starts empty: nothing leaks from the evicted state.
    assert "\nHAL-4433:" not in _call(server, {"session_id": "a"})
    sessions.close_all()
    assert len(sessions) == 0


def test_allowed_agent_ids_are_enforced(mark) -> None:
    server = create_mark_mcp_server(
        LocalMarkBackend(mark, default_agent_id=AGENT), default_agent_id=AGENT,
        allowed_agent_ids=[], briefing=True,
    )
    denied = _call(server, {"text": "SECRET-77?", "agent_id": "other"})
    assert denied.startswith("MARK briefing failed") and "not allowed" in denied
    assert len(server.mark_briefing_sessions) == 0       # no state created for denied calls


def test_resource_reads_without_creating_sessions(mark) -> None:
    server = _server(mark)
    _call(server, {"text": "Look into HAL-4433.", "session_id": "poll"})
    assert "TICKET HAL-4433" in _read(server, "mark://briefing/poll")
    assert _read(server, "mark://briefing/unknown") == ""
    assert (AGENT, "unknown") not in server.mark_briefing_sessions.keys()


def test_bad_input_and_unsupported_backends_never_raise(mark) -> None:
    server = _server(mark)
    assert "role must be" in _call(server, {"text": "x", "role": "system"})
    assert _call(server, {"text": "", "session_id": "empty"}).startswith("<mark_briefing>")

    class RemoteOnly:  # a backend without memory_for cannot anticipate
        async def retrieve(self, *a, **k): ...

    remote = create_mark_mcp_server(RemoteOnly(), briefing=True)
    assert "unavailable" in _call(remote, {"text": "HAL-4433"})


def test_briefing_does_not_block_the_event_loop(mark) -> None:
    server = _server(mark)
    loop_threads: set[int] = set()

    async def drive():
        loop_threads.add(threading.get_ident())
        ticks = 0

        async def ticker():
            nonlocal ticks
            while True:
                ticks += 1
                await asyncio.sleep(0.001)

        task = asyncio.create_task(ticker())
        await asyncio.gather(*(server.call_tool("mark_briefing", {"text": f"Check HAL-4433 {i}",
                                                                  "session_id": str(i), "wait": 5.0})
                               for i in range(3)))
        task.cancel()
        return ticks

    assert asyncio.run(drive()) > 5                      # the loop kept ticking during the calls
