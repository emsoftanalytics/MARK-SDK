"""split_history: the sliding window used when MARK replaces the conversation."""
from __future__ import annotations

import pytest

from mark.context.history import message_key, message_role, split_history, turn_text


def _call(cid: str) -> dict:
    return {"role": "assistant", "content": "",
            "tool_calls": [{"id": cid, "type": "function", "function": {"name": "lookup", "arguments": "{}"}}]}


def _result(cid: str) -> dict:
    return {"role": "tool", "tool_call_id": cid, "content": f"result {cid}"}


def _valid(messages: list) -> bool:
    """Every tool result follows its calling assistant message; every call has its results."""
    open_calls: set[str] = set()
    for message in messages:
        if message.get("tool_calls"):
            open_calls = {c["id"] for c in message["tool_calls"]}
        elif message["role"] == "tool":
            if message["tool_call_id"] not in open_calls:
                return False
            open_calls.discard(message["tool_call_id"])
        elif open_calls and message["role"] != "system":
            return False
    return True


def test_short_history_is_kept_whole() -> None:
    messages = [{"role": "system", "content": "S"}, {"role": "user", "content": "hi"}]
    split = split_history(messages, 6)
    assert split.kept == messages and split.dropped == []


def test_keeps_system_and_last_n() -> None:
    messages = [{"role": "system", "content": "S"}] + [
        {"role": "user" if i % 2 == 0 else "assistant", "content": f"m{i}"} for i in range(8)]
    split = split_history(messages, 3)
    assert [m["content"] for m in split.kept] == ["S", "m5", "m6", "m7"]
    assert [m["content"] for m in split.dropped] == ["m0", "m1", "m2", "m3", "m4"]


def test_window_never_starts_inside_a_tool_exchange() -> None:
    messages = [{"role": "user", "content": "q1"}, {"role": "assistant", "content": "a1"},
                {"role": "user", "content": "q2"}, _call("c1"), _result("c1"), _call("c2"),
                _result("c2"), {"role": "assistant", "content": "done"}]
    for keep in range(1, len(messages) + 1):
        split = split_history(messages, keep)
        assert _valid(split.kept), keep
        assert split.kept[0]["content"] == "q2" or keep >= 7   # current question kept
        assert len(split.kept) + len(split.dropped) == len(messages)


def test_parallel_tool_results_stay_together() -> None:
    call = {"role": "assistant", "content": "", "tool_calls": [
        {"id": "a", "function": {"name": "x", "arguments": "{}"}},
        {"id": "b", "function": {"name": "y", "arguments": "{}"}}]}
    messages = [{"role": "user", "content": "q"}, call, _result("a"), _result("b")]
    split = split_history(messages, 1)
    assert split.kept == messages   # extended back to the call, user turn kept
    assert _valid(split.kept)


def test_orphaned_tool_results_are_dropped() -> None:
    question = {"role": "user", "content": "q"}
    split = split_history([_result("lost"), _result("lost2"), question], 2)
    assert split.kept == [question] and len(split.dropped) == 2


def test_current_user_turn_is_kept_ahead_of_a_long_tool_loop() -> None:
    messages = [{"role": "user", "content": "question"}]
    for i in range(5):
        messages += [_call(f"c{i}"), _result(f"c{i}")]
    split = split_history(messages, 2)
    assert [m.get("content") for m in split.kept][:1] == ["question"]
    assert split.kept[1:] == messages[-2:] and _valid(split.kept)


def test_langchain_messages() -> None:
    pytest.importorskip("langchain_core")
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

    messages = [SystemMessage("S"), HumanMessage("old"), AIMessage("old answer"), HumanMessage("q"),
                AIMessage("", tool_calls=[{"name": "lookup", "args": {}, "id": "t1"}]),
                ToolMessage("r", tool_call_id="t1")]
    split = split_history(messages, 1)
    assert [m.type for m in split.kept] == ["system", "human", "ai", "tool"]
    assert [m.content for m in split.dropped] == ["old", "old answer"]
    assert message_role(messages[1]) == "user" and message_role(messages[4]) == "assistant"
    assert turn_text(messages[4]) == ""      # tool-call-only: nothing of its own to store
    assert turn_text(AIMessage("Checking.", tool_calls=[{"name": "lookup", "args": {}, "id": "t2"}])) \
        == "Assistant replied: Checking."
    failed = ToolMessage("bad args", tool_call_id="t1", name="lookup", status="error")
    assert turn_text(failed) == "Tool [lookup] failed: bad args"
    # A tool result keeps its key when LangGraph assigns an id on entering the state.
    assert message_key(ToolMessage("r", tool_call_id="t1")) == message_key(ToolMessage("r", tool_call_id="t1", id="x"))


def test_turn_text_and_keys() -> None:
    assert turn_text({"role": "user", "content": "My hive is Colmeia Azul"}) == "User said: My hive is Colmeia Azul"
    assert turn_text({"role": "tool", "name": "lookup", "content": "x"}) == "Tool [lookup] returned: x"
    assert turn_text({"role": "user", "content": ""}) == ""
    a, b = {"role": "user", "content": "x"}, {"role": "user", "content": "y"}
    assert message_key(a) == message_key(dict(a)) != message_key(b)
    with pytest.raises(ValueError):
        split_history([], 0)
