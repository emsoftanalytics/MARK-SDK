# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 MARK Contributors
"""Recall says which memories are no longer current (dev re-test major 5).

MCP mark_retrieve showed a superseded memory without a label and no scores;
``mark recall --open`` kept memories that only state an earlier fact value.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from mark.cli import main
from mark.memory.result_states import states
from mark.memory.runtime import MarkRuntime


def _seed(memory) -> dict[str, str]:
    ids = {
        "old": memory.remember("[2026-01-05] I live in Porto.", extract_facts=True),
        "new": memory.remember("[2026-06-01] I moved to Lisbon.", extract_facts=True),
        "v1": memory.remember("DECISION: the deploy target is Fly.io"),
    }
    ids["v2"] = memory.supersede(ids["v1"], "DECISION: the deploy target is Hetzner")
    return ids


def test_states_label_superseded_and_fact_outdated(tmp_path: Path) -> None:
    runtime = MarkRuntime.local(store_path=tmp_path / "s.db")
    try:
        memory = runtime.memory("a")
        ids = _seed(memory)
        assert any(f.value == "Lisbon" for f in memory.facts.current())
        result = memory.recall("Where do I live?")
        labelled = states(result, runtime.store, "a")
        assert labelled.get(ids["old"]) == "outdated" and ids["new"] not in labelled
        deploy = memory.recall("deploy target")
        assert states(deploy).get(ids["v1"]) == "superseded"
        # A memory stating only an earlier value of a current fact.
        memory.facts.set("ana", "city", "Porto", valid_from="2026-01-01")
        memory.facts.set("ana", "city", "Lisbon", valid_from="2026-05-01")
        flat = memory.remember("Ana rents a flat in Porto.")
        result = memory.recall("ana city")
        assert flat in [f.id for f in result.fragments]
        assert states(result, runtime.store, "a").get(flat) == "outdated"
    finally:
        runtime.shutdown()


def test_cli_recall_labels_and_open_filters(tmp_path: Path, capsys) -> None:
    db = str(tmp_path / "c.db")
    for text in ("[2026-01-05] I live in Porto.", "[2026-06-01] I moved to Lisbon."):
        assert main(["--no-server", "--db", db, "remember", text, "--extract-facts"]) == 0
    capsys.readouterr()
    assert main(["--no-server", "--db", db, "recall", "Where do I live?", "--json"]) == 0
    rows = {m["content"]: m for m in json.loads(capsys.readouterr().out)["memories"]}
    assert rows["[2026-01-05] I live in Porto."]["state"] == "outdated"
    assert main(["--no-server", "--db", db, "recall", "Where do I live?"]) == 0
    assert "I live in Porto. (outdated)" in capsys.readouterr().out
    assert main(["--no-server", "--db", db, "recall", "Where do I live?", "--open"]) == 0
    out = capsys.readouterr().out
    assert "Porto" not in out.split("F.")[-1].split("\n", 1)[-1] and "Lisbon" in out


def test_cli_recall_rejects_empty_query_and_bad_k(tmp_path: Path, capsys) -> None:
    db = str(tmp_path / "k.db")
    assert main(["--no-server", "--db", db, "remember", "something"]) == 0
    capsys.readouterr()
    assert main(["--no-server", "--db", db, "recall", "  "]) == 1
    assert "query must not be empty" in capsys.readouterr().err
    assert main(["--no-server", "--db", db, "recall", "something", "-k", "0"]) == 1
    assert "-k must be at least 1" in capsys.readouterr().err


def test_mcp_retrieve_shows_states_and_scores(tmp_path: Path) -> None:
    pytest.importorskip("mcp", reason="mcp extra not installed")
    Client = getattr(__import__("mcp"), "Client", None)
    if Client is None:
        pytest.skip("mcp.Client needs mcp >= 2")
    from mark.adapters.mcp.launcher import build_server

    server, runtime = build_server(tmp_path / "m.db", agent_id="proj")
    try:
        ids = _seed(runtime.memory("proj"))

        async def call(query):
            async with Client(server) as client:
                r = await client.call_tool("mark_retrieve", {"query": query})
                return "\n".join(getattr(p, "text", "") for p in r.content)
        deploy = asyncio.run(call("deploy target"))
        live = asyncio.run(call("Where do I live?"))
    finally:
        runtime.shutdown()
    line = next(x for x in deploy.splitlines() if ids["v1"][:8] in x)
    assert "(superseded)" in line and "score " in line
    assert "(outdated)" in next(x for x in live.splitlines() if "Porto" in x and x.startswith("- "))


def test_mcp_fact_list_honours_subject(tmp_path: Path) -> None:
    pytest.importorskip("mcp", reason="mcp extra not installed")
    Client = getattr(__import__("mcp"), "Client", None)
    if Client is None:
        pytest.skip("mcp.Client needs mcp >= 2")
    from mark.adapters.mcp.launcher import build_server

    server, runtime = build_server(tmp_path / "f.db", agent_id="proj")
    try:
        facts = runtime.memory("proj").facts
        facts.set("spendlog", "currency", "EUR")
        facts.set("billing", "port", "4444")

        async def call(args):
            async with Client(server) as client:
                r = await client.call_tool("mark_fact", args)
                return "\n".join(getattr(p, "text", "") for p in r.content)
        one = asyncio.run(call({"action": "list", "subject": "spendlog"}))
        every = asyncio.run(call({"action": "list"}))
    finally:
        runtime.shutdown()
    assert "EUR" in one and "billing" not in one
    assert "billing" in every and "spendlog" in every
