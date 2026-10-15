"""Time-aware recall (B5: since/until, dates named in queries) and multi-hop chains (B7)."""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import pytest

from mark.embeddings import HashEmbeddingProvider
from mark.intelligence import RetrievalPolicy
from mark.intelligence.retrieval_policy import SessionFilter
from mark.memory.runtime import MarkRuntime

_FILLER = [f"The {w} team reviewed the {o} on Monday (ref {i})." for i, (w, o) in enumerate(
    [("marketing", "budget"), ("QA", "release notes"), ("design", "logo"), ("HR", "holiday calendar"),
     ("mobile", "rate limits"), ("support", "postmortem"), ("data", "dashboard"), ("ops", "parking rota")] * 3)]


@pytest.fixture()
def runtime():
    runtime = MarkRuntime.local(embedder=HashEmbeddingProvider())
    yield runtime
    runtime.shutdown()


@pytest.fixture()
def mem(runtime):
    mem = runtime.memory("t")
    mem.store_many_sync(_FILLER)
    return mem


def _ids(result) -> list[str]:
    return [f.content for f in result.fragments]


def test_since_until_scope_by_event_time(runtime, mem) -> None:
    mem.store_sync("Deploy of checkout went fine.", occurred_at="2026-10-01")
    mem.store_sync("Deploy of checkout failed with 502s.", occurred_at="2026-10-09")
    pipeline = runtime.pipeline
    later = pipeline.retrieve("checkout deploy", "t", since="2026-10-05")
    assert [c for c in _ids(later) if "checkout" in c] == ["Deploy of checkout failed with 502s."]
    earlier = pipeline.retrieve("checkout deploy", "t", until=date(2026, 10, 5))
    assert [c for c in _ids(earlier) if "checkout" in c] == ["Deploy of checkout went fine."]
    both = pipeline.retrieve("checkout deploy", "t", session_filter=SessionFilter(since="2026-10-02", until="2026-10-08"))
    assert not [c for c in _ids(both) if "checkout" in c]


def test_since_accepts_natural_expressions(runtime, mem) -> None:
    yesterday = (datetime.now(timezone.utc) - timedelta(days=1)).date().isoformat()
    mem.store_sync("Standup moved to 11:00.", occurred_at=yesterday)
    mem.store_sync("Standup was at 9:30.", occurred_at="2020-01-01")
    contents = _ids(runtime.pipeline.retrieve("standup", "t", since="last week"))
    assert "Standup moved to 11:00." in contents and "Standup was at 9:30." not in contents


def test_invalid_since_is_a_value_error(runtime, mem) -> None:
    with pytest.raises(ValueError):
        runtime.pipeline.retrieve("standup", "t", since="whenever")


def test_query_naming_a_date_finds_memories_from_that_day(mem) -> None:
    last_tuesday = date.today() - timedelta(days=(date.today().weekday() - 1 - 1) % 7 + 1)
    mem.store_sync("We talked about switching the CI from Jenkins to GitHub Actions.",
                   occurred_at=last_tuesday.isoformat())
    assert "Jenkins" in mem.retrieve_sync("What did we discuss last Tuesday?").top().content
    assert "Jenkins" in mem.retrieve_sync(
        f"What did we talk about on {last_tuesday.day} {last_tuesday.strftime('%B')}?").top().content


def test_dates_written_in_memories_match_other_formats(mem) -> None:
    mem.store_sync("Invoice INV-5531 is due 05/03/2026 (UK format).")
    mem.store_sync("The board review is on the 3rd of May 2026 at the Lagos office.")
    mem.store_sync("Ship the beta next Tuesday, said Ana on 2026-10-08.")
    assert "INV-5531" in mem.retrieve_sync("Which invoices are due on 5 March 2026?").top().content
    assert "board review" in mem.retrieve_sync("What is on 2026-05-03?").top().content
    assert "beta" in mem.retrieve_sync("What ships on 13 October 2026?").top().content


def test_multi_hop_chain_lands_in_top_k(mem) -> None:
    mem.store_sync("Priya manages Tomasz.")
    mem.store_sync("Tomasz owns the ledger-sync service.")
    mem.store_sync("The ledger-sync service depends on the kafka-east cluster.")
    mem.store_sync("Kemi is Priya's manager and reports to the CTO, Haruto.")
    top5 = _ids(mem.retrieve_sync("Who manages the owner of ledger-sync?", RetrievalPolicy.BALANCED))[:5]
    assert {"Priya manages Tomasz.", "Tomasz owns the ledger-sync service."} <= set(top5)
    deep = _ids(mem.retrieve_sync("If kafka-east goes down, whose team lead should be told?", RetrievalPolicy.DEEP))[:5]
    assert {"Priya manages Tomasz.", "Tomasz owns the ledger-sync service."} <= set(deep)
    fast = _ids(mem.retrieve_sync("If kafka-east goes down, whose team lead should be told?", RetrievalPolicy.FAST))
    assert "Priya manages Tomasz." not in fast  # FAST stays single-shot


def test_hops_never_outrank_their_source(mem) -> None:
    mem.store_sync("Tomasz owns the ledger-sync service.")
    mem.store_sync("Priya manages Tomasz.")
    result = mem.retrieve_sync("Who owns ledger-sync?", RetrievalPolicy.DEEP)
    assert result.top().content == "Tomasz owns the ledger-sync service."
    assert result.scores == sorted(result.scores, reverse=True)
