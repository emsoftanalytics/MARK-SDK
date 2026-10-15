"""Command execution against one database, in-process or inside ``mark serve``.

:class:`_Engine` is the only place the CLI loads MARK; the import happens in
its constructor so merely importing this module stays standard-library only.
Results are converted to plain JSON-able dicts so the same payload works
in-process and over the loopback server.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from mark._cli.engine_ops import EngineOpsMixin, memory_json


def protocol_fingerprint() -> str:
    """Identity of the command interface: package version + every command's parameters.

    Client and server compare it during the handshake, so a client from a
    different build never sends arguments the server cannot accept (it runs
    in-process instead). Computed from signatures only — MARK isn't loaded.
    """
    import hashlib
    import inspect

    from mark._version import __version__

    parts = [__version__]
    for name in sorted(dir(_Engine)):
        if name.startswith("cmd_"):
            parts.append(f"{name}{inspect.signature(getattr(_Engine, name))}")
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()[:16]


class _Engine(EngineOpsMixin):
    """Loads MARK lazily and executes commands against one database.

    Read/write basics live here; lifecycle and data-management commands
    (forget, resolve, supersede, list, export, import, brief, agents) come
    from :class:`mark._cli.engine_ops.EngineOpsMixin`.
    """

    def __init__(self, db: Path) -> None:
        from mark._private_fs import ensure_private_dir
        from mark.memory.runtime import MarkRuntime

        ensure_private_dir(db.parent)  # 0700 when created here
        self.runtime = MarkRuntime.local(store_path=db)

    def close(self) -> None:
        self.runtime.shutdown()

    def run(self, cmd: str, args: dict[str, Any]) -> dict[str, Any]:
        """Dispatch *cmd* to its ``cmd_*`` method; only those are reachable remotely."""
        handler = getattr(self, f"cmd_{cmd.replace('-', '_')}", None)
        if handler is None:
            raise ValueError(f"unknown command {cmd!r}")
        return handler(**args)

    def _memory(self, agent: str):
        return self.runtime.memory(agent)

    def cmd_ingest(self, agent: str, lines: list[str], session: str | None = None,
                   tags: list[str] | None = None, extract_facts: bool = False,
                   kind: str | None = None) -> dict[str, Any]:
        import hashlib

        from mark.kinds import kind_tag, normalize_kind

        texts = [line.strip() for line in lines if line.strip()]
        kind = normalize_kind(kind)
        all_tags = [*(tags or []), *([kind_tag(kind)] if kind else [])]
        # Idempotent: a retried ingest (e.g. after the server was killed mid-request) maps the
        # same lines to the same ids and never duplicates them; each call is one transaction.
        key = "ingest:" + hashlib.sha256("\n".join(texts).encode("utf-8", "surrogatepass")).hexdigest()
        # extract_facts=True: built-in pattern packs, each line's "[YYYY-MM-DD]" stamp as valid_from.
        report = self._memory(agent).store_many_sync(texts, session_id=session, tags=all_tags or None,
                                                     extract_facts=True if extract_facts else None,
                                                     idempotency_key=key)
        result: dict[str, Any] = {"stored": report.stored}
        if report.skipped:
            result["skipped"] = len(report.skipped)
        if extract_facts:
            # Facts this call extracted (not the store-wide total).
            ids = [fid for fid in report if fid]
            result["extracted"] = self.runtime.store.facts_from_fragments(agent, ids)
            result["facts"] = len(self.runtime.store.fact_current(agent))
        return result

    def cmd_store(self, agent: str, text: str, session: str | None = None,
                  occurred_at: str | None = None, importance: float | None = None,
                  kind: str | None = None, tags: list[str] | None = None,
                  extract_facts: bool = False) -> dict[str, Any]:
        from mark.kinds import DEFAULT_IMPORTANCE

        memory = self._memory(agent)
        fid = memory.remember(
            text, session_id=session, occurred_at=occurred_at, kind=kind, tags=tags,
            importance=DEFAULT_IMPORTANCE if importance is None else importance,
            extract_facts=True if extract_facts else None,
        )
        result: dict[str, Any] = {"id": fid}
        if extract_facts:
            result["extracted"] = self.runtime.store.facts_from_fragments(agent, [fid])
        return result

    cmd_remember = cmd_store

    def cmd_recall(self, agent: str, query: str, k: int = 10, order: str = "relevance",
                   session: str | None = None, since: str | None = None, until: str | None = None,
                   tags: list[str] | None = None, kind: str | None = None) -> dict[str, Any]:
        if not any((session, since, until, tags, kind)):
            return self.cmd_recall_many(agent, [query], k, order)["results"][0]
        # Filtered recall goes through MarkMemory.recall (tag/kind/session
        # pre-filters, since/until on event time).
        k = max(1, min(int(k), 15))
        result = self._memory(agent).recall(query, k=k, session_id=session, since=since,
                                            until=until, tags=tags, kind=kind)
        return _result_json(result, k, order)

    def cmd_recall_many(self, agent: str, queries: list[str], k: int = 10,
                        order: str = "relevance") -> dict[str, Any]:
        from mark.intelligence.retrieval import RetrievalPolicy
        from mark.memory.timeline_order import apply_order

        k = max(1, min(int(k), 15))
        policy = RetrievalPolicy.DEEP if k > 10 else RetrievalPolicy.BALANCED
        results = self._memory(agent).retrieve_many_sync(list(queries), policy=policy)
        # Chronological order re-sorts the shown top-k, never widens it.
        return {"results": [_result_json(result, k, order) for result in results]}

    def cmd_timeline(self, agent: str, subject: str, attribute: str | None = None,
                     limit: int | None = None, since: str | None = None,
                     until: str | None = None) -> dict[str, Any]:
        timeline = self._memory(agent).timeline(subject, attribute, limit=limit,
                                                since=since, until=until)
        payload = timeline.to_dict()
        payload["text"] = timeline.as_text()
        return payload

    def cmd_fact_set(self, agent: str, subject: str, attribute: str, value: str,
                     valid_from: str | None = None, correction: bool = False) -> dict[str, Any]:
        fact = self._memory(agent).facts.set(subject, attribute, value, valid_from=valid_from,
                                             correction=correction)
        return {"fact": _fact_json(fact)}

    def cmd_fact_retract(self, agent: str, subject: str, attribute: str | None = None,
                         value: str | None = None) -> dict[str, Any]:
        return {"retracted": self._memory(agent).facts.retract(subject, attribute, value=value)}

    def cmd_fact_get(self, agent: str, subject: str, attribute: str | None = None) -> dict[str, Any]:
        return {"facts": [_fact_json(f) for f in self._memory(agent).facts.current(subject, attribute)]}

    def cmd_fact_history(self, agent: str, subject: str, attribute: str) -> dict[str, Any]:
        return {"facts": [_fact_json(f) for f in self._memory(agent).facts.history(subject, attribute)]}

    def cmd_stats(self, agent: str) -> dict[str, Any]:
        from mark.embeddings.info import embedder_info

        return {"memories": len(self.runtime.store.retrievable_ids(agent)),
                "facts": len(self.runtime.store.fact_current(agent)),
                "embedder": embedder_info(self.runtime.embedder).to_dict()}


def _fact_json(fact: Any) -> dict[str, Any]:
    return {
        "subject": fact.subject, "attribute": fact.attribute, "value": fact.value,
        "valid_from": fact.valid_from.isoformat(),
        "valid_to": fact.valid_to.isoformat() if fact.valid_to else None,
        "text": fact.as_text(), "current": fact.is_current,
    }


def _result_json(result: Any, k: int, order: str = "relevance") -> dict[str, Any]:
    from mark.intelligence.temporal_order import same_day_order
    from mark.memory.timeline_order import fragment_order

    shown = list(zip(result.fragments, result.scores))[:k]
    if order == "chronological":
        shown = [shown[i] for i, _ in fragment_order([fragment for fragment, _ in shown])]
    elif order != "relevance":
        raise ValueError(f"order must be 'relevance' or 'chronological', got {order!r}")
    fragments = [fragment for fragment, _ in shown]
    same_day = same_day_order(fragments)
    rows = [memory_json(fragment) for fragment in fragments]
    _mark_shared_dates(fragments, rows)
    for row, (fragment, score) in zip(rows, shown):
        row["score"] = round(score, 3)
        # Kept for JSON consumers of 0.2.0a9; the text output uses "order".
        row["same_day"] = "{}/{}".format(*same_day[fragment.id]) if fragment.id in same_day else None
    return {
        "query": result.query,
        "facts": [_fact_json(f) for f in result.facts],
        "memories": rows,
        "gap": result.gap_report.severity.value,
        "gap_reason": result.gap_report.reason,
        "missing": list(result.missing_anchors),
        "temporal_intent": result.temporal_intent,
    }


def _mark_shared_dates(fragments: list[Any], rows: list[dict[str, Any]]) -> None:
    """Set ``order`` = k (oldest first) on results that share a date with another result."""
    from mark.memory.timeline_order import fragment_order

    by_date: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        row["order"] = None
        by_date.setdefault(row["date"], []).append(index)
    for indices in by_date.values():
        if len(indices) > 1:
            ranked = fragment_order([fragments[i] for i in indices])
            for position, (local, _basis) in enumerate(ranked, start=1):
                rows[indices[local]]["order"] = position
