"""Command execution against one database, in-process or inside ``mark serve``.

:class:`_Engine` is the only place the CLI loads MARK; the import happens in
its constructor so merely importing this module stays standard-library only.
Results are converted to plain JSON-able dicts so the same payload works
in-process and over the loopback server.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any


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


class _Engine:
    """Loads MARK lazily and executes commands against one database."""

    def __init__(self, db: Path) -> None:
        from mark.memory.runtime import MarkRuntime

        db.parent.mkdir(parents=True, exist_ok=True)
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
                   tags: list[str] | None = None, extract_facts: bool = False) -> dict[str, Any]:
        texts = [line.strip() for line in lines if line.strip()]
        # extract_facts=True: built-in pattern packs, each line's "[YYYY-MM-DD]" stamp as valid_from.
        ids = self._memory(agent).store_many_sync(texts, session_id=session, tags=tags or None,
                                                  extract_facts=True if extract_facts else None)
        if not extract_facts:
            return {"stored": len(ids)}
        return {"stored": len(ids), "facts": len(self.runtime.store.fact_current(agent))}

    def cmd_store(self, agent: str, text: str, session: str | None = None,
                  occurred_at: str | None = None, importance: float = 0.8) -> dict[str, Any]:
        fid = self._memory(agent).store_sync(
            text, session_id=session, occurred_at=occurred_at, importance=importance,
        )
        return {"id": fid}

    def cmd_recall(self, agent: str, query: str, k: int = 10) -> dict[str, Any]:
        return self.cmd_recall_many(agent, [query], k)["results"][0]

    def cmd_recall_many(self, agent: str, queries: list[str], k: int = 10) -> dict[str, Any]:
        from mark.intelligence.retrieval import RetrievalPolicy

        k = max(1, min(int(k), 15))
        policy = RetrievalPolicy.DEEP if k > 10 else RetrievalPolicy.BALANCED
        results = self._memory(agent).retrieve_many_sync(list(queries), policy=policy)
        return {"results": [_result_json(result, k) for result in results]}

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
        return {"memories": len(self.runtime.store.retrievable_ids(agent)),
                "facts": len(self.runtime.store.fact_current(agent))}


def _fact_json(fact: Any) -> dict[str, Any]:
    return {
        "subject": fact.subject, "attribute": fact.attribute, "value": fact.value,
        "valid_from": fact.valid_from.isoformat(),
        "valid_to": fact.valid_to.isoformat() if fact.valid_to else None,
        "text": fact.as_text(), "current": fact.is_current,
    }


def _result_json(result: Any, k: int) -> dict[str, Any]:
    from mark.intelligence.temporal_order import same_day_order

    shown = list(zip(result.fragments, result.scores))[:k]
    same_day = same_day_order([fragment for fragment, _ in shown])
    return {
        "query": result.query,
        "facts": [_fact_json(f) for f in result.facts],
        "memories": [
            {"score": round(score, 3), "content": fragment.content, "id": fragment.id,
             "same_day": "{}/{}".format(*same_day[fragment.id]) if fragment.id in same_day else None}
            for fragment, score in shown
        ],
        "gap": result.gap_report.severity.value,
        "gap_reason": result.gap_report.reason,
        "missing": list(result.missing_anchors),
        "temporal_intent": result.temporal_intent,
    }
