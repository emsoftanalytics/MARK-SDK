"""``mark`` command line: local memory for agents and scripts.

    mark [--db PATH] [--agent ID] <command> ...

    ingest FILE [--session S] [--tag T ...]      store each non-empty line as a memory
    store TEXT [--session S] [--occurred-at T]   store one memory
    recall QUERY [-k N] [--json]                 hybrid recall (facts first, then memories)
    recall-many [FILE|-] [-k N] [--json]         one query per line (or a JSON list)
    fact set SUBJECT ATTRIBUTE VALUE [--valid-from T]
    fact get SUBJECT [ATTRIBUTE]                 current value(s)
    fact history SUBJECT ATTRIBUTE               every value over time
    stats                                        number of retrievable memories
    serve [--port N]                             keep MARK warm for fast calls

The database comes from ``--db`` or ``$MARK_DB`` (default ``./.mark/memory.db``),
the agent namespace from ``--agent`` or ``$MARK_AGENT`` (default ``__mark__``).

When ``mark serve`` is running for the same database, every other command is
sent to it over loopback HTTP instead of loading MARK in-process: a call then
costs one small Python start-up plus milliseconds of search, instead of
importing and opening MARK each time. The server binds to loopback only and
requires a random token stored next to the database in a 0600 state file.

This module imports only the standard library at start-up; MARK itself is
loaded lazily, and never on the client path when a server answers.
"""
from __future__ import annotations

import argparse
import json
import os
import secrets
import socket
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

DEFAULT_AGENT = "__mark__"
MAX_REQUEST_BYTES = 8 * 1024 * 1024
_LOOPBACK = {"127.0.0.1", "::1", "localhost"}


# ── configuration ─────────────────────────────────────────────────────────────

def _db_path(arg: str | None) -> Path:
    raw = arg or os.environ.get("MARK_DB") or str(Path(".mark") / "memory.db")
    return Path(raw).expanduser().resolve()


def _state_path(db: Path) -> Path:
    return db.with_name(db.name + ".server.json")


# ── command implementations (run in-process or inside the server) ─────────────

class _Engine:
    """Loads MARK lazily and executes commands against one database."""

    def __init__(self, db: Path) -> None:
        from mark.memory.runtime import MarkRuntime

        db.parent.mkdir(parents=True, exist_ok=True)
        self.runtime = MarkRuntime.local(store_path=db)

    def close(self) -> None:
        self.runtime.shutdown()

    def run(self, cmd: str, args: dict[str, Any]) -> dict[str, Any]:
        handler = getattr(self, f"cmd_{cmd.replace('-', '_')}", None)
        if handler is None:
            raise ValueError(f"unknown command {cmd!r}")
        return handler(**args)

    def _memory(self, agent: str):
        return self.runtime.memory(agent)

    def cmd_ingest(self, agent: str, lines: list[str], session: str | None = None,
                   tags: list[str] | None = None) -> dict[str, Any]:
        texts = [line.strip() for line in lines if line.strip()]
        ids = self._memory(agent).store_many_sync(texts, session_id=session, tags=tags or None)
        return {"stored": len(ids)}

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
                     valid_from: str | None = None) -> dict[str, Any]:
        fact = self._memory(agent).facts.set(subject, attribute, value, valid_from=valid_from)
        return {"fact": _fact_json(fact)}

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
    return {
        "query": result.query,
        "facts": [_fact_json(f) for f in result.facts],
        "memories": [
            {"score": round(score, 3), "content": fragment.content, "id": fragment.id}
            for fragment, score in list(zip(result.fragments, result.scores))[:k]
        ],
        "gap": result.gap_report.severity.value,
        "gap_reason": result.gap_report.reason,
        "missing": list(result.missing_anchors),
        "temporal_intent": result.temporal_intent,
    }


# ── server ────────────────────────────────────────────────────────────────────

def serve(db: Path, host: str = "127.0.0.1", port: int = 0, *, stop_event: Any = None) -> None:
    """Serve commands for *db* on loopback until interrupted (or ``stop_event`` is set)."""
    import signal
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    if host not in _LOOPBACK:
        raise SystemExit("mark serve only binds to loopback (127.0.0.1, ::1, localhost)")
    state = _state_path(db)
    if _live_server(state) is not None:
        raise SystemExit(f"a MARK server is already running for {db}")
    engine = _Engine(db)
    token = secrets.token_urlsafe(32)

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args: Any) -> None:  # keep stdout clean
            return

        def _reply(self, status: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self) -> None:  # noqa: N802 - http.server API
            if not secrets.compare_digest(self.headers.get("Authorization", ""), f"Bearer {token}"):
                self._reply(401, {"error": "unauthorized"})
                return
            length = int(self.headers.get("Content-Length") or 0)
            if length <= 0 or length > MAX_REQUEST_BYTES:
                self._reply(413, {"error": "request too large or empty"})
                return
            try:
                request = json.loads(self.rfile.read(length))
                self._reply(200, {"ok": True, "result": engine.run(request["cmd"], request.get("args", {}))})
            except Exception as exc:  # report, keep serving
                self._reply(400, {"ok": False, "error": f"{type(exc).__name__}: {exc}"})

    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    _write_state(state, {"host": host, "port": server.server_address[1], "token": token, "pid": os.getpid(),
                         "db": str(db)})
    stop = threading.Event()

    def _shutdown(*_: Any) -> None:
        if not stop.is_set():
            stop.set()
            threading.Thread(target=server.shutdown, daemon=True).start()

    if stop_event is not None:
        threading.Thread(target=lambda: (stop_event.wait(), _shutdown()), daemon=True).start()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, _shutdown)
        except (ValueError, OSError):  # not the main thread / unsupported
            pass
    print(f"MARK serving {db} on {host}:{server.server_address[1]} (Ctrl+C to stop)", flush=True)
    try:
        server.serve_forever(poll_interval=0.2)
    finally:
        server.server_close()
        engine.close()
        try:
            state.unlink()
        except FileNotFoundError:
            pass


def _write_state(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as handle:
        json.dump(data, handle)
    os.replace(tmp, path)


def _live_server(state: Path) -> dict[str, Any] | None:
    """Server details when a server for this database answers; stale files are ignored."""
    try:
        data = json.loads(state.read_text())
        with socket.create_connection((data["host"], int(data["port"])), timeout=0.3):
            return data
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _call_server(server: dict[str, Any], cmd: str, args: dict[str, Any]) -> dict[str, Any]:
    request = urllib.request.Request(
        f"http://{server['host']}:{server['port']}/",
        data=json.dumps({"cmd": cmd, "args": args}).encode(),
        headers={"Authorization": f"Bearer {server['token']}", "Content-Type": "application/json"},
        method="POST",
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))  # never via a proxy
    try:
        with opener.open(request, timeout=300) as response:
            payload = json.loads(response.read())
    except urllib.error.HTTPError as exc:
        payload = json.loads(exc.read() or b"{}")
    if not payload.get("ok"):
        raise RuntimeError(payload.get("error", "server error"))
    return payload["result"]


def execute(db: Path, cmd: str, args: dict[str, Any], *, use_server: bool = True) -> dict[str, Any]:
    """Run a command via a live server for *db*, else in-process."""
    server = _live_server(_state_path(db)) if use_server else None
    if server is not None:
        return _call_server(server, cmd, args)
    engine = _Engine(db)
    try:
        return engine.run(cmd, args)
    finally:
        engine.close()


# ── output ────────────────────────────────────────────────────────────────────

def _print_recall(result: dict[str, Any]) -> None:
    for fact in result["facts"]:
        print(f" F. {fact['text']}{'' if fact['current'] else '  [historical]'}")
    if not result["memories"] and not result["facts"]:
        print("(no memories found)")
    for rank, memory in enumerate(result["memories"], 1):
        print(f"{rank:2d}. [{memory['score']:.3f}] {memory['content']}")
    if result["gap"] not in {"none", "low"}:
        reason = f": {result['gap_reason']}" if result["gap_reason"] else ""
        print(f"-- memory gap: {result['gap']}{reason}")


# ── entry point ───────────────────────────────────────────────────────────────

def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mark", description="Local memory for agents and scripts.")
    parser.add_argument("--db", help="SQLite store path (default: $MARK_DB or ./.mark/memory.db)")
    parser.add_argument("--agent", default=os.environ.get("MARK_AGENT", DEFAULT_AGENT), help="agent namespace")
    parser.add_argument("--no-server", action="store_true", help="always run in-process")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("ingest", help="store each non-empty line of FILE")
    p.add_argument("file")
    p.add_argument("--session")
    p.add_argument("--tag", action="append", dest="tags")
    p = sub.add_parser("store", help="store one memory")
    p.add_argument("text")
    p.add_argument("--session")
    p.add_argument("--occurred-at")
    p = sub.add_parser("recall", help="search memory")
    p.add_argument("query")
    p.add_argument("-k", type=int, default=10)
    p.add_argument("--json", action="store_true")
    p = sub.add_parser("recall-many", help="search memory for many queries at once")
    p.add_argument("file", nargs="?", default="-")
    p.add_argument("-k", type=int, default=10)
    p.add_argument("--json", action="store_true")
    fact = sub.add_parser("fact", help="bi-temporal facts").add_subparsers(dest="fact_cmd", required=True)
    p = fact.add_parser("set")
    p.add_argument("subject")
    p.add_argument("attribute")
    p.add_argument("value")
    p.add_argument("--valid-from")
    p = fact.add_parser("get")
    p.add_argument("subject")
    p.add_argument("attribute", nargs="?")
    p = fact.add_parser("history")
    p.add_argument("subject")
    p.add_argument("attribute")
    sub.add_parser("stats", help="count memories and facts")
    p = sub.add_parser("serve", help="keep MARK warm for fast calls")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=0)
    return parser


def _read_queries(source: str) -> list[str]:
    text = sys.stdin.read() if source == "-" else Path(source).read_text(encoding="utf-8")
    stripped = text.strip()
    if stripped.startswith("["):
        loaded = json.loads(stripped)
        return [str(item["question"] if isinstance(item, dict) else item) for item in loaded]
    return [line.strip() for line in text.splitlines() if line.strip()]


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    db = _db_path(args.db)
    if args.cmd == "serve":
        serve(db, args.host, args.port)
        return 0

    agent = args.agent
    if args.cmd == "ingest":
        lines = Path(args.file).read_text(encoding="utf-8").splitlines()
        cmd, payload = "ingest", {"agent": agent, "lines": lines, "session": args.session, "tags": args.tags}
    elif args.cmd == "store":
        cmd, payload = "store", {"agent": agent, "text": args.text, "session": args.session,
                                 "occurred_at": args.occurred_at}
    elif args.cmd == "recall":
        cmd, payload = "recall", {"agent": agent, "query": args.query, "k": args.k}
    elif args.cmd == "recall-many":
        cmd, payload = "recall-many", {"agent": agent, "queries": _read_queries(args.file), "k": args.k}
    elif args.cmd == "fact":
        cmd = f"fact-{args.fact_cmd}"
        payload = {"agent": agent, "subject": args.subject}
        if args.fact_cmd == "set":
            payload.update(attribute=args.attribute, value=args.value, valid_from=args.valid_from)
        else:
            payload["attribute"] = args.attribute
    else:
        cmd, payload = "stats", {"agent": agent}

    try:
        result = execute(db, cmd, payload, use_server=not args.no_server)
    except (RuntimeError, ValueError, OSError) as exc:
        print(f"mark: {exc}", file=sys.stderr)
        return 1

    if getattr(args, "json", False):
        print(json.dumps(result, indent=2))
    elif cmd == "recall":
        _print_recall(result)
    elif cmd == "recall-many":
        for result_item in result["results"]:
            print(f"### {result_item['query']}")
            _print_recall(result_item)
    elif cmd.startswith("fact-"):
        facts = [result["fact"]] if "fact" in result else result["facts"]
        for fact in facts:
            print(fact["text"])
        if not facts:
            print("(no facts)")
    elif cmd == "ingest":
        print(f"stored {result['stored']} memories")
    elif cmd == "store":
        print(f"stored {result['id']}")
    else:
        print(f"{result['memories']} memories, {result['facts']} current facts")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
