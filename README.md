# MARK (Memory-Augmented Agent Recall Kit) Python SDK

MARK is local-first memory for AI agents. It installs as `mark-sdk`, imports
as `mark`, and keeps everything in one SQLite file: no account, server,
network, or API key. Store and recall make no LLM calls.

An agent remembers what it learns, decides, and produces; MARK recalls the
relevant parts later, keeps values that change over time as facts, and hands
the agent a short, token-budgeted brief before it starts.

## Quickstart (60 seconds)

```bash
pip install mark-sdk          # Python 3.11+; the core depends only on pydantic
```

```python
from mark import Mark

with Mark.local("./.mark") as mark:              # one SQLite file: ./.mark/memory.db
    mem = mark.memory

    mem.remember("We use FastAPI for the public API.", kind="decision")
    todo = mem.remember("TODO: add rate limiting to /login", kind="todo")
    mem.remember("[2026-03-01] atlas-core runs on port 6430.", extract_facts=True)

    print(mem.recall("What do we use for the API?", k=2).as_context())  # ready for a prompt
    print(mem.facts.get("atlas-core", "port").value)                      # 6430

    mem.resolve(todo)                     # done: it leaves the open-TODO list
    print(mem.brief().as_text())          # session-start brief: facts, TODOs, decisions
    mem.forget(match="rate limiting")     # whole words; gone, with any facts extracted from it
```

The same from a shell (the store defaults to `./.mark/memory.db`):

```bash
mark remember "We use FastAPI for the public API." --kind decision
mark remember "TODO: add rate limiting to /login"      # "TODO:" sets the kind
mark recall "What do we use for the API?"
mark brief
mark list --kind todo --open
```

**One store, one rule.** Python, the CLI and MCP find the store the same
way: a project folder keeps it in `<folder>/.mark/memory.db`, a folder named
`.mark` is the store folder itself, and a `.db` file is the database. So
`Mark.local(".")`, `Mark.local("./.mark")`, `mark --db .mark` and
`mark mcp --db .mark` all open `./.mark/memory.db`.

### The memory API

`mark.memory` (the default namespace `"__mark__"`, shared with the CLI, MCP
and LangChain integrations) and `mark.runtime.memory("agent-id")` (a
separate namespace) have the same verbs, arguments and defaults. Plain names
are synchronous; an `a` prefix marks the async twin (`await mem.arecall(...)`).

| Verb | What it does |
| --- | --- |
| `remember(text, kind=, tags=, session_id=, occurred_at=, extract_facts=)` | Store one memory; returns its id. Kinds: `decision`, `todo`, `bug`, `fix`, `preference`, `snippet`, `api`, `note`; without `kind=`, a leading `TODO:`, `DECISION:`/`ADR:`, `BUG:`, `FIX:`, `DONE:`/`NOTE:`, `API:` or `SNIPPET:` label sets it. `occurred_at` takes ISO text or words ("yesterday", "3 May 2026"). |
| `remember_many(texts, kind=, tags=, idempotency_key=)` | Store a batch in one transaction; returns a `StoreReport` (ids in input order, `skipped`, `existing`). |
| `recall(query, k=, kind=, tags=, session_id=, since=, until=, open=)` | Hybrid keyword + vector search (`open=True` leaves out resolved and superseded memories). Returns a result with `as_context()`, `as_text()`, `to_dict()`, `facts`, `missing_anchors`. |
| `brief(budget_tokens=400, focus=)` | One-call session brief: current facts, open TODOs, decisions, bugs and fixes, recent changes. |
| `facts` | Values that change over time: `set`, `get`, `get_all`, `history`, `as_of`, `add`/`remove`, `scheduled`, `retract`, `delete`. |
| `timeline(subject, attribute=)` | Everything known about a subject, oldest first, no top-k cap. |
| `supersede(id, text)` / `update(id, text)` | Replace a memory; the old one is kept as history and ranked down. |
| `resolve(id, note=)` | Close a TODO or bug. |
| `forget(id or match=, secure=)` | Delete a memory and the facts extracted from it; `secure=True` plus `vacuum()` erase the text from disk. |
| `get`, `list(kind=, open=)`, `count` | Read and page through memories. |
| `store_document(text_or_path)`, `remember_file(path)`, `documents`, `images` | Long documents, any file routed by content, image memories (see below). |
| `export_jsonl(path)` / `import_jsonl(path)` | Back up and restore (idempotent; bad lines are skipped and reported). `Mark` has the same for the whole store. |
| `embedder_info()` | Which embedder is active and how it was chosen. |

`mark.kinds.kind_of(memory.get(id))` gives a memory's kind. The 0.2.0a9
names (`store_sync`, `retrieve_sync`) still work; `mark.memory.retrieve()`
still works but is deprecated (it warns): use `recall(...).as_context()`.
Errors are `mark.MarkError` subclasses (`InvalidInputError`,
`MemoryTooLargeError`, `StoreClosedError`, `StoreVersionError`, ...), and
`mark.close()` is the same as `mark.shutdown()`.

## Install

| Extra | Adds |
| --- | --- |
| `mark-sdk[fast]` | numpy-vectorised vector search (recommended for large stores) |
| `mark-sdk[langchain]`, `[mcp]`, `[adapters]` | LangChain middleware and tools, the MCP server, or both |
| `mark-sdk[crypto]` | encryption at rest (`FernetEncryptionProvider`) |
| `mark-sdk[pdf]` | PDF text for `remember_file` |
| `mark-sdk[model2vec]`, `[fastembed]`, `[ollama]`, `[sentence-transformers]` | local semantic embedders |
| `mark-sdk[vision]` | CLIP image embeddings for image memory |

## Recipe: a personal assistant

First-person statements become facts about the user, newer statements win,
and anything can be forgotten for good. The user subject (who "I" is) is
stored with the store, so every later open, `mark recall` and MCP use it too;
set it from a shell with `mark config user-subject Ana` (`mark config show`
prints it).

```python
from mark import Mark

with Mark.local("./.mark", fact_extractor="patterns",   # facts from every write
                user_subject="Ana") as mark:            # "I"/"my" facts are about Ana
    me = mark.memory

    me.remember("I prefer tea over coffee.", kind="preference")
    me.remember("My dentist is Dr. Bello.")
    me.remember("I live in Lisbon.")
    me.remember("I moved to Porto in September.")               # the newer value wins

    print(me.facts.get("Ana", "location").value)                 # Porto
    print(me.recall("Where do I live?", k=2).as_context())       # the current fact comes first
    print(me.brief().as_text())                                  # includes "Lisbon -> Porto"

    pin = me.remember("My bank PIN is 4931.")
    print(me.get(pin).content)            # My bank PIN is [SECRET:pin]  (redacted on write)
    me.forget(pin, secure=True)           # delete it and its derived facts...
    mark.vacuum()                         # ...and erase the text from the file and its WAL

    mark.export_jsonl("backup.jsonl")     # restore with mark.import_jsonl("backup.jsonl")
```

`mark export backup.jsonl` and `mark import backup.jsonl` do the same from a
shell. Exports are written with owner-only permissions.

## Recipe: a coding agent

Give each memory a kind, close TODOs instead of leaving them open, and
supersede notes instead of piling up versions. Refer to code as
`file.py::Symbol`, which survives edits, rather than `file.py:123`.

```python
from mark import Mark

with Mark.local("./.mark") as mark:
    repo = mark.memory          # the namespace the `mark` CLI and MCP use too ("__mark__")
    repo.remember("DECISION: the job queue uses SQLite in WAL mode.", kind="decision")
    bug = repo.remember("BUG: uploader.py::Uploader.send drops the last chunk on retry.", kind="bug")
    old = repo.remember("Run the tests with `pytest -q`.", kind="snippet")

    repo.remember("FIX: Uploader.send flushes the final chunk before retrying.", kind="fix")
    repo.resolve(bug, note="fixed in 4f2c1e0")
    repo.supersede(old, "Run the tests with `uv run --extra dev pytest -q`.")

    print(repo.brief().as_text())                                  # start of the next session
    print(repo.recall("How do I run the tests?", k=1).top().content)
```

From the command line, which is what most coding agents call:

```bash
mark remember "DECISION: retries use exponential backoff with jitter" --kind decision
id=$(mark remember "TODO: cover uploader.py::Uploader.send with a retry test" | awk '{print $2}')
mark brief                         # facts, open TODOs, decisions, bugs and fixes
mark resolve "$id" --note "test added"
mark list --kind todo --open       # nothing left open
mark recall "Uploader.send" -k 3
mark fact set billing-api port 9191 --valid-from "last monday"
mark fact list billing-api
mark forget --match "exponential backoff" --dry-run    # lists what would go
mark forget --match "exponential backoff" --secure --yes
```

Output shows 8-character ids (`--json` for full ones), which every command
accepts. `recall` also takes `--kind` (open memories only, unless `--all`),
`--open`, `--tag`, `--session`, `--since`, `--until` (ISO dates or words such
as "last week"), `--chronological` and `--json`. `mark ingest FILE` stores
each line, sets kinds from their labels and is safe to repeat. Read commands
on a store that does not exist say so instead of creating it. `mark forget
--match` deletes memories containing those whole words (at least 3
characters): it lists them and asks first, refuses without a terminal unless
`--yes` is given, and needs `--all` for more than 50. `mark resolve` closes
TODOs and bugs only (replace anything else with `mark supersede`), and
revising an already superseded memory revises its current version.

`mark serve --detach` keeps the store warm in the background so each command
answers in about 0.1 s (loopback only, token-authenticated); it returns once
the server answers. Agents should always start it with `--detach`, since a
foreground `mark serve` blocks the shell. `mark serve status` says whether it
runs, `mark serve stop` ends it.

**MCP.** `mark mcp` (or `python -m mark.adapters.mcp`, same options) serves
the same store to any MCP client over stdio. A client configuration:

```json
{"mcpServers": {"mark": {"command": "python",
  "args": ["-m", "mark.adapters.mcp", "--db", "/path/to/project/.mark"]}}}
```

Tools: `mark_brief` (call it first), `mark_retrieve`, `mark_write` (with
`kind`), `mark_observe`, `mark_fact`, `mark_timeline`, `mark_list` (e.g. open
TODOs), `mark_resolve`, `mark_supersede` and `mark_forget` (by id or
`match`; a match hitting several memories needs `confirm=true`).
`mark_write`/`mark_observe` take `extract_facts=true`. Outputs show
8-character ids, which every tool accepts; `mark_retrieve` shows each
memory's score, labels superseded, resolved and outdated ones, and stays
within `budget_tokens` (default 1500), showing each long memory's best
passage. Options: `--db`, `--agent` (default namespace), `--briefing` (adds
`mark_briefing`), `--preset default|small|tiny`, `--tools` (a subset; may
name `mark_briefing`), `--allow-agent` (limit the namespaces a client may
use), `--user-subject` and `--extract-facts` (record facts from every
write). Errors, including unknown arguments, come back as MCP error results
with one sentence. The packaged `mark-usage` skill
(`mark.middlewares.skills`) teaches an agent these habits.

## Documents and files

```python
from pathlib import Path

from mark import Mark

Path("runbook.md").write_text("# Runbook\n\n## Rollback\nRun `make rollback`, then page #payments-ops.\n")

with Mark.local("./.mark") as mark:
    docs = mark.runtime.memory("docs")
    record = docs.store_document("runbook.md")      # or a text string; chunks follow headings
    print(record.title, len(record.chunk_ids))
    print(docs.recall("What is the rollback procedure?", k=1).as_text())  # "# Runbook › Rollback ..."
    print([d.title for d in docs.documents.list()])            # list, get, chunks, count, forget
```

`remember_file(path)` routes by the file's content: text (UTF-8, UTF-16/32
with a byte-order mark, Latin-1), Markdown, code, HTML, and Word, Excel and
PowerPoint files become documents, PDFs too with `mark-sdk[pdf]`, images become image
memories (path and SHA-256, never the bytes), and audio or video go through
a transcriber you pass (`mark.media.OpenAITranscriber` for any
OpenAI-compatible server). Long memories render as their best passage, so a
large document never crowds out the answer.

## Facts that change over time

A fact is the value of an attribute of a subject. Later values supersede
earlier ones by when they became true (`valid_from`), history is kept, and
"what was it on 20 March?" has an answer.

```python
from mark import Mark

with Mark.local("./.mark") as mark:
    facts = mark.memory.facts
    facts.set("atlas-core", "port", 6430, valid_from="2026-03-01")
    facts.set("atlas-core", "port", 4444, valid_from="2026-03-30")
    facts.set("atlas-core", "port", 5000, valid_from="2026-03-15")   # arrives late

    print(facts.get("atlas-core", "port").value)                      # 4444 (types are kept)
    print([f.value for f in facts.history("atlas-core", "port")])     # [6430, 5000, 4444]
    print(facts.as_of("atlas-core", "port", "20 March 2026").value)   # 5000 (ISO or words)
    print(facts.set("atlas-core", "port", 4500, valid_from="2099-01-01").as_text())
    # atlas-core port = 4500 (scheduled from 2099-01-01)   -- not current until then
    print(mark.memory.timeline("atlas-core").as_text())               # every event, oldest first
```

With `fact_extractor="patterns"` (or `extract_facts=True` on one write, or
`mark ingest FILE --extract-facts`), stated values become facts: "moved from
port A to port B", "X is owned by Y", "I live in Porto", key: value lines,
JSON/YAML records, Markdown tables, prices and measurements with units,
ticket status ("DONE: ABC-12 ...") and values stated in developer notes
("DECISION: the job queue uses SQLite"). "No longer" retracts a value and
"Correction:" replaces one. `LLMFactExtractor` uses a local model from a
durable background queue for phrasing the rules miss.

Recall understands time as well: "now" and "originally" questions order
records by event time, dates such as "3rd of May", "last Tuesday" or "Q3"
are understood, `since=`/`until=` filter by event time, and free-text updates
("moved from", "switched to", "used to") rank ahead of what they replace.
Names and identifiers that occur nowhere in memory are listed in
`missing_anchors`, so an agent can say it does not know.

## Context for any agent loop

`ContextWatch` follows a conversation and prepares a briefing before each
model call, within a hard token budget: current facts for the subjects in
play, the request's best lines, corrections of outdated values the agent
states, and subjects that are not in memory.

```python
from mark import ContextWatch, Mark
from mark.context.watch import inject

with Mark.local("./.mark") as mark:
    ops = mark.runtime.memory("ops")
    ops.facts.set("checkout-api", "port", 9090, valid_from="2026-03-20")
    messages = [{"role": "system", "content": "You are an on-call engineer."},
                {"role": "user", "content": "Is checkout-api healthy? Also check zephyr-9."}]

    with ContextWatch(ops, budget_tokens=300) as watch:
        watch.observe_messages(messages)                       # only messages it has not seen
        prompt = inject(messages, watch.briefing(wait=1.0))    # new list, briefing in the system message
        print(prompt[0]["content"])
```

Replies the model writes are stored as low-trust context: they never anchor a
briefing and never appear as user facts.

**MARK as the context window.** `MarkChat` talks to any OpenAI-compatible
endpoint with the standard library only. Each call sends the system prompt,
a budgeted briefing and the last `keep_last` messages; everything older lives
in MARK.

```python
from mark import Mark, MarkChat

with Mark.local("./.mark") as mark:
    with MarkChat(mark.runtime.memory("chat"), model="llama3.2:3b",
                  base_url="http://localhost:11434/v1", keep_last=2) as chat:
        chat.send("My hive is called Colmeia Azul.")
        chat.send("Name one spring flower bees like. One word.")
        chat.send("And one summer flower? One word.")
        print(chat.send("What is my hive called?"))     # from MARK, not from the window
```

**Small local models.** Presets size the briefing to what a model can use:
`preset="small"` (the `MarkChat` default, 200 tokens, for ~1-3B models) and
`"tiny"` (160 tokens, for models under 1B); `ContextWatch.for_preset(memory,
"small")` gives the same watch. On a 25-question personal-assistant test with
real local models, llama3.2:3b answered 24 of 25 with about 233 prompt
tokens per turn (0.2.0a9: 21 with 534), and llama3.2:1b 20 of 25 (0.2.0a9:
7). The opt-in `mark-memory-compact` skill exists for hosts that want an
instruction block; on small models the self-explanatory briefing headers
worked better than any added instructions.

**LangChain.** `MarkAgentMiddleware` adds the same behaviour to any LangChain
v1 agent (`invoke`, `ainvoke`, `astream`). The example uses a local Ollama
model, which needs `pip install "mark-sdk[langchain]" langchain-ollama` (or
use `langchain-openai` pointed at Ollama's OpenAI-compatible endpoint, or any
other chat model):

```python
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model

from mark import Mark
from mark.adapters.backend import LocalMarkBackend
from mark.adapters.langchain.middleware import MarkAgentMiddleware

model = init_chat_model("ollama:llama3.2:3b")                # any LangChain chat model

with Mark.local("./.mark") as mark:
    mark.memory.remember("Run the tests with `uv run --extra dev pytest`.", kind="snippet")
    middleware = MarkAgentMiddleware(backend=LocalMarkBackend(mark), budget_tokens=400,
                                     history="mark", keep_last=6)
    agent = create_agent(model, tools=[], middleware=[middleware])
    result = agent.invoke({"messages": [("user", "How do I run the tests?")]})
    print(result["messages"][-1].content)
```

`budget_tokens` caps everything the middleware injects, including its skill
text. With `history="mark"` the model sees the last `keep_last` messages plus
MARK's context; dropped turns are stored first and a tool call is never
separated from its result. `create_mark_tools(LocalMarkBackend(mark))` (in
`mark.adapters.langchain`) exposes memory as tools instead.

## Integrations

| Surface | Entry point | Use it when |
| --- | --- | --- |
| Python | `Mark.local()`, `mark.memory`, `mark.runtime.memory(id)` | Your code owns the agent loop. |
| Command line | `mark remember / recall / brief / resolve / forget / fact / timeline / export / config` | Shell scripts and coding agents that run commands. |
| MCP | `mark mcp`, `python -m mark.adapters.mcp --db PATH` | Any MCP client (desktop assistants, IDE agents). |
| LangChain | `MarkAgentMiddleware`, `create_mark_tools` | LangChain v1 agents. |
| Plain loops | `ContextWatch` + `inject`, `MarkChat`, `mark.wrap_agent(fn)` | Any OpenAI-style message list or plain function. |

All of them default to the namespace `"__mark__"`, so memory written through
one is visible to the others.

## Embedders

The default embedder is the built-in hash embedder (version 3): deterministic,
no download, multilingual-aware (CJK text is segmented; emoji and symbols are
searchable), and lexical, so recall matches words rather than meaning. For
paraphrase recall, use a local semantic model:

```python
from mark import Mark

with Mark.local("./.mark", embedder="auto") as mark:   # a model already on disk, else hash
    print(mark.embedder_info().as_text())                # e.g. "hash (lexical, dim=256, source=auto)"
```

`"auto"` never downloads or contacts a server and warns once when it falls
back to hash. Explicit specs: `"model2vec[:model]"`, `"fastembed[:model]"`,
`"ollama[:model][|url]"`, `"openai:<base_url>|<model>"`, `"hash[:dim]"`.
**A store remembers its embedder.** The embedder a store was created with
is recorded in it, and `Mark.local(path)`, the CLI and MCP use it whenever no
embedder is given (`embedder_info().source == "store"`). If it cannot be
loaded on this machine (its extra is not installed or its model not
downloaded), the store opens keyword-only with one warning saying what to
install; its vectors are left as they are. An explicit `embedder=` or
`MARK_EMBEDDER` re-embeds existing memory once with the new embedder
(instead of mixing vector spaces) and becomes the store's embedder.

## Speed

Measured with `benchmarks/scale/bench.py` (offline, default embedder, numpy
installed, SQLite on disk):

| Memories | Recall p50 | Recall p95 | Recall right after a write | Cold start |
| --- | --- | --- | --- | --- |
| 100,000 | 35 ms | 50 ms | about 40 ms | 1.7 s |
| 1,000,000 | 180-270 ms | about 400 ms | about 200 ms | about 8 s of CPU |

Keyword search scales with the rarest query terms, a write updates the
cached scope instead of rescanning it, and agents with 50,000 or more
memories keep vector and scope snapshots so a large store reopens quickly.
Wall-clock times swing with machine load: the million-memory cold start
took about 9 s on a quiet machine and 14-16 s on a loaded one. For a
long-running agent, `mark serve --detach` pays the cold start once.
Reproduce with `uv run --extra fast python benchmarks/scale/bench.py --n
100000`; CI runs `--n 20000 --check` against committed budgets.

## Security defaults

- **Secrets are redacted on write.** API keys and tokens, private keys,
  passwords, card numbers and PINs become placeholders such as
  `[SECRET:stripe_key]` before anything reaches disk. Contact details are kept
  unless you pass `redact=RedactionPolicy(pii=True)` (from `mark.security`);
  `redact=False` turns redaction off for data you control.
- **Prompt-injection screening.** Instruction-like memories ("ignore previous
  instructions" in about 25 languages, role reassignment, chat-template
  markers, instructions hidden in invisible HTML or encoded text) are tagged
  `mark:untrusted` and labelled as untrusted wherever they are rendered;
  `Mark.local(..., injection="quarantine")` keeps them out of recall. Both
  screens are heuristics: treat recalled tool, document and web text as data.
- **Private files.** On POSIX the store directory is `0700` and the database
  and exports are `0600`.
- **Encryption at rest** with `mark-sdk[crypto]`:
  `Mark.local(".", encryption=FernetEncryptionProvider(passphrase=...))`. An
  encrypted store refuses to open without its key, and exports it only with
  `decrypt=True`. Its first identifier recall in a process builds keyed
  token sketches (a few seconds at 20,000 memories); a long-running process
  can pay that at start with `mark.runtime.warm_up(sketches=True)`. The `mark`
  command line does not open encrypted stores.
- **No implicit network.** Model servers are contacted only when you name
  them; `mark serve` binds to loopback and requires a token.

See [SECURITY.md](SECURITY.md) for the details and their limits.

## Upgrading

Stores upgrade in place on first open, after an automatic backup: MARK
copies the older store to `memory.db.v<N>-backup-<UTC time>.db` next to it
(owner-only), checks the copy, then migrates and checks again. On any
failure the store is left as it was (or restored from the backup) and the
error says why, with sizes. Stop every `mark serve`, MCP server and agent
that uses the store first: 0.2.0a10 refuses to migrate while an older
server runs (`StoreInUseError`).

```bash
mark migrate --dry-run   # schemas, the backup it would make, free-space check
mark migrate             # upgrade now (verified backup first)
mark backups             # list backups: schema, size, date
mark backups restore .mark/memory.db.v5-backup-20261025T101500Z.db --yes
```

```python
from mark import Mark

mark = Mark.local(".", backup_dir="/mnt/big-disk/mark-backups")  # backup elsewhere
# Mark.local(".", backup_on_upgrade=False) or MARK_BACKUP_ON_UPGRADE=0: no backup
```

A backup holds the same memories as the store (an encrypted store's backup
stays encrypted); delete it once you no longer need it. Once
upgraded, a store must not be reopened with 0.2.0a8 or older, which do not
detect the newer schema (0.2.0a9 refuses it with `StoreVersionError`). The
[changelog](CHANGELOG.md) lists every upgrade note, including the
behaviour changes (redaction by default, whole-word `forget(match=)`,
`resolve()` for TODOs and bugs only, curated top-level imports).

## Examples

[examples/](examples/) holds short scripts that run offline: remember and
recall, an LLM with and without memory, sessions, facts over time, briefings,
redaction, briefings in a plain loop, visual continuity, MARK as the context
window, coding-agent memory, and documents and files.

```bash
python examples/run_live_examples.py
MARK_LLM_URL=http://localhost:11434/v1 MARK_LLM_MODEL=llama3.2:3b python examples/09_context_replacement.py
```

## MARK Cloud — coming soon

Watch out for MARK Cloud updates.

## Contributing and support

- Bugs and ideas: [open an issue](https://github.com/emsoftanalytics/MARK-SDK/issues/new/choose).
- Contributing: read [CONTRIBUTING.md](CONTRIBUTING.md). Work happens on
  branches and lands on `main` through reviewed pull requests.
- Design and code layout: [ARCHITECTURE.md](ARCHITECTURE.md).
- Plans: [ROADMAP.md](ROADMAP.md).
- Questions and use cases: [Discussions](https://github.com/emsoftanalytics/MARK-SDK/discussions).
- Security issues: report privately, see [SECURITY.md](SECURITY.md).

## License and trademark

Copyright 2025-2026 Somtochukwu Owen Ikeanyi. Licensed under the Apache
License, Version 2.0: see [LICENSE](LICENSE) and [NOTICE](NOTICE). If you
redistribute MARK or a modified version, keep the copyright notices and
include the NOTICE file, as the license requires.

"MARK" and the MARK logo are trademarks of Somtochukwu Owen Ikeanyi. The
license does not grant permission to use them, other than to describe where
the software comes from; please don't use them to name or promote your own
product.
