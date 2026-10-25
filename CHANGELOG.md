# Changelog

All notable changes to the MARK Python SDK will be documented here.

This project follows semantic versioning for public package releases.

## 0.2.0a10 - 2026-10-18

The last alpha before 1.0: one memory API across Python, the command line
and MCP, documents and files, safer defaults, and recall that stays fast at
a million memories.

### Upgrade notes

- **Back up first.** Copy `memory.db` (with its `-wal` and `-shm` files)
  before the first open with 0.2.0a10. The store is upgraded in place to
  schema version 6.
- **Stop everything that uses the store** (`mark serve`, MCP servers,
  agents) before upgrading. 0.2.0a10 refuses to migrate a store while a
  0.2.0a8 or 0.2.0a9 `mark serve` is running on it and raises
  `StoreInUseError`; `mark serve` also refuses to start next to one.
- **No way back to older versions.** 0.2.0a9 refuses an upgraded store
  (`StoreVersionError`). 0.2.0a8 and older have no such guard: never reopen
  an upgraded store with them. They write into schema 6 without updating its
  new tables, and on an encrypted store opened without the key they write
  plaintext. 0.2.0a10 quarantines such rows (tagged `mark:unreadable`,
  reported by `integrity_check()`, left out of exports), but the plaintext
  stays on disk until it is forgotten with `secure=True` and the store is
  vacuumed.
- **One-time re-embedding.** The hash embedder is now version 3: memories
  that contain CJK text, emoji or symbols are re-embedded once on the first
  open; other memories are untouched.
- **The store remembers its embedder.** Opening a store without
  `embedder=` reuses the embedder it was created with. If that embedder
  cannot be loaded, the store opens keyword-only with one warning instead of
  re-embedding. An explicit `embedder=` or `MARK_EMBEDDER` re-embeds once and
  becomes the store's embedder.
- **Secrets are redacted by default** on every write; pass `redact=False`
  to opt out for data you control. Errors, stack traces and HTTP failures
  are now remembered by default (`FailureFilter` is opt-in).
- **`MarkChat` uses the `"small"` preset by default** (a 200-token
  briefing); pass `preset="default"` for the previous size.
- **Curated top-level exports.** `mark` exports 29 names. Every other former
  top-level name still imports, with a `DeprecationWarning` naming the
  submodule to import it from.
- **`forget(match=...)` matches whole words** (not substrings), needs at
  least 3 characters and `all=True` above 50 matches. `mark forget --match`
  lists the matches and asks first (`--yes` for scripts); MCP `mark_forget`
  needs `confirm=true` when several memories match.
- **`resolve()` closes only TODOs and bugs**; other kinds raise
  `InvalidInputError` (use `supersede()`).
- **Encrypted stores** build their token sketches on the first identifier
  recall or timeline in each process (a few seconds at 20,000 memories).
  Keep one long-running process and call
  `runtime.warm_up(sketches=True)` at start, or opt in to
  `persist_token_sketches()`.

### Added

- **One memory API.** `mark.memory` and `runtime.memory(agent_id)` share the
  verbs `remember`, `remember_many`, `recall`, `brief`, `forget`, `get`,
  `list`, `count`, `update`, `supersede`, `resolve`, `facts`, `timeline`,
  `store_document`, `remember_file`, `documents`, `images`, `export_jsonl`,
  `import_jsonl`, `vacuum` and `embedder_info`, with the same arguments and
  defaults. Plain names are synchronous and an `a` prefix marks the async
  twin (`arecall`). `store_sync`, `retrieve_sync` and `retrieve` keep
  working; `mark.memory.retrieve()` is deprecated in favour of
  `recall(...).as_context()`.
- **Memory kinds**: `remember(text, kind="todo")` (decision, todo, bug, fix,
  preference, snippet, api, note), or a leading `TODO:`, `DECISION:`/`ADR:`,
  `BUG:`, `FIX:`, `DONE:`/`NOTE:`, `API:` or `SNIPPET:` label in Python, the
  CLI and MCP; `list(kind=..., open=True)`, `recall(open=True)`,
  `MemoryFragment.kind` and `mark.kinds.kind_of()`.
- **Memory lifecycle**: `resolve()` closes TODOs and bugs; `supersede()` and
  `update()` replace a memory and keep its history (revising an already
  superseded memory revises its current version, `current_version()`);
  `forget()` cascades to the facts extracted from the memory;
  `forget(secure=True)` plus `vacuum()` erase the text from the database,
  its WAL and any vector snapshot; `facts.delete()` hard-deletes facts.
- **Safe forgetting by text**: `forget(match=...)` matches whole words and
  phrases and `dry_run=True` returns a `ForgetPreview`; `mark forget --match`
  offers `--dry-run`, `--yes` and `--all`.
- **Session brief**: `memory.brief()`, `mark brief` and the MCP `mark_brief`
  tool return one budgeted block with current facts, open TODOs, recent
  decisions, bugs paired with their fixes, and recent changes.
- **Data management**: `get()` (short ids accepted), `count()`, paginated
  `list()`, `Mark.agents()`, and JSONL backup and restore (`export_jsonl` /
  `import_jsonl`, `mark export` / `mark import`): imports are idempotent,
  skip malformed lines with a report, and restore each fact once; encrypted
  stores export only with `decrypt=True`.
- **Documents and files**: `store_document()` stores long text or a file as
  chunk memories linked to one document; chunks follow headings, paragraphs
  and code blocks and carry their section path; `memory.documents` lists,
  gets and forgets documents. `remember_file()` routes any file by its
  content: text (UTF-8, UTF-16/32 with a byte-order mark, Latin-1), Markdown,
  code, HTML, Word, Excel and PowerPoint (with speaker notes), PDF (new `pdf`
  extra), images, and audio or video through a transcriber
  (`mark.media.OpenAITranscriber`).
- **Fact extraction** understands first-person statements as facts about
  the user, preferences (`user.likes`, `user.prefers`), pets, job and phone
  number changes, rescheduled appointments, prices and measurements with
  units, counts, tools and project settings, key/value lines, JSON and YAML
  records, config blocks, Markdown tables, lists of ports, owners, emails or
  languages, numbered steps, ticket status and values stated in developer
  notes ("the CLI entry point is ..."). "No longer", "quit", "gave up",
  "used to" and "Correction:" retract or replace a value, "from A to B"
  keeps A in history, and "Starting <date>" schedules a value. Only the
  user's words in a chat are extracted, never assistant or tool text.
  Writes and documents report how many facts they produced.
- **A user subject stored with the store**: `Mark.local(user_subject=...)`,
  `configure_user_subject()`, `mark config user-subject NAME` and
  `mark mcp --user-subject` name who "I" is; Python, the CLI and MCP use it
  for extraction and for "I/my" questions.
- **Facts**: values with a future `valid_from` are scheduled
  (`facts.scheduled()`, `Fact.is_scheduled`); values keep their Python type;
  `facts.add()`/`facts.remove()` handle attributes with several current
  values; `facts.get_all()`; `valid_from`, `as_of` and `occurred_at` accept
  natural dates; `mark fact list [SUBJECT]`.
- **Time in recall**: natural and relative dates in questions and memories
  ("3rd of May", "last Tuesday", "Q3", "in August", "last December"),
  `since=`/`until=` filters, and dates named in a question favour memories
  from that period.
- **Multi-hop recall**: names and identifiers in the best results lead to
  the next fact, so two- and three-step chains reach the top results; graph
  edges accept any relation name.
- **Presets by model size**: `preset="small"` (200 tokens, for ~1-3B
  models) and `"tiny"` (160 tokens) on `MarkChat`, `MarkAgentMiddleware`,
  `ContextWatch.for_preset()` and `mark mcp --preset`. On a 25-question
  personal-assistant test with real local models, llama3.2:3b answered 24 of
  25 with about 233 prompt tokens per turn (0.2.0a9: 21 with 534), and
  llama3.2:1b 20 of 25 (0.2.0a9: 7).
- **`MarkChat`** stores each user message as written and extracts facts from
  it (`extract_facts=True`), stores the model's reply as low-trust context,
  and can stream (`on_token`) and report progress (`on_status`).
- **MCP**: new tools `mark_brief`, `mark_fact`, `mark_timeline`, `mark_list`,
  `mark_resolve`, `mark_supersede` and `mark_forget`; `mark_write` takes
  `kind`, and `mark_write`/`mark_observe` take `extract_facts`. Outputs use
  8-character ids, which every tool accepts. `mark mcp` and `python -m
  mark.adapters.mcp` take `--db`, `--agent`, `--briefing`, `--preset`,
  `--tools` (which may name `mark_briefing`), `--allow-agent`,
  `--user-subject` and `--extract-facts`.
- **Command line**: `remember` (with `--kind`, `--tag`, `--importance`,
  `--extract-facts`), `forget`, `resolve`, `supersede`, `list`, `export`,
  `import`, `brief`, `agents`, `config`, `mcp`, and `serve --detach`,
  `serve status` and `serve stop`; `recall` gains `--open`, `--kind` (open
  memories unless `--all`), `--session`, `--since`, `--until` and `--tag`.
- **The store remembers its embedder**: `Mark.local(path)`, the CLI and MCP
  reuse it when no embedder is given, and `embedder_info()` shows the active
  embedder and how it was chosen (`source="store"`).
- **Prompt-injection screening** on every write: instruction overrides,
  role reassignment and messages addressed to an AI reader in about 25
  languages, smuggled tool calls and chat-template markers, instructions
  hidden in invisible HTML, exfiltration links built from templates, and
  letter-spaced, leetspeak or base64-encoded payloads. Flagged memories are
  tagged `mark:untrusted` with reasons under `metadata["mark_security"]` and
  labelled as untrusted wherever they are rendered; `injection="quarantine"`
  keeps them out of recall.
- **Write options** on `Mark.local()` and `MarkRuntime.local()`: `redact=`,
  `injection=`, `max_memory_bytes=`, `fact_job_lease=` and `user_subject=`.
- `store_many_sync` / `remember_many` return a `StoreReport` that lists
  skipped items and why (`strict=True` raises instead), and
  `idempotency_key=` makes retries safe.
- `MarkRuntime.register_closer()` stops background readers before
  `shutdown()` closes the store; `MarkRuntime.warm_up()` loads the index
  ahead of the first recall.
- Every result type (recall results, timelines, briefings, context bundles)
  offers `as_context()`, `as_text()` and `to_dict()`; recall results list
  replaced memories in `result.outdated`.
- `SessionFilter.exclude_tags`.
- Encrypted stores: exact identifier recall, and `persist_token_sketches()`
  to keep timeline sketches on disk (with a documented leakage trade-off).
- The opt-in `mark-memory-compact` skill for hosts that want a short
  instruction block.
- `benchmarks/scale/bench.py` measures store, recall, cold start and recall
  quality at any size, with regression budgets for CI.
- Examples for coding-agent memory and for documents and files.

### Changed

- **Secret redaction is on by default** on every write path (single and
  batched writes, observations, healed and quarantined memories, memory
  blocks, documents, files, transcripts and caller metadata). Secrets become
  labelled placeholders such as `[SECRET:stripe_key]`, so recall still shows
  that a key existed. Redaction recognises secrets by structure: any
  secret-named key in env, YAML, JSON, TOML and connection strings,
  command-line secret flags (`--api-key`, `mysql -p`, `sshpass -p`,
  `curl -u` and others), netrc entries, URL credentials, private key blocks,
  vendor tokens (OpenAI, Anthropic, Stripe, AWS, GitHub, GitLab, Slack,
  Google, Hugging Face, PyPI, Docker Hub, Vault, Discord, Telegram and
  more), Luhn-checked card numbers, PINs, 12-24 word seed phrases, and
  passwords written in prose in about 20 languages. Template placeholders
  such as `your-api-key-here` are kept. `RedactionPolicy(pii=True)` also
  removes emails, phone numbers (international and national formats),
  checksum-validated IBANs and SSNs.
- **Errors are remembered by default** so agents can learn from failures;
  `FailureFilter` is an explicit opt-in on `ConsolidationGate`.
- The top-level `mark` namespace is curated (29 names); see the upgrade
  notes.
- **Recall at scale**: at 100,000 memories p50 35 ms (was 309 ms) and p95
  50 ms (was 538 ms); at 1,000,000 p50 about 180-270 ms (was 5.1 s).
  Keyword queries scale with their rarest terms. A recall right after a
  write applies only that write's changes instead of rescanning the scope
  (about 200 ms at a million memories).
- **Faster reopening**: embeddings load in bulk, agents with 50,000 or more
  memories keep a vector snapshot and a scope snapshot, and snapshots load
  memory-mapped. Cold start is 1.7 s at 100,000 memories (was 11 s) and
  about 8 s of CPU at a million (about 9 s wall on a quiet machine; 0.2.0a9
  took 48 s). `mark serve stop` rewrites a snapshot only when many memories
  changed.
- **Multilingual keyword search**: CJK text is segmented, other languages
  are no longer run through the English stemmer, accented words match their
  unaccented spelling, emoji and symbols are searchable, in keyword
  search and in the hash embedder (version 3), and Chinese, Japanese and
  Korean names are recognised in timelines, fact lookups and "not in
  memory" checks. Store schema version 6 adds
  the multilingual keyword index, vector and scope snapshots, and an index
  for TTL expiry; existing stores upgrade in place.
- **Recall prefers what is current**: updates ("moved to", "switched to",
  "no longer", "I now report to", "Update: ...", "Correction:") rank above
  what they replace even when the old value is not named; outdated
  statements are labelled and listed below their replacement; conflicting
  statements come newest first and flagged; "originally/previously"
  questions reverse the order. Superseded and resolved memories are demoted
  only for current-state questions, so error text, "why did ... fail" and
  "how did we fix" questions still find resolved bugs and fixes, and
  "What TODOs are still open?" lists open notes of that kind first.
- **Honest results**: results are sorted by the score they display, ties
  are ordered the same way on every call, and unrelated memories no longer
  show a perfect score. "Not in memory" reports missing code identifiers and
  names as written with one suggested action, never for an exact identifier
  match, and less often for names that exist in memory. Empty and
  punctuation-only queries return nothing.
- **Leaner context**: facts attach only when the question asks for them
  (with everyday synonyms such as live/location and boss/manager, at most
  three per subject); no zero-relevance padding; `as_context(max_tokens=N)`
  is a hard cap; long memories show their best passage; a document
  contributes at most two chunks, with the chunk holding the answer first;
  exact literal hits (emoji, CJK, identifiers) win over vector-only matches.
  On a coding-assistant test, recall context fell from about 1,250 to about
  160 tokens per question.
- **Briefings separate the user from the model**: assistant replies, tool
  output and errors are stored as low-trust context, never anchor a briefing
  and never appear as user facts; questions and requests are never echoed
  back as notes. Briefings put the request's best lines first, show dense
  fact lines with the value they replaced, pair bugs with their fixes, drop
  outdated lines under the small and tiny presets (and mark them otherwise),
  carry the user's current facts into first-person requests, show a
  correction for one turn only, and keep open TODOs and bugs at small
  budgets.
- **Briefings and recall are faster**: outdated-line detection reads each
  memory once, and matching known fact subjects stays fast with thousands
  of subjects (about 3.7 ms with 10,000).
- The LangChain middleware stores each reply once as low-trust context,
  tags tool results and errors, and counts its skill text and memory
  wrapper inside `budget_tokens`.
- **Durability**: background fact jobs held by a process that died are
  requeued on the next open and `facts.extract_pending()` waits at most
  30 s by default; `shutdown()` stops workers within `timeout`
  (default 5 s), returns unfinished jobs to the queue and closes the store,
  which then raises `StoreClosedError`; `Mark.close()` is an alias.
- **Embedder outages no longer lose writes**: network embedders retry
  timeouts, connection errors and HTTP 429/5xx with backoff; if the
  embedder stays down, memories are stored unembedded, found by keyword
  meanwhile, and embedded once it is back, without stalling a recall. Errors
  that remain are an `EmbeddingHTTPError` naming the endpoint. `"auto"` warns
  once when it falls back to the hash embedder, and model download errors
  explain `HF_HOME`.
- **One store-path rule**: `Mark.local("./.mark")`, `mark --db .mark` and
  `mark mcp --db .mark` open the same `./.mark/memory.db`; `Mark.local()`
  also accepts a project folder or a `.db` file.
- **`mark serve`** holds an OS file lock released even when the server is
  killed, warms its index in the background, and stops on an authenticated
  request. Commands in flight when a server dies finish in-process without
  duplicates. Served commands cost about 0.1 s.
- **Command-line output** shows 8-character ids (accepted by every command),
  stored dates, kinds, relevance scores and outdated, superseded and
  resolved labels (`--open` drops them); `ingest` reports new and existing
  memories and the facts it extracted, and a retried ingest never stores
  duplicates.
- MCP errors, including unknown arguments, come back as one-sentence error
  results; `mark_retrieve` stays within `budget_tokens` (default 1500), shows
  each long memory's best passage, scores and state labels, and strips
  control characters.
- `wrap_agent()` agents are callable (`agent(...)`, `invoke`, `ainvoke`).
- Stores in a `.mark` folder are made owner-only when opened.

### Fixed

- Closing a store while another thread was querying it could crash the
  process; `close()` now waits up to 5 s for running statements and refuses
  new ones with `StoreClosedError`, and a `ContextWatch` that was never
  closed is stopped before its runtime or store closes.
- A newer MARK could migrate a store under a running older `mark serve`,
  whose later writes then missed the new tables.
- Documents, files and transcripts kept their full original text before
  redaction; they are now redacted and screened before anything is stored,
  and every chunk of a document flagged as an injection is tagged
  `mark:untrusted`.
- `RetrievalResult.as_text()` and `brief()` defuse `</memory_context>`,
  `<system>` and other boundary or role-tag lookalikes, as `as_context()`
  already did.
- Opening an encrypted store that holds plaintext rows no longer takes
  seconds per row, and fact values stored as a bytes representation are
  repaired.
- A shut-down store could be reopened silently by a later call; shutdown no
  longer leaks worker threads or file handles.
- The first `ContextWatch` briefing for a new request waits for its prefetch
  even when the caller passes a shorter `wait`.
- Writes validate their input and raise `InvalidMemoryError` (or
  `MemoryTooLargeError` above `max_memory_bytes`, default 1 MB) instead of
  raw pydantic, JSON or Unicode errors; bytes are no longer coerced to text;
  unknown block names raise `UnknownBlockError`, including over MCP. One
  invalid item no longer fails a whole batch.
- An empty file raises a MARK error, and image memory refuses files that are
  not images (`NotAnImageError`).
- A back-dated change now beats an undated earlier statement, and "moved
  ... in September" is dated to September.
- Long floats such as `0.30000000000000004` no longer match versions such
  as `0.3.0`.
- `update_embedding()` and secure forgetting keep vector snapshots correct.
- Redaction no longer leaves partial values such as `[REDACTED]&3` or
  `+234 [PHONE]`, and `ContentSanitizer` no longer truncates content at
  10,000 characters.
- Command-line fixes: read commands on a missing store say so instead of
  creating it; `mark recall` rejects an empty query and `-k 0`;
  `serve stop` exits 1 when no server runs and no longer fails when the
  server closes the connection while stopping; `--no-server` is accepted after
  the sub-command; `brief` on an empty store says so; no more "same day"
  notes.
- `mark --version` prints the installed version (`mark-sdk 0.2.0a10`).

## 0.2.0a9 - 2026-10-15

### Added

- Automatic fact extraction: with `Mark.local(..., fact_extractor="patterns")`
  or `store_sync(..., extract_facts=True)` / `store_many_sync(...,
  extract_facts=True)`, stated changes in stored text ("moved from port A to
  port B", "is owned by", "status: resolved") are recorded as bi-temporal
  facts, dated by the memory's event time and linked to their source memory.
  Present-tense descriptions are recognised too ("runs on port 6430", "runs
  in eu-west-1", "dana owns X", "is based in Berlin", "reports to Y", "uses
  PostgreSQL 15", "is assigned to Y"), including lines that list several
  facts about one subject in comma- or "and"-separated clauses.
  Built-in rule packs (`PatternFactExtractor`, `RulePack`, `BUILTIN_PACKS`)
  run inline and accept custom packs; `LLMFactExtractor` uses a local
  OpenAI-compatible model from a durable background queue, and
  `memory.facts.extract_pending(wait=True)` drains that queue on demand and
  waits for jobs the worker is still running, so their facts are readable
  when it returns (`wait=False` or a timeout in seconds). Any object
  implementing the `FactExtractor` protocol can be used. On the retention
  benchmark's large profile, the generic packs reached precision 1.0 on every
  target attribute, with recall 1.0 for owner, port, region, database, team,
  city, and ticket assignee, and 0.96 for ticket status.
- `mark ingest --extract-facts` records the facts that ingested lines state.
- Subject timelines: `memory.timeline(subject, attribute=None, limit=,
  since=, until=)` returns every fact and memory about a subject, oldest
  first and without a top-k cap, as a `Timeline` with `latest()`,
  `as_text()`, `as_context()`, and `to_dict()`. `mark timeline SUBJECT
  [ATTRIBUTE]` offers the same with `--limit`, `--since`, `--until`, and
  `--json`.
- `retrieve_sync(..., order="chronological")` and `mark recall
  --chronological` list recall results oldest first by event time.
- Briefings for any agent loop: `ContextWatch.observe_messages()` follows an
  OpenAI-style or LangChain message list, `inject(messages, briefing)` places
  the briefing in the system message without accumulating copies, and
  `ContextWatch.for_small_models()` is a compact preset for small local
  models. `ContextWatch.refresh()` re-fetches subjects after new memories are
  stored.
- MCP briefings: `create_mark_mcp_server(..., briefing=True)` adds a
  `mark_briefing` tool and a `mark://briefing/{session_id}` resource, with one
  isolated watch per session.
- Token budgets: `ContextBudget(max_tokens, fill=False, token_counter=None)`
  caps injected context in tokens and keeps lines by priority (corrections,
  then unknown subjects, facts, records, and linked subjects). It is available
  as `ContextWatch(budget_tokens=...)`, `inject(..., budget_tokens=...)`,
  `mark_briefing(budget_tokens=...)`, and
  `MarkAgentMiddleware(budget_tokens=..., token_counter=...)`, which applies
  one cap to everything the middleware injects. `Briefing.usage` reports the
  tokens used per section.
- Context replacement: `MarkAgentMiddleware(history="mark", keep_last=6)`
  sends the model only recent messages plus MARK's context. Dropped turns are
  stored first, tool calls are never separated from their results, and the
  full history is sent if storing fails. `MarkChat` is a reference chat loop
  for any OpenAI-compatible endpoint that works the same way, and
  `split_history` is available for custom loops.
- Embedder selection by spec string: `embedder="auto"`, `"hash[:dim]"`,
  `"model2vec[:model]"`, `"fastembed[:model]"`, `"ollama[:model][|url]"`, or
  `"openai:<base_url>|<model>"`, plus `auto_embedder()`, `resolve_embedder()`,
  and the `MARK_EMBEDDER` and `MARK_EMBEDDER_API_KEY` environment variables.
  `"auto"` uses a model already on disk and never downloads or contacts a
  server.
- Image memory: `memory.images` (`ImageMemory`) remembers image files by path
  and SHA-256, links them to their subjects' facts, finds them by caption or
  CLIP similarity (`ClipImageEmbeddingProvider`, `mark-sdk[vision]`),
  verifies files, and checks a frame against expected facts with a describer
  such as `OpenAIVisionDescriber`.
- Async `memory.retrieve_many()`.
- Examples for briefings in a plain loop, visual continuity, and MARK as the
  context window.

### Changed

- Hybrid retrieval adapts to the embedder: lexical embedders keep rank
  fusion, and semantic models use score fusion (`FusionConfig`, set through
  `RetrievalPipeline(fusion=...)` or `pipeline.configure_fusion()`;
  `EmbeddingProvider.semantic` marks the kind). On a 60-query evaluation
  (paraphrase, synonym, identifier, and name queries), hybrid hit@1 rose from 44 to 51 (potion-base-8M), 45 to 54
  (potion-retrieval-32M), 48 to 59 (bge-small), and 49 to 58 (MiniLM); the
  default hash embedder is unchanged.
- The default embedder is still the deterministic hash embedder; semantic
  models are opt-in.
- `RetrievalResult.facts` contains only the facts that answer the query (the
  attribute asked about, the role a who-question names, such as `owner` for
  "Who owns X?" or `assignee` for "Who is assigned to X?", or all of them for
  an open question). A who-question about a subject without that fact
  returns no facts rather than unrelated ones.
- Briefings no longer report subjects that only the model mentioned as "Not
  in memory"; such subjects appear once memory confirms them. Bare numbers
  (ports, counts, years, dates) are never reported as unknown subjects or as
  `missing_anchors`.
- On POSIX, a new `.mark/` directory is created with mode `0700` and new
  database files with mode `0600`.
- Store schema version 5 adds the fact-extraction job queue and extraction
  ledger; existing stores upgrade in place.

### Fixed

- An encrypted store opened without its key, or with the wrong one, could
  fail on read or accept new writes in plaintext. It now raises
  `EncryptionRequiredError` or `EncryptionKeyError`, and supplying a key for
  a store that already holds plaintext raises `PlaintextStoreError` (an
  empty store may become encrypted). The errors are in `mark.errors` and
  `mark.security`.
- Switching to an embedding provider whose dimension is not known in advance
  did not re-embed existing memories; switching embedders now re-embeds once.
- Opening a new store from several processes at once could fail while
  enabling WAL mode.
- A store written by a newer MARK is now refused with `StoreVersionError`
  before anything is written, so an older installation can no longer migrate
  or modify a store it does not understand.
- A memory whose stored content cannot be decrypted (for example plaintext
  written into an encrypted store by an older release opened without its key)
  is set aside as unreadable instead of making every retrieval fail.

## 0.2.0a8 - 2026-10-15

### Added

- Anticipatory context: `ContextWatch` follows an agent's conversation, tool
  calls and tool results, prefetches memory about the subjects in play
  (including linked subjects, such as the service named by a ticket), and
  provides a compact briefing before each model call without the agent
  issuing a search.
- `MarkAgentMiddleware` injects the anticipatory briefing automatically
  (`anticipate=True` by default), including corrections when the model states
  a value that memory records as superseded, and notices for subjects that are
  not in memory.
- Bi-temporal facts: `memory.facts.set/get/current/history/as_of` record values
  that change over time with validity intervals. Late-arriving updates are
  placed correctly in history, corrections are audited rather than deleted,
  and fact values are encrypted at rest when encryption is enabled. Several
  changes recorded for the same time are kept as a sequence; `correction=True`
  replaces a wrong value instead.
- Time-aware retrieval: queries about the current or original state of a
  subject return the newest or earliest matching record first, using
  `occurred_at` or a leading date stamp as the event time.
- `occurred_at=` on `store_sync` and `store_many_sync` to record when an event
  happened, independent of when it was stored.
- Batch operations: `retrieve_many` / `retrieve_many_sync` answer several
  queries in one pass, and `store_many_sync` embeds and writes many memories
  in a single transaction.
- `mark` command-line interface: `ingest`, `store`, `recall`, `recall-many`,
  `fact set|get|history`, `stats`, and `serve`.
- `mark serve` keeps a store warm for fast command-line use, restricted to
  loopback with token authentication and a per-database lock.
- `RetrievalResult` exposes `facts`, `temporal_intent`, and `missing_anchors`.
- `memory.facts.retract()` and `mark fact retract` withdraw facts recorded by
  mistake; retracted facts are kept for audit.
- Recall output labels events that share a date with their order that day,
  so the latest of several same-day changes is clear.

### Changed

- Gap reports now flag named entities and identifiers that do not occur
  anywhere in memory, even when other words in the query match.
- `import mark` loads public names on first use, reducing import time from
  roughly 370 ms to under 10 ms. All public names remain available.
- The default vector index switches to numpy only once an index is large
  enough to benefit, so short-lived processes with small stores start faster.
- Store schema version 4 adds the facts table; existing stores upgrade in
  place.

### Fixed

- Events with the same date were not ordered by when they happened.
- Keyword search over large filtered scopes could miss in-scope matches.
- Term matching treated unrelated identifiers with a shared prefix (for
  example `titan` and `titanium`) as the same word.
- Concurrent creation of a new store from several processes could fail.
- A `mark` client and a running `mark serve` from different builds could fail
  mid-command; the client now detects the mismatch and runs locally.
- Several modules could not be imported on their own because of circular
  imports.

## 0.2.0a7 - 2026-10-15

### Breaking changes

- Explicit retrieval policies (`FAST`, `BALANCED`, `DEEP`) are applied as
  specified; use `RetrievalPolicy.AUTO` for automatic sizing.
- `FernetEncryptionProvider` requires a key or passphrase; pass
  `ephemeral=True` for a temporary key.
- The default web search skill used by gap healing requires a `web`
  permission; `MarkAgentMiddleware(auto_heal=...)` defaults to `False`.
- `BusSnapshot.facts_by_trust(level)` returns facts at or above `level`.
- Docker sandbox workspaces must be inside an explicit `workspace_root`.
- Redaction uses `[REDACTED]` for secrets and `[EMAIL]` for email addresses;
  sync envelopes no longer include a content hash.
- `HashEmbeddingProvider` uses a new 256-dimension feature space; existing
  stores are re-embedded automatically.

### Known limitations

- Encryption at rest covers memory content; tags, metadata,
  session ids, graph labels, and embeddings remain plaintext so they can be
  filtered. See SECURITY.md.

### Added

- Hybrid retrieval combining keyword (BM25) and vector search with rank
  fusion, so exact identifiers, names, and paraphrases are all found.
- Calibrated relevance scores and per-channel hit counts on retrieval results.
- `ReinforceMode` to control how retrieval affects memory importance.
- Local embedding providers for OpenAI-compatible servers (llama.cpp, LM
  Studio, vLLM, LocalAI), Ollama batch embedding, FastEmbed, and Model2Vec.
- Automatic re-embedding when the embedding model changes.
- Vectorised search with the optional `fast` extra.
- Asynchronous `MarkAgentMiddleware` support (`ainvoke`, `astream`).
- MCP namespace allowlist (`allowed_agent_ids`) and support for `mcp` 2.x.
- Optional signing of trust-bus messages.
- Persistent memory feedback through `SimpleMemory.feedback()`.

### Fixed

- Retrieval considered only an agent's 200 most important memories.
- One agent's memories could displace another agent's results.
- Retrieval inflated the importance of whatever it returned.
- Gap reports did not flag queries with no relevant memory.
- Memories failing integrity checks could still be returned.
- Thread-safety issues in the vector index, embedding cache, and in-memory store.
- `SentenceTransformerEmbeddingProvider` could not be constructed.
- `MarkAgentMiddleware` failed on asynchronous runs, leaked context between
  concurrent runs, and could inject stale context into later turns.
- Trust levels could be forged, asynchronous writes bypassed governance, and
  web lookups were vulnerable to server-side request forgery.
- Secret redaction missed common credential formats.
- Encrypted stores kept unkeyed hashes of plaintext content.
- The Docker sandbox could not execute code and left containers running
  after timeouts.
- Maintenance operations could modify sealed memory blocks.
- Editable installs could import an empty `mark` package.

## 0.2.0a6 - 2026-06-16

### Added

- Added `AgentRun`, a reusable per-run lifecycle object for `MarkAgent`
  executions, including prompts, session data, metadata, retrieved memories,
  policy decisions, skill outputs, stored output IDs, and middleware mutations.
- Added production-oriented `MarkAgent` support for optional middleware,
  policy actions, skill execution, output persistence, per-run metadata, and
  runtime-backed recall while preserving a no-LLM fallback path.
- Added `mark.middlewares`, a framework-neutral middleware layer for wrapping
  local `store`, `observe`, and `retrieve` operations through public runtime
  APIs.
- Added local middleware for recall defaults, observation source/tags/metadata,
  compressor binding, query-expander binding, explicit gap healing, lifecycle
  run-cycle triggers, local tracing, sync envelope preparation, governance,
  trust-bus publishing, blocked-by-default sandbox execution, and local media
  continuity memory.
- Added middleware extras so developers can install the full middleware bundle
  with `mark-sdk[middleware]` or choose specific battery targets such as
  `middleware-governance`, `middleware-trust-bus`, `middleware-sandbox`, and
  `middleware-media-continuity`.
- Added `MarkRuntime.add_middleware()`, `MarkRuntime.use()`, and
  `Mark.local(..., middleware=[...])` for ergonomic middleware composition.
- Added broader MCP adapter tool options for scoped retrieval, observation,
  canonical writes, tags, session IDs, agent IDs, and block filters.
- Added packaged MARK usage skills under `mark.middlewares.skills` so installed
  users can load middleware guidance through `importlib.resources`.
- Added regression coverage for LangChain tool sync calls inside running event
  loops, native async `ainvoke()` usage, middleware context injection, MCP tool
  options, and `MarkAgent` lifecycle behavior.
- Live developer usage examples under `examples/`:
  - `01_local_memory_live.py` for local `Mark.local()` storage/retrieval.
  - `02_agent_ab_live.py` for with/without MARK agent context comparison.
  - `03_sessions_and_observe_live.py` for session-aware `observe()` continuity.
  - `04_sync_envelope_live.py` for redacted sync envelope preparation.
  - `run_live_examples.py` to run the live examples as MVP smoke tests.
- Python 3.10+ release coverage in CI and release workflows
  (`3.10`, `3.11`, `3.12`, `3.13`, `3.14`).
- Graph-scoped memory blocks: blocks now group nodes and the edges between
  them per topic, session, or world-bible scope (schema v2 with in-place
  migration of existing stores).
  - `BlockGraph` (`memory.blocks()`): many-to-many node membership,
    block-scoped edges, forward/backward `BlockLink`s, and BFS traversal.
  - `BlockChain` (`memory.chain()`): `seal()` computes a deterministic SHA-256
    content hash over immutable member fields and chains sealed blocks through
    `prev_block_hash`; `verify()` pinpoints corrupted, missing, or injected
    members; `verify_chain()` checks whole-chain integrity.
  - Block quarantine: a quarantined block's fragments leave retrieval and
    traversal without affecting other blocks or agent function; `release()`
    restores it.
  - Retrieval accepts `block_id` / `block_ids` filters
    (`retrieve_sync(..., block_id=...)`), and quarantined-block members are
    always excluded from results.
- Functional test suites for the MCP server adapter and LangChain tools
  adapter (round-trip write/retrieve/observe against the local backend).
- `examples/getting_started_with_mark.ipynb`: runnable notebook walkthrough —
  the same coding agent without MARK, with `MarkAgentMiddleware`, with MARK
  LangChain tools, and over MCP, plus memory inspection and block provenance.
- URL safety validation for the web lookup skill: only public http/https
  hosts are fetchable (file://, loopback, private, and link-local targets are
  rejected).
- Docstrings across the entire public API surface (modules, classes, and
  functions).

### Changed

- Changed the SDK license from MIT to Apache-2.0 for explicit patent protection.
- Split middleware into one package per battery under `mark.middlewares`.
- Reworked LangChain tool creation to expose `StructuredTool` instances with
  both sync and async implementations backed by the same MARK backend.
- Clarified LangChain guidance: middleware is the automatic context-loading
  path, tools are the deliberate/canonical memory-write path, and combining both
  is an advanced mode that should avoid duplicate context.
- Updated the LangChain extra to install the agent middleware dependency surface
  required by the current adapter tests.
- Reframed `MarkAgentMiddleware` as a message-context memory layer and made the
  optional `thinking_model` documentation explicitly experimental and opt-in.
- Repositioned public documentation around MARK as an agent memory
  runtime for creative continuity, workflow state, middleware integration,
  structured recall, and provenance rather than a generic memory store.
- Expanded architecture and examples documentation to explain agent-loop
  integration, retrieval/context injection, creation memory, and planned
  proof-oriented demos for character consistency and multi-agent coding.
- Updated README and examples documentation so media, video, music, and
  generated demo outputs are not described as current package examples.
- Core install now depends only on `pydantic`; unused `openai` and
  `langchain-openai` core dependencies were removed.
- The `adapters` extra now installs LangChain and MCP dependencies only.
- README rewritten as a concise install/usage guide with adapter examples;
  ARCHITECTURE.md now documents the local runtime including memory blocks.
- Sync redaction now recursively redacts metadata, source fields, tags, and
  optional event payloads before a sync envelope leaves local code.
- The explicit web fetch skill now rejects hostnames that resolve to private,
  loopback, link-local, reserved, multicast, or unspecified addresses.

### Fixed

- Fixed `MarkAgent` so configured policies, skills, middleware, and output
  storage participate in agent runs instead of being isolated constructor data.
- Fixed LangChain MARK tools raising inside notebook/async environments when a
  sync `invoke()` happened while an event loop was already running.
- Fixed LangChain middleware context handling for missing system messages and
  string/list message content.
- Fixed canonical `mark_write(..., canonical=True)` memories to store with high
  importance and canonical tags so retrieval ranks them above episodic
  observations.
- Fixed MCP and backend retrieval wiring so block filters are passed through the
  public runtime/backend API consistently.

### Removed

- Removed packaged media/image/video/music demo files and their test claims from
  the current SDK package.
- Removed heavy provider-demo dependencies such as `gradio-client` and `pillow`
  from SDK extras and lockfile state.
- The empty LlamaIndex adapter namespace and its extra.
- Tests and tutorial scripts are no longer packaged in the sdist; the wheel
  ships only the `mark` package.

## 0.2.0a4 - 2026-06-07

### Changed

- Renamed the PyPI distribution from `mark` to `mark-sdk` while preserving the
  Python import name `mark`, because the public PyPI name `mark` is already
  occupied by another project.
- Added PyPI project URLs pointing at the public SDK repository:
  `emsoftanalytics/MARK-SDK`.
- Reworked the README into a release-ready quickstart with install, tutorial,
  build, publish, and repository strategy sections.
- Added a runnable `tutorial.py` coding-task walkthrough using the installed
  `mark` package import.
- Added a GitHub Actions release workflow for tests, build artifacts, and PyPI
  Trusted Publishing.

## 0.2.0a3 - 2026-06-07

### Added

- `LocalMemoryStore` lifecycle methods: `update_importance`, `touch` (records
  `last_accessed_at` and increments `access_count` in fragment metadata),
  `list_edges`, `delete_edge`, `update_edge_weight`, `edges_to_node`,
  `integrity_check`, `export`, and `schema_version`.
- Schema versioning via `schema_info` table and `_MIGRATIONS` dict. Pre-versioned
  databases (created before `schema_info` existed) are detected from fragment
  presence and migrated forward automatically.
- `MarkRuntime` hippocampus lifecycle: `working_memory()`, `consolidate()`,
  `prune()`, `run_cycle()`, and `integrity_check()`.
- Hebbian reinforcement wired through `RetrievalPipeline.retrieve()`: each
  retrieved fragment is touched (access count incremented) and its importance
  boosted via dampened LTP. Co-retrieved fragment pairs have shared graph edges
  strengthened via `EdgeCoActivation`. Each edge is strengthened at most once
  per retrieval (dedup by edge ID). Both outgoing and incoming edges are
  considered (undirected strengthening).
- `touch()` now runs before `on_access()` in the reinforcement loop so that
  `access_count` is current when the Hebbian boost is computed — first-access
  reinforcement now takes effect immediately.
- 20 new tests in `test_hippocampus.py` covering store methods, lifecycle,
  reinforcement order, co-activation (outgoing and incoming), schema versioning,
  export, and pre-versioned DB migration.
- Session-aware retrieval with exact-session, session-prefix, tags, tier, and
  scope filters.
- Graph API for nodes, edges, neighborhoods, attached fragments, graph context,
  and graph-expanded retrieval.
- Local media memory helpers for character, object, location, scene/session, and
  world-bible continuity.
- Contextual compression with no-op, window, and developer-provided LLM modes.
- Structured extraction with deterministic extraction, optional developer LLM
  extraction, and merge/dedup logic.
- Query expansion with no-op, keyword, and developer-provided LLM variants.
- Embedding cache provider for lightweight local reuse.
- JSON-to-SQLite migration for legacy `memory.json` stores.
- Deduplication consolidation with clustering, centroids, provenance, graph-node
  reattachment, and `run_cycle()` integration.
- Trust-aware global bus with publisher trust, subscriptions, snapshots,
  persisted rehydration, and trust-filtered retrieval.
- Local governance heuristics: sanitizer, validation gate, duplicate hash gate,
  failure filter, consolidation gate, and in-process audit log.
- Local observability: tracer, session trace, JSONL event replay, and runtime
  maintenance events.
- Sync preparation with `SyncDelta`, block filtering, local redaction, optional
  event inclusion, and optional caller-supplied upload forwarding.
- Local RL/plasticity data structures and tests kept Apache-2.0-safe.

### Changed

- `Mark.local()` is now the single public entrypoint. It delegates to
  `MarkRuntime.local()` (SQLite at `.mark/memory.db`), wraps it in
  `SimpleMemory`, and exposes the full runtime via `mark.runtime`. The
  `Mark.local()` / `MarkRuntime.local()` split is resolved: `MarkRuntime`
  remains importable for explicit use, but developers should start with
  `Mark.local()`.
- README and ARCHITECTURE updated to reflect the unified entrypoint and SQLite
  default. JSON references removed.
- `ConsolidationManager` now runs local governance before promotion and records
  audit entries through `MarkRuntime.governance_audit_log()`.
- `MarkRuntime.run_cycle()` now performs consolidation, deduplication, and
  pruning, returning a richer maintenance summary.
- `MarkRuntime.trust_bus()` and `global_bus()` now return stable runtime-scoped
  instances instead of fresh wrappers on every call.

### Fixed

- Co-activation double-strengthening: the same edge could previously be
  strengthened once as an outgoing edge from fragment A and again as an incoming
  edge from fragment B in a single retrieval. A `seen_edges` set now ensures each
  edge is updated at most once per call.
- Reinforcement order: `touch()` now precedes `on_access()`, and the fragment is
  re-fetched so the boost formula sees the correct `access_count`.
- Pre-versioned SQLite DBs (no `schema_info` table) are now detected by checking
  for existing fragment rows; they are migrated forward and stamped with the
  current schema version rather than being silently treated as fresh databases.
- Secret redaction now catches `API key: ...` and `sk-...` token patterns in
  local sync envelopes.
- Trust-bus messages can be rehydrated from persisted global-bus fragments.
- Deduplication now reattaches nodes from duplicate fragments to the kept
  canonical fragment before deleting the duplicate.

## 0.2.0a2 - Unreleased

### Added

- SDK architecture document covering `Mark.local()`, `MarkRuntime.local()`,
  runtime extension points, plugin placement, and MCP placement.
- Coding-task tutorial showing MARK as a tool and middleware with a simple
  with/without MARK benchmark.
- Repo-level example relocation for generated notebook documentation assets.

### Changed

- README now documents the difference between `Mark.local()` and
  `MarkRuntime.local()`.

## 0.2.0a1 - Previous alpha checkpoint

### Added

- Lab-derived local hippocampus/cortex runtime through `MarkRuntime.local()`.
- SQLite local memory store, local vector index, graph-aware retrieval pipeline,
  plugin hooks, and stronger memory lifecycle types.
- Conversation memory, persona memory, global memory bus, working memory,
  consolidation, contradiction detection, local plasticity, RL signal schemas,
  policy safety constraints, optional embedding adapters, and Docker sandbox
  support.

### Changed

- Built-in skills keep the async skill contract and support synchronous usage
  through `run_sync()`.
- The package version now uses PyPI-compatible pre-release numbering for SDK
  phase checkpoints.

## 0.1.0 - Previous foundation

### Added

- Local `Mark.local()` runtime with JSON-backed project memory.
- Lab-derived `MarkRuntime.local()` runtime with SQLite storage, local vector
  index, graph expansion, retrieval pipeline, and plugin hooks.
- Memory types for fragments, blocks, documents, nodes, edges, scopes, tiers,
  lifecycle states, gap reports, and attribution.
- Agent wrapper, policy registry, skill registry, built-in MARK memory skill,
  and packaged `mark-usage` skill instructions.
- Conversation memory, persona memory, global memory bus, working memory,
  consolidation, contradiction detection, local plasticity, and safety schemas.
- Optional local embedding adapters for Ollama and sentence-transformers.
- Local sandbox support for Docker-backed code execution.
- Coding-task tutorial showing MARK as a tool and middleware with simple
  benchmarks.

### Changed

- Built-in skills keep the async skill contract and support synchronous usage
  through `run_sync()`.
- Public SDK comments describe extension points by capability only.

### Security

- Added content hashing, optional local encryption provider interfaces, redaction
  helpers, and raw-data opt-in boundaries for local signal collection.
