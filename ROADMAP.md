# MARK SDK Roadmap

This roadmap covers the open Python SDK: work that can be implemented,
tested, documented, and shipped in the `mark-sdk` package. Plans change as we
learn from real workloads; open a discussion if something here matters to you.

## Current release: 0.2.0a10

The last alpha before 1.0. Everything from 0.2.0a9 (hybrid recall,
bi-temporal facts and extraction, timelines, briefings, token budgets,
context replacement, image memory, the CLI, MCP and LangChain integrations,
encryption at rest), plus:

- **One memory API.** `remember`, `recall`, `brief`, `supersede`, `resolve`,
  `forget` and friends on every memory object, with `a`-prefixed async twins,
  memory kinds, and one rendering family (`as_context`, `as_text`,
  `to_dict`) on every result; the same verbs in the CLI and as MCP tools.
- **Memory lifecycle and data management.** Cascading and secure forget,
  whole-word `forget(match=)` with a preview, vacuum, open-TODO listing,
  paginated listing, JSONL backup and restore.
- **Remember anything.** Documents and files (`store_document`,
  `remember_file`, including PDF and Office files), first-person and
  structured facts with a stored user subject, typed, scheduled and
  multi-valued facts, natural dates, multi-hop recall, and current-over-stale
  ranking for free text.
- **Context that fits small models.** Provenance-aware briefings, presets by
  model size, and a one-call session brief for CLI, MCP and Python.
  llama3.2:3b answers 24 of 25 personal-assistant probes with about 233
  prompt tokens per turn.
- **Speed at scale.** Recall p50 35 ms at 100k memories and about 180-270 ms
  at 1M; 1.7 s cold start at 100k and about 8 s of CPU at 1M; multilingual
  keyword search; `mark serve --detach` for warm command-line calls.
- **Safe by default.** Structure-based secret redaction and multilingual
  prompt-injection screening on every write; a store that remembers its
  embedder; no migration under a running older server; a close gate for
  stores shared with background threads; bounded shutdown and crash-safe
  fact jobs.

## Next: 1.0.0

1.0 freezes the public API and makes stability the priority:

- **API freeze.** The memory verbs, result rendering, CLI commands and MCP
  tool names become stable under semantic versioning. Deprecated aliases
  get a published removal schedule.
- **Stability and upgrade guarantees.** Upgrades from every 0.2.0 alpha
  store, documented compatibility, and release-blocking regression suites
  (scale benchmark, retention A/B, small-model briefing probes).
- **Windows and macOS CI.** Test every release on all three platforms, and
  fuzz the parsers and the store.

Known gaps to close before or soon after 1.0:

- **Cross-lingual recall.** A question in one language does not yet find a
  memory written in another (for example English to Spanish or Russian),
  with any built-in embedder.
- **Indirect questions** that share no words with the answer remain the
  weakest recall category with the default hash embedder.
- **Cold start at a million memories** takes about 8 s of CPU, but 14-16 s
  of wall time on a loaded machine; the goal is under 10 s wall anywhere.
- **Multi-hop questions for small models.** Chains that combine two facts
  about the user (a visitor coming to "my" city after a move) still fail
  with 1-3B models.
- **Facts from developer notes.** `TODO:` and `DECISION:` notes always set
  a memory's kind, but become facts only when they state a value in a
  recognised pattern ("the job queue uses SQLite"); open TODOs and decisions
  such as "we use Postgres for billing-api" yield no fact yet.
- **Encryption support in the command line**, and alias resolution in
  timelines (one subject under several names).

## Later

Under consideration: sealing fact timelines into the block chain, an entity
channel as a third retrieval signal, an optional local reranker, and a
reproducible long-memory evaluation harness.

## Good areas for contributors

- **Examples and recipes** for coding, research, and creative-continuity
  agents, each comparing behaviour with and without MARK.
- **Adapter hardening**: LangChain edge cases, MCP client setup notes, tests.
- **Retrieval diagnostics**: explain why a memory was returned.
- **Migrations**: fixtures for older alpha stores.
- **Redaction coverage** for sync envelopes and logs.
- **Documentation** that describes what MARK does today, without claims the
  SDK does not support.

## Not in scope for this package

- Account-based or hosted infrastructure (MARK Cloud — coming soon).
- Required network calls during normal local use.
- Heavy media-generation dependencies in the install path.
- Generated media in the repository or distribution.

## How to help

Start with a small issue, work on a branch, and open a pull request. See
[CONTRIBUTING.md](CONTRIBUTING.md).
