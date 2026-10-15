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
  `to_dict`) on every result.
- **Memory lifecycle and data management.** Cascading and secure forget,
  vacuum, open-TODO listing, paginated listing, JSONL backup and restore.
- **Remember anything.** Documents and files (`store_document`,
  `remember_file`), first-person and structured facts, typed, scheduled and
  multi-valued facts, natural dates, multi-hop recall, and current-over-stale
  ranking for free text.
- **Context that fits small models.** Provenance-aware briefings, presets by
  model size, and a one-call session brief for CLI, MCP and Python.
- **Speed at scale.** Recall p50 35 ms at 100k memories and 182 ms at 1M;
  1.7 s cold start at 100k; multilingual keyword search.
- **Safe by default.** Secret redaction and prompt-injection screening on
  every write; bounded shutdown and crash-safe fact jobs.

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

- **Paraphrase recall with the default embedder.** llama3.2:3b answers 20 of
  25 probes with MARK; reaching 22 needs either a semantic embedder by
  default or better lexical bridging (synonyms, related words) in the hash
  embedder.
- **First recall after a write at 1M memories.** It recomputes the
  retrievable-id scope (about 0.15 s at 100k, 2-3 s at 1M); the scope should
  update incrementally.
- **Tool names the user mentions** can be reported as "not in memory"; they
  should be recognised as part of the request.
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
