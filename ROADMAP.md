# MARK SDK Roadmap

This roadmap covers the open Python SDK: work that can be implemented,
tested, documented, and shipped in the `mark-sdk` package. Plans change as we
learn from real workloads; open a discussion if something here matters to you.

## Current release: 0.2.0a8

- Local memory through `Mark.local()`, with memory blocks, sessions, and agent
  namespaces in one SQLite file.
- Hybrid recall: keyword (BM25) and vector search with rank fusion, and no LLM
  calls on store or recall.
- Time-aware recall for "now" and "original" questions, using `occurred_at` or
  a leading date stamp.
- Bi-temporal facts: `set`, `get`, `current`, `history`, `as_of`, `retract`,
  corrections, and same-day sequences.
- Gap reports that name subjects which are not in memory.
- Batch store and recall (`store_many_sync`, `retrieve_many_sync`).
- `ContextWatch` briefings, and LangChain middleware that injects them
  automatically (`anticipate=True`).
- LangChain tools, an MCP server, and the `mark` command line with
  `mark serve`.
- Local embedding providers: Ollama, OpenAI-compatible servers, FastEmbed,
  Model2Vec, sentence-transformers, and the zero-dependency default.
- Provenance blocks (seal, verify, quarantine), lifecycle maintenance, opt-in
  middleware, and encryption of content and fact values at rest.

## Next: 0.2.0a9

- **Automatic fact extraction.** Record facts from stored text, so agents do
  not have to call `facts.set` themselves.
- **Subject timelines.** One view of everything memory knows about a subject,
  in event order.
- **Proactive context beyond LangChain.** Briefings for MCP clients and for
  plain agent loops.
- **Stronger default embeddings.** Better semantic recall out of the box,
  while keeping the core install light.
- **Image memory.** Remember images for visual continuity across generations
  and sessions.
- **Token-budgeted context.** Fit injected memory to a token budget, and
  replace stale context instead of only appending to it.

## Later

- **Cold start and backup.** Faster first use of a large store, and simple
  backup and restore.
- **A simpler API.** Fewer entry points for common tasks.
- **Windows and macOS CI.** Test every release on all three platforms.

Under consideration: sealing fact timelines into the block chain, an entity
channel as a third retrieval signal, a background extraction queue, an
optional local reranker, and a reproducible long-memory evaluation harness.

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
