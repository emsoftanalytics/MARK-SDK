# MARK SDK Roadmap

This roadmap describes the open Python SDK direction. It is intentionally
focused on work that can be implemented, tested, documented, and shipped in the
`mark-sdk` package.

MARK's current promise is simple: agents should not forget important work.

## Current Release: 0.2.0a6

The current alpha provides:

- Local-first memory through `Mark.local()`.
- Persistent memory storage and retrieval.
- `MarkAgent` for memory-aware callable and LLM workflows.
- Optional middleware for recall, observation, compression, query expansion,
  governance, lifecycle hooks, observability, sync envelope preparation,
  trust-bus publishing, sandbox hooks, gap healing, and media continuity.
- LangChain tools and middleware adapters.
- MCP server adapter for exposing MARK memory to compatible clients.
- Graph-scoped memory blocks with provenance sealing, verification, and
  quarantine.
- Offline examples, tests, and packaging checks.

## Near Term

These are the best areas for contributors who want to help MARK become more
usable and reliable.

- Improve beginner documentation:
  - Add small recipes for common agent memory patterns.
  - Keep README examples short and runnable.
  - Expand troubleshooting for installation, optional extras, and notebooks.
- Grow examples:
  - Add local-only examples for coding agents, research agents, and creative
    continuity.
  - Keep heavyweight provider clients and generated media outside the package.
  - Add examples that compare behavior with and without MARK.
- Strengthen adapters:
  - Add more LangChain edge-case tests.
  - Improve MCP tool documentation and client setup notes.
  - Keep adapters as thin glue over public MARK APIs.
- Improve contributor experience:
  - Label starter issues.
  - Keep tests fast and offline by default.
  - Document release and package-boundary expectations.

## SDK Quality

These items harden MARK for serious adoption.

- Retrieval quality:
  - Add clearer retrieval policies and examples.
  - Add tests around ranking, session filters, block filters, and metadata.
  - Improve diagnostics for why a memory was returned.
- Middleware behavior:
  - Keep middleware opt-in and composable.
  - Document when to use middleware versus tools.
  - Add tests for middleware ordering, stop conditions, and failure handling.
- Storage and migration:
  - Keep SQLite migrations explicit and tested.
  - Improve export/import examples.
  - Add fixtures for older alpha stores.
- Security and privacy:
  - Keep the core offline by default.
  - Maintain URL safety checks for web lookup.
  - Expand redaction coverage for sync envelopes and logs.

## Community Ideas

Good community contributions include:

- Framework adapters and examples.
- Better memory inspection and debugging tools.
- Local benchmark scripts for long-running agent tasks.
- More realistic examples for software projects, research workflows, and
  creative continuity.
- Documentation that explains what MARK can do today without making claims the
  SDK does not yet support.

## Not In Scope For This SDK Package

The SDK should not absorb every possible product feature. To keep the package
clear and deployable, the following are not part of the core package today:

- Account-based infrastructure.
- Required network calls during normal local memory use.
- Heavy media generation dependencies in the install path.
- Generated media artifacts in the repository or distribution bundle.
- Service-side logic outside the local package boundary.

## How To Help

Start with a small issue, open a branch, and submit a Pull Request. The best
first contributions are focused docs fixes, missing tests, example improvements,
adapter hardening, and small runtime bugs with a clear reproduction.

See [CONTRIBUTING.md](CONTRIBUTING.md) for the branch and Pull Request flow.
