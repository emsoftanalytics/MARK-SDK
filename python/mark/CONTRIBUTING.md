# Contributing to MARK SDK

Thanks for your interest in making agent memory better. Contributions of all
sizes are welcome — bug reports, docs fixes, tests, and features.

## Quick start

```bash
git clone https://github.com/emsoftanalytics/MARK-SDK.git
cd MARK-SDK
uv sync --extra dev          # or: pip install -e ".[dev]"
uv run pytest                # full suite, no network needed
```

The test suite runs entirely offline — if a test you write needs network or a
provider key, gate it behind an optional extra and skip when unavailable.

## What we're looking for

- **Bug fixes** — especially around retrieval, sessions, blocks, and migration.
- **Tests** — coverage for edge cases you hit in real agent workloads.
- **Docs** — README/notebook improvements, clearer docstrings, typo fixes.
- **Adapters** — completing and hardening framework adapters under
  `mark.adapters` (thin glue over public APIs only).
- **Performance** — faster retrieval/indexing on large local stores, with
  benchmarks.

Issues labeled [`good first issue`](https://github.com/emsoftanalytics/MARK-SDK/labels/good%20first%20issue)
and [`help wanted`](https://github.com/emsoftanalytics/MARK-SDK/labels/help%20wanted)
are the best places to start.

## Ground rules

- This package is local-first: the core must work without network, Docker,
  accounts, or API keys. New required dependencies are almost never accepted —
  use optional extras.
- Public functions, classes, and modules carry docstrings.
- Every behavior change comes with a test. `uv run pytest` must pass.
- Schema changes need a migration entry (`LocalMemoryStore._MIGRATIONS`) and a
  migration test.
- Match the style of the surrounding code; no sweeping reformat-only PRs.

## Pull requests

1. Fork, create a topic branch, make your change.
2. Run `uv run pytest` and `uv build`.
3. Open a PR with: what changed, why, and how you tested it. Link the issue it
   fixes if there is one.
4. Expect a review within a few days. Small, focused PRs get merged fastest.

First-time contributors: feel free to open a draft PR early and ask questions —
we'd rather help you land it than have you stuck.

## Reporting bugs and requesting features

Use the issue templates — they take about two minutes and give us what we need
to act fast. If you're not sure whether something is a bug, open an issue
anyway; "this behavior surprised me" reports regularly uncover real problems.

For security issues, please do **not** open a public issue — see
[SECURITY.md](SECURITY.md).

## License

By contributing you agree your contributions are licensed under the MIT
license that covers this package.
