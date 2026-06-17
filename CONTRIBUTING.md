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

## Contribution flow

Please do not push directly to `main`.

All contributions should be made from a separate branch and submitted through a
Pull Request. This keeps the SDK stable, allows tests and packaging checks to
run, and gives maintainers a chance to review changes before they are merged.

Recommended flow:

1. Fork the repository or create a feature branch from `main`.
2. Make your changes in that branch.
3. Run the relevant checks locally.
4. Open a Pull Request against `main`.
5. Respond to review feedback.
6. A maintainer will merge the PR when it is ready.

Maintainers review contributions before they land on `main`. This is how MARK
protects release quality while still making it easy for new contributors to
help.

## Ground rules

- This package is local-first: the core must work without network, Docker,
  accounts, or API keys. New required dependencies are almost never accepted —
  use optional extras.
- Public functions, classes, and modules carry docstrings.
- Every behavior change comes with a test. `uv run pytest` must pass.
- Schema changes need a migration entry (`LocalMemoryStore._MIGRATIONS`) and a
  migration test.
- Match the style of the surrounding code; no sweeping reformat-only PRs.
- Keep changes scoped to the SDK/runtime unless an issue explicitly asks for
  something else.
- Do not commit generated media, local output folders, secrets, provider keys,
  or machine-specific configuration.

## Pull requests

1. Fork or branch from `main`, then make your change on that branch.
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

By contributing you agree your contributions are licensed under the Apache-2.0
license that covers this package.
