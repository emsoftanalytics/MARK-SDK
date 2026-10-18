# Contributing to MARK SDK

Contributions of all sizes are welcome: bug reports, documentation fixes,
tests, and features.

## Quick start

```bash
git clone https://github.com/emsoftanalytics/MARK-SDK.git
cd MARK-SDK
uv sync --extra dev                # or: pip install -e ".[dev]"
uv run --extra dev pytest          # full suite, offline
```

The test suite runs offline. If a test needs a network service, a model, or
an optional dependency, gate it behind the matching extra and skip it when the
dependency is unavailable.

Adapter tests need their extras:

```bash
uv run --extra dev --extra adapters pytest tests/test_langchain_middleware.py tests/test_adapters_mcp_tools.py
```

When several pytest sessions share a machine (parallel branches, worktrees,
or agents), give each its own temporary directory so they never clean up each
other's files:

```bash
uv run --extra dev pytest -p no:cacheprovider --basetemp="$(mktemp -d)/pytest"
```

The scale benchmark checks speed and recall quality against the budgets in
`benchmarks/scale/thresholds.json`; run it before and after any change to the
store, the index, or the retrieval pipeline (CI runs the same check). The
CPU-time budgets (cold start, recall right after a write) gate the check;
wall-clock budgets only warn, because they swing several-fold with machine
load:

```bash
uv run --extra fast python benchmarks/scale/bench.py --n 20000 --check
```

[ARCHITECTURE.md](ARCHITECTURE.md) has a module map and the write, recall, and
briefing data flows. Read it before changing the store or the retrieval
pipeline.

## What we are looking for

- **Bug fixes**, especially in retrieval, facts, sessions, blocks, and
  migrations.
- **Tests** for edge cases from real agent workloads.
- **Documentation**: clearer examples, docstrings, and troubleshooting.
- **Adapters**: thin, tested glue over the public API under `mark.adapters`.
- **Performance** improvements with a benchmark that shows the gain.

Issues labelled
[`good first issue`](https://github.com/emsoftanalytics/MARK-SDK/labels/good%20first%20issue)
and
[`help wanted`](https://github.com/emsoftanalytics/MARK-SDK/labels/help%20wanted)
are good places to start.

## House rules

### Code

- **Local first.** The core must work without network, Docker, accounts, or
  API keys. New required dependencies are almost never accepted; use an
  optional extra.
- **Small files.** Every Python file is at most 300 lines. Split large modules
  by responsibility (see the mixins in `mark/store/_sqlite/` and
  `mark/memory/facade/`), and keep existing import paths working when you
  split one. New store methods go in a new mixin module registered in
  `store/sqlite.py`.
- **API naming.** Public verbs are plain and synchronous (`remember`,
  `recall`, `forget`); the async twin has the same name with an `a` prefix
  (`arecall`) and the same arguments. A verb that exists on `mark.memory`
  exists on `runtime.memory(agent_id)` with the same defaults, and results a
  model may read offer `as_context()`, `as_text()` and `to_dict()`. Renamed
  public names keep working as aliases (with a `DeprecationWarning` once
  callers have had a release to migrate). New top-level names in `mark` are
  rare; most belong in a submodule.
- **Module docstrings.** Every module starts with a docstring that says what it
  owns and, where useful, how it fits with its neighbours. Public functions and
  classes have docstrings. Comment the reason for non-obvious code.
- **Cold imports.** Every module must import on its own in a fresh interpreter.
  `tests/test_cold_imports.py` checks this; do not hide a circular import
  behind import order. `import mark` loads public names lazily, so keep heavy
  imports inside the functions that need them.
- **Schema changes** need a migration in `mark/store/_sqlite/ddl.py`
  (`_MIGRATIONS` or `_COLUMN_MIGRATIONS`), a bump of `CURRENT_SCHEMA_VERSION`,
  and a test that upgrades an older store.
- Match the style of the surrounding code. Do not submit reformat-only changes.
- Never commit generated media, local output folders, secrets, provider keys,
  or machine-specific configuration.

### Tests

- Every behaviour change has a test, including bug fixes (a test that fails
  before the fix).
- `uv run --extra dev pytest` must pass before a pull request is opened.
- Tests must not depend on a model, network, or wall-clock timing unless they
  are skipped when it is unavailable.

### Changelog

Add an entry to `CHANGELOG.md` under the upcoming version for every
user-visible change.

- Write in plain, professional language for SDK users.
- Record features (`Added`, `Changed`, `Breaking changes`) and fixes (`Fixed`).
- Describe what changed for the user, not how the code was refactored.
- Do not record formatting, internal renames, or test-only changes.

## Contribution flow

Do not push directly to `main`.

1. Fork the repository or create a branch from `main`.
2. Make the change on that branch.
3. Run `uv run --extra dev pytest` and `uv build`.
4. Open a pull request against `main` that says what changed, why, and how it
   was tested. Link the issue it fixes.
5. Respond to review. A maintainer merges the pull request when it is ready.

Small, focused pull requests are reviewed fastest. Draft pull requests and
questions are welcome.

## Reporting bugs and requesting features

Use the issue templates. If you are not sure whether something is a bug, open
an issue anyway.

For security issues, do not open a public issue. See
[SECURITY.md](SECURITY.md).

## License

By contributing, you agree that your contributions are licensed under the
Apache-2.0 license that covers this package.
