## What does this PR do?

<!-- One or two sentences. Link the issue it fixes: Fixes #123 -->

## How was it tested?

- [ ] `uv run --extra dev pytest` passes locally
- [ ] Adapter changes: `uv run --extra dev --extra adapters pytest tests/test_langchain_*.py tests/test_adapters_mcp_tools.py` passes
- [ ] New or changed behaviour is covered by a test

## Checklist

- [ ] No new required dependencies or network calls in the core
- [ ] Python files stay at or below 300 lines and start with a module docstring
- [ ] Every new module imports on its own (`tests/test_cold_imports.py`)
- [ ] Schema changes include a migration and a migration test
- [ ] `CHANGELOG.md` has an entry for user-visible features or fixes
- [ ] Docs and docstrings updated where behaviour changed
- [ ] No generated media, heavy media dependencies, secrets, keys, or private data in the diff
