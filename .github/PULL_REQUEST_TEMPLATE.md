## What does this PR do?

<!-- One or two sentences. Link the issue it fixes: Fixes #123 -->

## How was it tested?

- [ ] `uv run --extra dev pytest` passes locally
- [ ] Adapter changes were tested with `uv run --extra dev --extra adapters --extra langchain --extra mcp pytest tests/test_langchain_middleware.py tests/test_adapters_mcp_tools.py`
- [ ] New behavior is covered by a test
- [ ] Docstrings updated where behavior changed

## Checklist

- [ ] No new required dependencies or network calls
- [ ] Example assets and generated outputs are not included in the package
- [ ] No heavy media dependencies such as `gradio-client` or `pillow` were added to package extras
- [ ] Schema changes include a `_MIGRATIONS` entry and a migration test
- [ ] No secrets, keys, or private data in the diff
