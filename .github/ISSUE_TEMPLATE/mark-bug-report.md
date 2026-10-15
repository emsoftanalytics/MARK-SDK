---
name: Mark Bug report
about: Report something in MARK that does not work as documented
title: ''
labels: ''
assignees: ''

---

**Describe the bug**
A clear description of what went wrong.

**To reproduce**
A minimal script or command that shows the problem, for example:

```python
from mark import Mark

with Mark.local(".") as mark:
    ...
```

**Expected behaviour**
What you expected to happen.

**Actual behaviour**
What happened instead. Include the full traceback or command output.

**Environment**
- MARK version (`python -c "import mark; print(mark.__version__)"`):
- Python version:
- OS:
- Extras installed (for example `fast`, `langchain`, `mcp`, `crypto`):
- Embedding provider, if not the default:
- Integration, if any (LangChain middleware, tools, MCP, `mark` CLI, `mark serve`):

**Additional context**
Anything else that helps, such as store size or whether the store is encrypted.
Do not include secrets or private memory content. Report security issues
privately (see SECURITY.md).
