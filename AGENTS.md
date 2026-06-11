# MARK Python SDK Agent Guide

This package implements the local-first MARK SDK. When working in this package, treat MARK as the agent cognitive runtime: local memory, context assembly, policies, skills, and agent wrapping.

## Product Boundary

- `mark` is local-first and must work without network, Docker, cloud accounts, or hosted APIs.
- `mark-cloud` owns HTTP/cloud sync, auth, tenancy, and hosted control-plane behavior.
- `mark-adapters` owns LangChain, MCP, LangGraph, and framework-specific integrations.
- Do not turn `mark` into a presentation app or product UI.

## Agent Usage Pattern

When demonstrating MARK usage, show the same task without MARK and with MARK.

When MARK is active, treat it as persistent working memory, not an optional lookup step. The intended loop is:

- recall from MARK before planning,
- ground memory against current evidence,
- act with remembered context,
- reflect on durable outcomes,
- remember verified facts, decisions, and procedures for future agents.

Use MARK to improve:

- recall: stored project facts are retrieved,
- planning: steps are grounded in project context,
- reasoning: decisions cite known constraints,
- continuity: durable outcomes are written back.

## SDK Integration Modes

Agents should be able to use MARK as:

- direct runtime: `Mark.local()` and `mark.wrap_agent(...)`,
- tool: callable retrieval/write functions,
- skill: `src/mark/skills/mark-usage/SKILL.md`,
- MCP: protocol server exposing retrieve/write/list operations,
- middleware: context retrieval before model/tool execution and memory writeback after accepted outcomes.

Middleware and direct runtime examples should make MARK feel natural: the agent should receive context before it reasons and save durable outcomes after the task is accepted.

## Development Rules

- Keep local SDK dependencies minimal.
- Add tests with every behavioral change.
- Do not require LangChain, MCP, or cloud packages in the local SDK core.
- Optional examples may mention LangChain, but local tests must run without provider API keys.
- Do not store secrets in examples or memory fixtures except fake redaction test values.
- Package non-Python skill assets explicitly so installed users can load them through `importlib.resources`.

## Verification

From `sdk/python/mark`, run:

```bash
uv run --extra dev pytest
uv build
```

Before committing, verify the package contains the MARK usage skill:

```bash
uv run python -B -c "from importlib.resources import files; print(files('mark.skills').joinpath('mark-usage/SKILL.md').read_text().splitlines()[1])"
```
