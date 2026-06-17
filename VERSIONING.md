# Versioning

`mark-sdk` uses package-level Semantic Versioning.

The PyPI distribution name is `mark-sdk`; the Python import name remains
`mark`.

## Source of Truth

- Package metadata: `pyproject.toml`
- Runtime export: `mark.__version__`
- Git release tags: `vX.Y.Z`

The package version and `mark.__version__` must stay identical.

## Version Policy

- Patch: bug fixes, security fixes, documentation corrections, and compatible internal improvements.
- Minor: compatible public API additions, new local runtime features, new optional integrations, or expanded examples.
- Major: breaking public API changes after `1.0.0`.

Before `1.0.0`, minor versions may still refine APIs, but every breaking change must be called out in release notes.

## Planned Milestones

- `0.1.x`: local memory, context, policies, skills, agent wrapper, examples.
- `0.2.0aN`: pre-release checkpoints for the stronger local hippocampus/cortex runtime.
- `0.2.x`: stable release line for SQLite memory, vector retrieval, graph expansion,
  plugin boundaries, and stronger local SDK documentation.
- `0.3.x`: stable agent middleware interfaces and framework adapters.
- `0.4.x`: hardened sync-envelope contracts and adapter compatibility.
- `1.0.0`: stable local SDK API, documented compatibility, CI, packaging, and security review.

## Release Checklist

1. Update `pyproject.toml`.
2. Update `src/mark/_version.py`.
3. Run `uv run --extra dev pytest`.
4. Commit the version change.
5. Tag the release:

```bash
git tag v0.2.0a6
```

6. Build and inspect the release artifacts:

```bash
uv build
python -m twine check dist/*
```

7. Publish from GitHub Actions using PyPI Trusted Publishing. Do not store a
long-lived PyPI token in the repository.
