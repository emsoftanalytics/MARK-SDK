# Versioning

`mark-sdk` uses package-level Semantic Versioning. The PyPI distribution is
`mark-sdk`; the import name is `mark`.

## Source of truth

- Version: `src/mark/_version.py` (`__version__`). `pyproject.toml` reads it
  through Hatch (`dynamic = ["version"]`), so there is one place to change.
- Runtime: `mark.__version__`.
- Git tags: `vX.Y.Z` or `vX.Y.ZaN` for pre-releases, for example `v0.2.0a10`.
  The release workflow fails if the tag does not match `__version__`.
- Release notes: `CHANGELOG.md`.

## Version policy

- **Patch:** bug fixes, security fixes, documentation corrections, and
  compatible internal improvements.
- **Minor:** compatible public API additions, new local features, new optional
  integrations.
- **Major:** breaking public API changes after `1.0.0`.

Before `1.0.0`, minor versions and alpha releases may still change APIs.
Every breaking change is listed under "Breaking changes" in the changelog.

## Milestones

- `0.1.x`: local memory, context, policies, skills, agent wrapper, examples.
- `0.2.0aN`: alpha checkpoints for the local runtime: hybrid retrieval, facts,
  briefings, the CLI, MCP and LangChain integrations. The current release is
  `0.2.0a10`, the last alpha.
- `1.0.0` (next): the stable local SDK API with documented compatibility.
  The memory verbs, result rendering, CLI commands and MCP tool names are
  frozen, and deprecated aliases get a published removal schedule.

## Release checklist

1. Set `__version__` in `src/mark/_version.py`.
2. Move the changelog entries under a heading for the new version and date.
3. Run the tests:

   ```bash
   uv run --extra dev pytest
   ```

4. Commit, then tag the release with the same version:

   ```bash
   git tag v0.2.0a10
   ```

5. Build and inspect the artifacts:

   ```bash
   uv build
   python -m twine check dist/*
   ```

6. Push the tag. GitHub Actions tests, builds, and publishes to PyPI with
   Trusted Publishing. Do not store a long-lived PyPI token in the repository.
