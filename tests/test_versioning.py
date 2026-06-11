try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib  # type: ignore[no-redef]
from pathlib import Path

import mark


def test_runtime_version_matches_project_metadata():
    pyproject = Path(__file__).parents[1] / "pyproject.toml"
    metadata = tomllib.loads(pyproject.read_text(encoding="utf-8"))

    assert mark.__version__ == metadata["project"]["version"]
