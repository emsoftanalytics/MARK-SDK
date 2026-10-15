try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib  # type: ignore[no-redef]
from pathlib import Path

import mark

ROOT = Path(__file__).parents[1]


def test_version_is_single_sourced_from_version_module():
    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert "version" in metadata["project"]["dynamic"]
    assert metadata["tool"]["hatch"]["version"]["path"] == "src/mark/_version.py"
    assert mark.__version__ == mark._version.__version__


def test_import_resolves_to_regular_package_not_namespace():
    # A stray site-packages/mark/ directory (e.g. from hatch force-include in an
    # editable install) turns `mark` into an empty namespace package.
    assert mark.__file__ is not None
    assert Path(mark.__file__).name == "__init__.py"
    assert hasattr(mark, "Mark")


def test_wheel_config_has_no_force_include():
    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    wheel = metadata["tool"]["hatch"]["build"]["targets"]["wheel"]

    assert "force-include" not in wheel
