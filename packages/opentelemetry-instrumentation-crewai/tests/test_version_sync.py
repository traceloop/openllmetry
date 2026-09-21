"""Regression test for issue #4450.

`version.py` had drifted to 0.36.0 while the package released as 0.62.3, so
every span and metric reported the wrong instrumentation scope version. Keep the
in-tree `__version__` locked to the packaged version so it cannot silently
diverge again.
"""
import pathlib
import tomllib

from opentelemetry.instrumentation.crewai.version import __version__

_PYPROJECT = (
    pathlib.Path(__file__).resolve().parents[1] / "pyproject.toml"
)


def test_version_matches_pyproject():
    with _PYPROJECT.open("rb") as f:
        data = tomllib.load(f)
    packaged = data["project"]["version"]
    assert __version__ == packaged, (
        f"version.py ({__version__}) is out of sync with pyproject.toml "
        f"({packaged}); the instrumentation scope version would be wrong"
    )
