import re
from pathlib import Path

from opentelemetry.instrumentation.crewai.version import __version__

PYPROJECT_PATH = Path(__file__).parent.parent / "pyproject.toml"


def _released_version() -> str:
    text = PYPROJECT_PATH.read_text()
    match = re.search(r'(?m)^version = "([^"]+)"', text)
    assert match, f"could not find a 'version = \"...\"' line in {PYPROJECT_PATH}"
    return match.group(1)


def test_instrumentation_scope_version_matches_released_package_version():
    released_version = _released_version()
    assert __version__ == released_version, (
        f"opentelemetry.instrumentation.crewai.__version__ ({__version__!r}) must match "
        f"the package's released version ({released_version!r}); spans and metrics report "
        "this value as the instrumentation scope version."
    )
