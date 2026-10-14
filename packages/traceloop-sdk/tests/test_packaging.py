import importlib.metadata as metadata

from packaging.requirements import Requirement


def _declared_dependency_names() -> set[str]:
    requires = metadata.requires("traceloop-sdk") or []
    return {
        req.name
        for req in map(Requirement, requires)
        if req.marker is None or req.marker.evaluate({"extra": ""})
    }


def test_requests_is_declared_as_a_runtime_dependency():
    assert "requests" in _declared_dependency_names()


def test_httpx_is_declared_as_a_runtime_dependency():
    assert "httpx" in _declared_dependency_names()


def test_extra_only_requirements_are_not_counted_as_runtime():
    assert "pandas" not in _declared_dependency_names()
