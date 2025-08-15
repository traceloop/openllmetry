"""Test that both legacy and new Google GenAI libraries can be imported and instrumented."""

from opentelemetry.instrumentation.google_generativeai import (
    GoogleGenerativeAiInstrumentor,
)
from opentelemetry.instrumentation.google_generativeai.utils import is_package_installed


def test_legacy_library_detection():
    """Test that the legacy library can be detected."""
    has_legacy = is_package_installed("google.generativeai")
    assert has_legacy, "google.generativeai should be installed"


def test_new_library_detection():
    """Test that the new library can be detected."""
    has_new = is_package_installed("google.genai")
    assert has_new, "google.genai should be installed"


def test_instrumentor_initialization():
    """Test that the instrumentor can be initialized correctly."""
    instrumentor = GoogleGenerativeAiInstrumentor()

    # Should have appropriate dependencies
    deps = instrumentor.instrumentation_dependencies()
    assert len(deps) > 0, "Should have instrumentation dependencies"

    # Should prefer new library over legacy
    if is_package_installed("google.genai"):
        assert "google-genai" in deps[0]
    else:
        assert "google-generativeai" in deps[0]


def test_wrapped_methods_selection():
    """Test that the correct wrapped methods are selected based on available library."""
    instrumentor = GoogleGenerativeAiInstrumentor()
    methods = instrumentor._wrapped_methods()

    assert len(methods) > 0, "Should have wrapped methods"

    # Check that the right package is being wrapped
    for method in methods:
        package = method.get("package", "")
        if is_package_installed("google.genai"):
            assert "google.genai" in package
        else:
            assert "google.generativeai" in package
