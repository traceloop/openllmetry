"""Test that the google-genai library instrumentation works."""
import wrapt
from opentelemetry.instrumentation.google_generativeai import (
    GoogleGenerativeAiInstrumentor,
)
from google.genai.models import Models, AsyncModels


def _is_instrumented(func):
    """
    OpenTelemetry instrumentations wrap functions using wrapt.
    Presence of __wrapped__ or a wrapt wrapper means instrumented.
    """
    return hasattr(func, "__wrapped__") or isinstance(
        func,
        (wrapt.FunctionWrapper, wrapt.BoundFunctionWrapper),
    )


def test_google_genai_instrumentation_lifecycle():
    """Validate instrumentation, idempotency, and cleanup."""

    instrumentor = GoogleGenerativeAiInstrumentor()

    # --- ensure clean state ---
    instrumentor.uninstrument()

    assert not _is_instrumented(Models.generate_content)
    assert not _is_instrumented(Models.generate_content_stream)
    assert not _is_instrumented(AsyncModels.generate_content)
    assert not _is_instrumented(AsyncModels.generate_content_stream)

    # --- instrument ---
    instrumentor.instrument()

    assert _is_instrumented(Models.generate_content)
    assert _is_instrumented(Models.generate_content_stream)
    assert _is_instrumented(AsyncModels.generate_content)
    assert _is_instrumented(AsyncModels.generate_content_stream)

    # --- instrumentation is idempotent ---
    instrumentor.instrument()
    assert _is_instrumented(Models.generate_content)

    # --- uninstrument ---
    instrumentor.uninstrument()

    assert not _is_instrumented(Models.generate_content)
    assert not _is_instrumented(Models.generate_content_stream)
    assert not _is_instrumented(AsyncModels.generate_content)
    assert not _is_instrumented(AsyncModels.generate_content_stream)


def test_instrumentation_dependencies():
    """Test that the instrumentor has correct dependencies."""
    instrumentor = GoogleGenerativeAiInstrumentor()
    deps = instrumentor.instrumentation_dependencies()

    assert len(deps) == 1
    assert "google-genai >= 1.0.0" in deps[0]


def test_wrapped_methods():
    """Test that the correct methods are wrapped."""
    instrumentor = GoogleGenerativeAiInstrumentor()
    methods = instrumentor._wrapped_methods()

    assert len(methods) == 4

    # Should be using new library methods
    packages = [method.get("package", "") for method in methods]
    assert all("google.genai" in pkg for pkg in packages)

    # Should have both sync and async methods
    objects = [method.get("object", "") for method in methods]
    assert "Models" in objects
    assert "AsyncModels" in objects

    # Should wrap both generate_content and generate_content_stream
    wrapped_methods = [method.get("method", "") for method in methods]
    assert wrapped_methods.count("generate_content") == 2
    assert wrapped_methods.count("generate_content_stream") == 2
