"""Test that the google-genai library instrumentation works."""

from opentelemetry.instrumentation.google_generativeai import (
    GoogleGenerativeAiInstrumentor,
)


def test_library_instrumentation():
    """Test that the google-genai library gets properly instrumented."""
    # Import the library
    from google import genai
    from google.genai.models import AsyncModels, Models

    # Set up instrumentor
    instrumentor = GoogleGenerativeAiInstrumentor()

    # Verify methods are not wrapped initially
    assert not hasattr(Models.generate_content, '__wrapped__')
    assert not hasattr(Models.generate_content_stream, '__wrapped__')
    assert not hasattr(AsyncModels.generate_content, '__wrapped__')
    assert not hasattr(AsyncModels.generate_content_stream, '__wrapped__')

    try:
        instrumentor.instrument()

        # Verify all methods are now wrapped
        assert hasattr(Models.generate_content, '__wrapped__')
        assert hasattr(Models.generate_content_stream, '__wrapped__')
        assert hasattr(AsyncModels.generate_content, '__wrapped__')
        assert hasattr(AsyncModels.generate_content_stream, '__wrapped__')

        # Verify they're callable
        assert callable(Models.generate_content)
        assert callable(Models.generate_content_stream)
        assert callable(AsyncModels.generate_content)
        assert callable(AsyncModels.generate_content_stream)

        # Test that we can create a client
        client = genai.Client(api_key="test_key")
        assert client is not None
        assert hasattr(client, 'models')
        assert isinstance(client.models, Models)

    finally:
        instrumentor.uninstrument()

        # Verify methods are unwrapped
        assert not hasattr(Models.generate_content, '__wrapped__')
        assert not hasattr(Models.generate_content_stream, '__wrapped__')
        assert not hasattr(AsyncModels.generate_content, '__wrapped__')
        assert not hasattr(AsyncModels.generate_content_stream, '__wrapped__')


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
