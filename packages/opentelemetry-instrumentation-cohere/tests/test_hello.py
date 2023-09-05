"""Hello unit test module."""

from opentelemetry_instrumentation_cohere.hello import hello


def test_hello():
    """Test the hello function."""
    assert hello() == "Hello opentelemetry-instrumentation-cohere"
