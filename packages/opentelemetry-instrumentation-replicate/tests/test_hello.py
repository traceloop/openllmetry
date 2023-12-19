"""Hello unit test module."""

from opentelemetry.instrumentation.replicate.hello import hello


def test_hello():
    """Test the hello function."""
    assert hello() == "Hello opentelemetry-instrumentation-replicate"
