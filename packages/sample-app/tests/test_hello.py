"""Hello unit test module."""

from sample_app.hello import hello


def test_hello():
    """Test the hello function."""
    assert hello() == "Hello sample-app"
