import sys


def test_import():
    import opentelemetry.instrumentation.replicate  # noqa: F401

    assert "replicate" not in sys.modules
