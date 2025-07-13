import sys

def test_import():
    import opentelemetry.instrumentation.replicate

    assert "replicate" not in sys.modules
