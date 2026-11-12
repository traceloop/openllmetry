import os

from opentelemetry.instrumentation.anthropic import AnthropicInstrumentor

ENV = "ANTHROPIC_OPEN_TELEMETRY"


def test_instrument_disables_native_sdk_tracing(monkeypatch):
    """anthropic >= 1.14 would otherwise emit a second span with the same usage."""
    monkeypatch.setenv(ENV, "true")
    monkeypatch.delenv(ENV, raising=False)
    instrumentor = AnthropicInstrumentor()
    instrumentor.instrument()
    try:
        assert os.environ[ENV] == "false"
    finally:
        instrumentor.uninstrument()


def test_uninstrument_removes_the_default_it_wrote(monkeypatch):
    """A client built after uninstrument() gets the SDK's native tracing again."""
    monkeypatch.setenv(ENV, "true")
    monkeypatch.delenv(ENV, raising=False)
    instrumentor = AnthropicInstrumentor()
    instrumentor.instrument()
    instrumentor.uninstrument()
    assert ENV not in os.environ


def test_instrument_keeps_explicit_native_tracing_setting(monkeypatch):
    monkeypatch.setenv(ENV, "true")
    instrumentor = AnthropicInstrumentor()
    instrumentor.instrument()
    try:
        assert os.environ[ENV] == "true"
    finally:
        instrumentor.uninstrument()
        assert os.environ[ENV] == "true"


def test_an_explicit_false_is_not_removed_by_uninstrument(monkeypatch):
    """The variable can come from the environment (opentelemetry-instrument)."""
    monkeypatch.setenv(ENV, "false")
    instrumentor = AnthropicInstrumentor()
    instrumentor.instrument()
    instrumentor.uninstrument()
    assert os.environ[ENV] == "false"


def test_ownership_survives_a_second_construction(monkeypatch):
    """BaseInstrumentor returns a singleton, so __init__ runs again on the same object."""
    monkeypatch.setenv(ENV, "true")
    monkeypatch.delenv(ENV, raising=False)
    AnthropicInstrumentor().instrument()
    AnthropicInstrumentor().uninstrument()
    assert ENV not in os.environ
