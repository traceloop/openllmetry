"""Regression tests for the standard OTel content capture switch."""

from opentelemetry.instrumentation.openai.utils import should_send_prompts

OTEL_VAR = "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT"
TRACELOOP_VAR = "TRACELOOP_TRACE_CONTENT"


def test_otel_switch_disables_capture(monkeypatch):
    """The OpenAI instrumentation used to ignore this variable."""
    monkeypatch.delenv(TRACELOOP_VAR, raising=False)
    monkeypatch.setenv(OTEL_VAR, "false")
    assert should_send_prompts() is False


def test_default_is_unchanged(monkeypatch):
    """With neither variable set, content capture stays on."""
    monkeypatch.delenv(OTEL_VAR, raising=False)
    monkeypatch.delenv(TRACELOOP_VAR, raising=False)
    assert should_send_prompts() is True


def test_traceloop_switch_still_disables_capture(monkeypatch):
    """Existing behavior is preserved when the OTel variable is unset."""
    monkeypatch.delenv(OTEL_VAR, raising=False)
    monkeypatch.setenv(TRACELOOP_VAR, "false")
    assert should_send_prompts() is False
