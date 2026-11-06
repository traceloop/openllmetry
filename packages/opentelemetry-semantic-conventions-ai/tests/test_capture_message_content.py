"""Tests for the shared content capture switch."""

import pytest
from opentelemetry import context

from opentelemetry.semconv_ai.utils import should_send_prompts

OTEL_VAR = "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT"
TRACELOOP_VAR = "TRACELOOP_TRACE_CONTENT"


@pytest.fixture(autouse=True)
def capture_env(monkeypatch):
    """Start every test from a clean environment."""
    monkeypatch.delenv(OTEL_VAR, raising=False)
    monkeypatch.delenv(TRACELOOP_VAR, raising=False)


def test_default_is_enabled():
    """With neither variable set, capture stays on, as it was before."""
    assert should_send_prompts() is True


def test_traceloop_switch_still_disables_capture(monkeypatch):
    """The existing Traceloop switch keeps working."""
    monkeypatch.setenv(TRACELOOP_VAR, "false")
    assert should_send_prompts() is False


def test_otel_switch_disables_capture(monkeypatch):
    """The standard OTel variable is honored; it used to be ignored."""
    monkeypatch.setenv(OTEL_VAR, "false")
    assert should_send_prompts() is False


@pytest.mark.parametrize("value", ["SPAN_ONLY", "EVENT_ONLY", "SPAN_AND_EVENT"])
def test_otel_capture_values_enable_capture(monkeypatch, value):
    """The OTel convention's capture values keep content on."""
    monkeypatch.setenv(OTEL_VAR, value)
    assert should_send_prompts() is True


def test_otel_no_content_disables_capture(monkeypatch):
    """NO_CONTENT is the OTel convention's explicit off value."""
    monkeypatch.setenv(OTEL_VAR, "NO_CONTENT")
    assert should_send_prompts() is False


def test_otel_switch_takes_precedence(monkeypatch):
    """An explicit OTel setting wins over the Traceloop fallback."""
    monkeypatch.setenv(TRACELOOP_VAR, "true")
    monkeypatch.setenv(OTEL_VAR, "false")
    assert should_send_prompts() is False

    monkeypatch.setenv(TRACELOOP_VAR, "false")
    monkeypatch.setenv(OTEL_VAR, "true")
    assert should_send_prompts() is True


def test_context_override_still_enables_capture(monkeypatch):
    """The SDK's per-association allow list can still enable capture."""
    monkeypatch.setenv(OTEL_VAR, "false")
    token = context.attach(context.set_value("override_enable_content_tracing", True))
    try:
        assert should_send_prompts() is True
    finally:
        context.detach(token)
