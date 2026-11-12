import os

import pytest
from opentelemetry.instrumentation.anthropic import AnthropicInstrumentor

MESSAGE = {
    "id": "msg_1",
    "type": "message",
    "role": "assistant",
    "model": "claude-sonnet-5-5",
    "content": [{"type": "text", "text": "hi"}],
    "stop_reason": "end_turn",
    "stop_sequence": None,
    "usage": {"input_tokens": 1000, "output_tokens": 200},
}


def test_instrument_disables_native_sdk_tracing(monkeypatch):
    """anthropic >= 1.14 would otherwise emit a second span with the same usage."""
    monkeypatch.delenv("ANTHROPIC_OPEN_TELEMETRY", raising=False)
    instrumentor = AnthropicInstrumentor()
    instrumentor.instrument()
    try:
        assert os.environ["ANTHROPIC_OPEN_TELEMETRY"] == "false"
    finally:
        instrumentor.uninstrument()


def test_instrument_keeps_explicit_native_tracing_setting(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_OPEN_TELEMETRY", "true")
    instrumentor = AnthropicInstrumentor()
    instrumentor.instrument()
    try:
        assert os.environ["ANTHROPIC_OPEN_TELEMETRY"] == "true"
    finally:
        instrumentor.uninstrument()
