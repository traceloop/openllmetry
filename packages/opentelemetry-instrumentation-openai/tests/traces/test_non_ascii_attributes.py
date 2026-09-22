"""Regression test for issue #4426.

Non-ASCII characters in the content written to gen_ai.* JSON span attributes
were escaped to \\uXXXX sequences because json.dumps defaults to
ensure_ascii=True. The instrumentation now passes ensure_ascii=False so the
attribute preserves the original UTF-8 text.
"""
import json

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)

from opentelemetry.instrumentation.openai.shared import _set_tool_definitions_json


def _span():
    return TracerProvider().get_tracer(__name__).start_span("test")


def test_tool_definitions_preserve_non_ascii():
    span = _span()
    tool_defs = [
        {
            "name": "get_current_weather",
            "description": "Узнать погоду",  # Russian
            "parameters": {"city": "Бостон"},
        }
    ]
    _set_tool_definitions_json(span, tool_defs)

    raw = dict(span.attributes)[GenAIAttributes.GEN_AI_TOOL_DEFINITIONS]
    # The stored JSON must contain the original UTF-8 text, not \uXXXX escapes.
    assert "Узнать погоду" in raw
    assert "\\u" not in raw
    # And it must still round-trip to the same structure.
    assert json.loads(raw) == tool_defs


def test_tool_definitions_ascii_unchanged():
    span = _span()
    tool_defs = [{"name": "ping", "description": "check health"}]
    _set_tool_definitions_json(span, tool_defs)
    raw = dict(span.attributes)[GenAIAttributes.GEN_AI_TOOL_DEFINITIONS]
    assert json.loads(raw) == tool_defs
