"""gen_ai.tool.definitions is recorded in both attributes and events mode, and only
when content tracing is on."""

import json
from unittest.mock import patch

import pytest
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)

from .helpers import MODEL

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the weather for a city",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
    },
}


def _chat_with_tools(caller):
    # The request is stopped before the network: tool definitions are recorded
    # from the request arguments, before the provider call.
    with patch(caller.request_patch_target(), side_effect=Exception("stop before the network")):
        with pytest.raises(Exception, match="stop before the network"):
            caller.chat(
                model=MODEL,
                messages=[{"role": "user", "content": "What's the weather in Paris?"}],
                tools=[WEATHER_TOOL],
            )


@pytest.mark.parametrize("instrumentation", ["instrument_legacy", "instrument_with_content"])
def test_tool_definitions_are_recorded(request, instrumentation, caller, span_exporter):
    request.getfixturevalue(instrumentation)

    _chat_with_tools(caller)

    (span,) = span_exporter.get_finished_spans()
    assert json.loads(span.attributes[GenAIAttributes.GEN_AI_TOOL_DEFINITIONS]) == [WEATHER_TOOL]


def test_tool_definitions_are_not_recorded_without_content(instrument_with_no_content, caller, span_exporter):
    _chat_with_tools(caller)

    (span,) = span_exporter.get_finished_spans()
    assert GenAIAttributes.GEN_AI_TOOL_DEFINITIONS not in span.attributes
