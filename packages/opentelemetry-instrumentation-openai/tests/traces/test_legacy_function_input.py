"""Legacy function-calling messages must survive into gen_ai.input.messages."""

import json
from unittest.mock import MagicMock

from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)

from opentelemetry.instrumentation.openai.shared.chat_wrappers import (
    _set_input_messages,
)


def _recorded_messages(span):
    for call in reversed(span.set_attribute.call_args_list):
        if call.args[0] == GenAIAttributes.GEN_AI_INPUT_MESSAGES:
            return json.loads(call.args[1])
    raise AssertionError("gen_ai.input.messages was never set")


def test_legacy_function_call_and_function_role_are_recorded():
    span = MagicMock()
    messages = [
        {"role": "user", "content": "what is the weather in Sao Paulo?"},
        {
            "role": "assistant",
            "content": None,
            "function_call": {"name": "get_weather", "arguments": '{"city": "Sao Paulo"}'},
        },
        {"role": "function", "name": "get_weather", "content": '{"temp_c": 25}'},
    ]

    _set_input_messages(span, messages)

    recorded = _recorded_messages(span)
    assert len(recorded) == 3, recorded

    assistant = recorded[1]
    tool_call = [p for p in assistant["parts"] if p["type"] == "tool_call"]
    assert tool_call, assistant
    assert tool_call[0]["name"] == "get_weather"

    function_result = recorded[2]
    assert function_result["role"] == "tool", function_result
    response = [p for p in function_result["parts"] if p["type"] == "tool_call_response"]
    assert response and response[0]["response"] == '{"temp_c": 25}', function_result


def test_tuple_content_is_mapped_like_a_list():
    span = MagicMock()
    messages = [
        {
            "role": "user",
            "content": ({"type": "text", "text": "hello"},),
        },
    ]

    _set_input_messages(span, messages)

    recorded = _recorded_messages(span)
    assert recorded == [
        {"role": "user", "parts": [{"content": "hello", "type": "text"}]},
    ], recorded

