import json
from unittest.mock import Mock, patch

from opentelemetry.instrumentation.langchain.span_utils import (
    set_chat_request,
    set_request_params,
)
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)


def _span_attributes(span: Mock) -> dict:
    return {call.args[0]: call.args[1] for call in span.set_attribute.call_args_list}


def test_tool_definitions_preserve_source_type():
    span = Mock()
    span.is_recording.return_value = True
    span_holder = Mock()
    kwargs = {
        "invocation_params": {
            "tools": [
                {
                    "type": "custom_tool",
                    "function": {
                        "name": "get_weather",
                        "description": "Get the weather",
                        "parameters": {"type": "object"},
                    },
                }
            ]
        }
    }

    set_request_params(span, kwargs, span_holder)

    tool_definitions = json.loads(_span_attributes(span)[GenAIAttributes.GEN_AI_TOOL_DEFINITIONS])
    assert tool_definitions == [
        {
            "type": "custom_tool",
            "name": "get_weather",
            "description": "Get the weather",
            "parameters": {"type": "object"},
        }
    ]


def test_tool_definitions_default_to_function_type():
    span = Mock()
    span.is_recording.return_value = True
    span_holder = Mock()
    kwargs = {
        "invocation_params": {
            "tools": [
                {
                    "name": "get_weather",
                    "description": "Get the weather",
                    "input_schema": {"type": "object"},
                }
            ]
        }
    }

    set_request_params(span, kwargs, span_holder)

    tool_definitions = json.loads(_span_attributes(span)[GenAIAttributes.GEN_AI_TOOL_DEFINITIONS])
    assert tool_definitions[0]["type"] == "function"


def test_legacy_function_definitions_include_function_type():
    span = Mock()
    span.is_recording.return_value = True
    span_holder = Mock()
    kwargs = {
        "invocation_params": {
            "functions": [
                {
                    "name": "get_weather",
                    "description": "Get the weather",
                    "parameters": {"type": "object"},
                }
            ]
        }
    }

    with patch(
        "opentelemetry.instrumentation.langchain.span_utils.should_send_prompts",
        return_value=True,
    ):
        set_chat_request(span, {}, [], kwargs, span_holder)

    tool_definitions = json.loads(_span_attributes(span)[GenAIAttributes.GEN_AI_TOOL_DEFINITIONS])
    assert tool_definitions[0]["type"] == "function"
