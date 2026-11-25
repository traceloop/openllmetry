"""Unit tests for the request/response to OTel GenAI attribute mapping.

These cover message shapes the recorded cassettes don't (system prompts,
multimodal content, tool calls and tool results) without calling the API.
"""

import json
from enum import Enum
from importlib.metadata import version

import pytest
from opentelemetry.instrumentation.together.span_utils import (
    map_finish_reason,
    set_input_attributes,
    set_model_input_attributes,
    set_model_response_attributes,
    set_output_attributes,
)
from opentelemetry.instrumentation.together.utils import TRACELOOP_TRACE_CONTENT
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.semconv_ai import LLMRequestTypeValues, SpanAttributes

CHAT = LLMRequestTypeValues.CHAT
COMPLETION = LLMRequestTypeValues.COMPLETION

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the weather for a city",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
    },
}


class FakeSpan:
    def __init__(self):
        self.attributes = {}

    def is_recording(self):
        return True

    def set_attribute(self, name, value):
        self.attributes[name] = value


@pytest.fixture
def span():
    return FakeSpan()


@pytest.fixture
def content_enabled(monkeypatch):
    monkeypatch.setenv(TRACELOOP_TRACE_CONTENT, "true")


@pytest.fixture
def content_disabled(monkeypatch):
    monkeypatch.setenv(TRACELOOP_TRACE_CONTENT, "false")


def _json_attribute(span, name):
    return json.loads(span.attributes[name])


def test_chat_input_messages_cover_all_roles_and_parts(span, content_enabled):
    kwargs = {
        "messages": [
            {"role": "system", "content": "You are a weather bot."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is in this picture?"},
                    {"type": "image_url", "image_url": {"url": "https://example.com/cat.png"}},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
                ],
            },
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "Sunny"},
        ],
        "tools": [WEATHER_TOOL],
    }

    set_input_attributes(span, CHAT, kwargs)

    assert _json_attribute(span, GenAIAttributes.GEN_AI_INPUT_MESSAGES) == [
        {"role": "system", "parts": [{"type": "text", "content": "You are a weather bot."}]},
        {
            "role": "user",
            "parts": [
                {"type": "text", "content": "What is in this picture?"},
                {"type": "uri", "modality": "image", "uri": "https://example.com/cat.png"},
                {"type": "blob", "modality": "image", "mime_type": "image/png", "content": "AAAA"},
            ],
        },
        {
            "role": "assistant",
            "parts": [
                {"type": "tool_call", "name": "get_weather", "id": "call_1", "arguments": {"city": "Paris"}},
            ],
        },
        {"role": "tool", "parts": [{"type": "tool_call_response", "id": "call_1", "response": "Sunny"}]},
    ]
    assert _json_attribute(span, GenAIAttributes.GEN_AI_TOOL_DEFINITIONS) == [WEATHER_TOOL]


def test_completion_input_is_a_single_user_message(span, content_enabled):
    set_input_attributes(span, COMPLETION, {"prompt": "Once upon a time"})

    assert _json_attribute(span, GenAIAttributes.GEN_AI_INPUT_MESSAGES) == [
        {"role": "user", "parts": [{"type": "text", "content": "Once upon a time"}]}
    ]


def test_chat_output_messages_include_reasoning_and_tool_calls(span, content_enabled):
    response = {
        "choices": [
            {
                "index": 0,
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": "Let me check.",
                    "reasoning_content": "The user wants the weather.",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'},
                        }
                    ],
                },
            },
            {
                "index": 1,
                "finish_reason": "eos",
                "message": {"role": "assistant", "content": "It is sunny."},
            },
        ]
    }

    set_output_attributes(span, CHAT, response)

    assert _json_attribute(span, GenAIAttributes.GEN_AI_OUTPUT_MESSAGES) == [
        {
            "role": "assistant",
            "parts": [
                {"type": "reasoning", "content": "The user wants the weather."},
                {"type": "text", "content": "Let me check."},
                {"type": "tool_call", "name": "get_weather", "id": "call_1", "arguments": {"city": "Paris"}},
            ],
            "finish_reason": "tool_call",
        },
        {
            "role": "assistant",
            "parts": [{"type": "text", "content": "It is sunny."}],
            "finish_reason": "stop",
        },
    ]


def test_completion_output_messages(span, content_enabled):
    response = {"choices": [{"index": 0, "finish_reason": "length", "text": "there was a span."}]}

    set_output_attributes(span, COMPLETION, response)

    assert _json_attribute(span, GenAIAttributes.GEN_AI_OUTPUT_MESSAGES) == [
        {
            "role": "assistant",
            "parts": [{"type": "text", "content": "there was a span."}],
            "finish_reason": "length",
        }
    ]


def test_content_is_not_recorded_when_tracing_content_is_disabled(span, content_disabled):
    kwargs = {"model": "m", "messages": [{"role": "user", "content": "secret"}], "tools": [WEATHER_TOOL]}
    response = {
        "model": "m",
        "choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": "secret"}}],
    }

    set_model_input_attributes(span, kwargs)
    set_input_attributes(span, CHAT, kwargs)
    set_output_attributes(span, CHAT, response)
    set_model_response_attributes(span, response)

    assert GenAIAttributes.GEN_AI_INPUT_MESSAGES not in span.attributes
    assert GenAIAttributes.GEN_AI_OUTPUT_MESSAGES not in span.attributes
    assert GenAIAttributes.GEN_AI_TOOL_DEFINITIONS not in span.attributes
    assert "secret" not in json.dumps(span.attributes)
    # Non-content attributes are still recorded.
    assert span.attributes[GenAIAttributes.GEN_AI_RESPONSE_MODEL] == "m"
    assert span.attributes[GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS] == ["stop"]


def test_request_parameters(span):
    set_model_input_attributes(
        span,
        {
            "model": "m",
            "max_tokens": 100,
            "temperature": 0.5,
            "top_p": 0.9,
            "top_k": 40,
            "frequency_penalty": 0.1,
            "presence_penalty": 0.2,
            "repetition_penalty": 1.1,
            "seed": 7,
            "n": 2,
            "stop": "END",
            "stream": False,
        },
    )

    # gen_ai.request.model is set at span creation, not here.
    assert span.attributes == {
        GenAIAttributes.GEN_AI_REQUEST_MAX_TOKENS: 100,
        GenAIAttributes.GEN_AI_REQUEST_TEMPERATURE: 0.5,
        GenAIAttributes.GEN_AI_REQUEST_TOP_P: 0.9,
        GenAIAttributes.GEN_AI_REQUEST_TOP_K: 40,
        GenAIAttributes.GEN_AI_REQUEST_FREQUENCY_PENALTY: 0.1,
        GenAIAttributes.GEN_AI_REQUEST_PRESENCE_PENALTY: 0.2,
        SpanAttributes.GEN_AI_REQUEST_REPETITION_PENALTY: 1.1,
        GenAIAttributes.GEN_AI_REQUEST_SEED: 7,
        GenAIAttributes.GEN_AI_REQUEST_CHOICE_COUNT: 2,
        GenAIAttributes.GEN_AI_REQUEST_STOP_SEQUENCES: ["END"],
        SpanAttributes.GEN_AI_IS_STREAMING: False,
    }


@pytest.mark.skipif(
    int(version("together").split(".")[0]) < 2, reason="NOT_GIVEN/omit exist from together 2.x"
)
def test_unset_argument_sentinels_are_not_recorded(span):
    """together 2.x/3.x apps may pass NOT_GIVEN / omit explicitly for unset arguments.
    They aren't valid attribute values, so they must not reach the span."""
    from together._types import NOT_GIVEN, omit

    set_model_input_attributes(
        span,
        {
            "model": "m",
            "max_tokens": NOT_GIVEN,
            "temperature": omit,
            "top_p": NOT_GIVEN,
            "seed": omit,
            "n": NOT_GIVEN,
            "stop": omit,
            "stream": NOT_GIVEN,
        },
    )

    assert span.attributes == {SpanAttributes.GEN_AI_IS_STREAMING: False}


def test_choice_count_is_not_recorded_when_it_is_the_default(span):
    set_model_input_attributes(span, {"model": "m", "n": 1})

    assert GenAIAttributes.GEN_AI_REQUEST_CHOICE_COUNT not in span.attributes


def test_response_usage_includes_cache_and_reasoning_tokens(span):
    set_model_response_attributes(
        span,
        {
            "id": "resp-1",
            "model": "m",
            "choices": [{"finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": 77,
                "completion_tokens": 67,
                "total_tokens": 144,
                "prompt_tokens_details": {"cached_tokens": 66},
                "completion_tokens_details": {"reasoning_tokens": 23},
            },
        },
    )

    assert span.attributes == {
        GenAIAttributes.GEN_AI_RESPONSE_MODEL: "m",
        GenAIAttributes.GEN_AI_RESPONSE_ID: "resp-1",
        GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS: ["stop"],
        GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS: 77,
        GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS: 67,
        SpanAttributes.GEN_AI_USAGE_TOTAL_TOKENS: 144,
        GenAIAttributes.GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS: 66,
        SpanAttributes.GEN_AI_USAGE_REASONING_TOKENS: 23,
    }


class _V1FinishReason(str, Enum):
    """Mirrors together 1.x's `FinishReason` str enum."""

    EOS = "eos"
    TOOL_CALLS = "tool_calls"


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("stop", "stop"),
        ("eos", "stop"),
        ("length", "length"),
        ("tool_calls", "tool_call"),
        ("function_call", "tool_call"),
        (_V1FinishReason.EOS, "stop"),
        (_V1FinishReason.TOOL_CALLS, "tool_call"),
        ("something_new", "something_new"),
        (None, ""),
        ("", ""),
    ],
)
def test_map_finish_reason(raw, expected):
    assert map_finish_reason(raw) == expected
