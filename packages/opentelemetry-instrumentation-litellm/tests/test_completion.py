"""Chat completion tracing tests.

These use litellm's built-in ``mock_response`` so they need no API key or network —
litellm still resolves the provider (``openai`` for ``gpt-*``) and returns a normalized
``ModelResponse``, exercising the same code path as a real call.
"""

import json

import litellm
import pytest
from opentelemetry.instrumentation.litellm import (
    _map_finish_reason,
    _set_response_attributes,
)
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.semconv_ai import LLMRequestTypeValues, SpanAttributes

MESSAGES = [{"role": "user", "content": "What is the capital of France?"}]


def _input_messages(attrs):
    return json.loads(attrs[GenAIAttributes.GEN_AI_INPUT_MESSAGES])


def _output_messages(attrs):
    return json.loads(attrs[GenAIAttributes.GEN_AI_OUTPUT_MESSAGES])


def test_completion(instrument_legacy, span_exporter):
    response = litellm.completion(
        model="gpt-3.5-turbo",
        messages=MESSAGES,
        mock_response="The capital of France is Paris.",
    )

    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == ["litellm.chat"]
    span = spans[0]
    attrs = span.attributes

    assert attrs[GenAIAttributes.GEN_AI_PROVIDER_NAME] == "openai"
    assert (
        attrs[GenAIAttributes.GEN_AI_OPERATION_NAME]
        == GenAIAttributes.GenAiOperationNameValues.CHAT.value
    )
    assert attrs[GenAIAttributes.GEN_AI_REQUEST_MODEL] == "gpt-3.5-turbo"

    assert _input_messages(attrs) == [
        {
            "role": "user",
            "parts": [{"type": "text", "content": "What is the capital of France?"}],
        }
    ]

    output = _output_messages(attrs)
    assert output[0]["role"] == "assistant"
    assert output[0]["parts"] == [
        {"type": "text", "content": "The capital of France is Paris."}
    ]
    assert output[0]["finish_reason"] == "stop"
    assert attrs[GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS] == ("stop",)

    assert attrs[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS] > 0
    assert attrs[SpanAttributes.LLM_USAGE_TOTAL_TOKENS] > 0
    assert response.choices[0].message.content == "The capital of France is Paris."


@pytest.mark.asyncio
async def test_acompletion(instrument_legacy, span_exporter):
    await litellm.acompletion(
        model="gpt-3.5-turbo",
        messages=MESSAGES,
        mock_response="Paris.",
    )

    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == ["litellm.chat"]
    attrs = spans[0].attributes
    assert attrs[GenAIAttributes.GEN_AI_PROVIDER_NAME] == "openai"
    assert _output_messages(attrs)[0]["parts"] == [
        {"type": "text", "content": "Paris."}
    ]


def test_completion_serializes_input_tool_calls(instrument_legacy, span_exporter):
    """An assistant turn carrying tool_calls (and null content) must serialize the
    tool calls on the input side, not collapse to a bare role. Regression for input
    messages dropping tool_calls while the output path serialized them."""
    messages = [
        {"role": "user", "content": "What's the weather in Paris?"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_abc",
                    "type": "function",
                    "function": {
                        "name": "get_current_weather",
                        "arguments": '{"location": "Paris"}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_abc",
            "content": '{"temperature": 18}',
        },
    ]

    litellm.completion(
        model="gpt-3.5-turbo",
        messages=messages,
        mock_response="It's 18°C in Paris.",
    )

    input_messages = _input_messages(span_exporter.get_finished_spans()[0].attributes)

    assert input_messages[0]["role"] == "user"

    # The assistant turn keeps its tool call instead of collapsing to just a role,
    # and arguments are parsed into a dict per the OTel tool_call part schema.
    assistant = input_messages[1]
    assert assistant["role"] == "assistant"
    assert assistant["parts"] == [
        {
            "type": "tool_call",
            "name": "get_current_weather",
            "id": "call_abc",
            "arguments": {"location": "Paris"},
        }
    ]

    # The tool result is linked back to its call via a tool_call_response part.
    tool_message = input_messages[2]
    assert tool_message["role"] == "tool"
    assert tool_message["parts"] == [
        {
            "type": "tool_call_response",
            "id": "call_abc",
            "response": '{"temperature": 18}',
        }
    ]


def test_completion_emits_metrics(instrument_legacy, span_exporter, metric_reader):
    from opentelemetry.semconv_ai import Meters

    litellm.completion(
        model="gpt-3.5-turbo",
        messages=MESSAGES,
        mock_response="The capital of France is Paris.",
    )

    metrics_data = metric_reader.get_metrics_data()
    metric_names = {
        metric.name
        for rm in metrics_data.resource_metrics
        for sm in rm.scope_metrics
        for metric in sm.metrics
    }
    assert Meters.LLM_TOKEN_USAGE in metric_names
    assert Meters.LLM_OPERATION_DURATION in metric_names


def test_completion_no_content_when_disabled(instrument_legacy, span_exporter, monkeypatch):
    monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", "False")
    litellm.completion(
        model="gpt-3.5-turbo",
        messages=MESSAGES,
        mock_response="Paris.",
    )

    attrs = span_exporter.get_finished_spans()[0].attributes
    # Content-bearing message attributes are omitted.
    assert GenAIAttributes.GEN_AI_INPUT_MESSAGES not in attrs
    assert GenAIAttributes.GEN_AI_OUTPUT_MESSAGES not in attrs
    # Non-content attributes are still present, including finish_reasons (which is
    # not gated by the content opt-in).
    assert attrs[GenAIAttributes.GEN_AI_REQUEST_MODEL] == "gpt-3.5-turbo"
    assert attrs[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS] > 0
    assert attrs[GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS] == ("stop",)


def test_map_finish_reason_normalizes_to_otel_vocabulary():
    assert _map_finish_reason("tool_calls") == "tool_call"
    assert _map_finish_reason("function_call") == "tool_call"
    assert _map_finish_reason("stop") == "stop"
    assert _map_finish_reason(None) == ""
    assert _map_finish_reason("") == ""


def test_finish_reasons_are_mapped_and_deduped(tracer_provider, span_exporter):
    """gen_ai.response.finish_reasons must map provider values (tool_calls ->
    tool_call) and contain unique values only, and the per-message output finish
    reason must be mapped too."""
    tracer = tracer_provider.get_tracer(__name__)
    span = tracer.start_span("litellm.chat")
    response_dict = {
        "id": "chatcmpl-x",
        "model": "gpt-4o",
        "choices": [
            {
                "index": 0,
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "c1",
                            "type": "function",
                            "function": {"name": "f", "arguments": "{}"},
                        }
                    ],
                },
            },
            {
                "index": 1,
                "finish_reason": "tool_calls",
                "message": {"role": "assistant", "content": "hi"},
            },
        ],
    }

    _set_response_attributes(span, LLMRequestTypeValues.CHAT, response_dict)
    span.end()

    attrs = span_exporter.get_finished_spans()[0].attributes
    # Deduped to a single mapped value despite two tool_calls choices.
    assert attrs[GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS] == ("tool_call",)

    output = json.loads(attrs[GenAIAttributes.GEN_AI_OUTPUT_MESSAGES])
    assert output[0]["finish_reason"] == "tool_call"
    tool_parts = [p for p in output[0]["parts"] if p["type"] == "tool_call"]
    assert tool_parts[0]["name"] == "f"


def test_tool_definitions_captured(instrument_legacy, span_exporter):
    """tools passed to completion() are captured in gen_ai.tool.definitions."""
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the weather for a city",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        }
    ]
    litellm.completion(
        model="gpt-3.5-turbo",
        messages=MESSAGES,
        tools=tools,
        mock_response="It is sunny.",
    )

    attrs = span_exporter.get_finished_spans()[0].attributes
    tool_defs = json.loads(attrs[GenAIAttributes.GEN_AI_TOOL_DEFINITIONS])
    assert tool_defs == [
        {
            "type": "function",
            "name": "get_weather",
            "description": "Get the weather for a city",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        }
    ]
