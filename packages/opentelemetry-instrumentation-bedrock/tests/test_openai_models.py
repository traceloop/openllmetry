"""OpenAI models on Bedrock (openai.gpt-oss-*, openai.gpt-5.x, openai.gpt-6-*).

invoke_model / invoke_model_with_response_stream take and return an OpenAI
chat-completions body for these models, and _get_vendor_model resolves them to
model_vendor "openai". Payloads below are trimmed from real
us.openai.gpt-6-sol and openai.gpt-oss-120b-1:0 responses.
"""

import json
from unittest.mock import MagicMock, patch

from opentelemetry.instrumentation.bedrock import _get_vendor_model
from opentelemetry.instrumentation.bedrock.guardrail import guardrail_handling
from opentelemetry.instrumentation.bedrock.span_utils import (
    BEDROCK_GUARDRAIL_INPUT_FILTER,
    set_model_choice_span_attributes,
    set_model_message_span_attributes,
    set_model_span_attributes,
)
from opentelemetry.instrumentation.bedrock.streaming_wrapper import (
    AsyncStreamingWrapper,
    StreamingWrapper,
)
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.semconv_ai import SpanAttributes


def _mock_span():
    """Return a mock span that records set_attribute calls."""
    span = MagicMock()
    span.is_recording.return_value = True
    attrs = {}

    def set_attr(key, value):
        attrs[key] = value

    span.set_attribute.side_effect = set_attr
    span._attrs = attrs
    return span


def _mock_metric_params():
    mp = MagicMock()
    mp.duration_histogram = None
    mp.token_histogram = None
    return mp


def _chunk(payload):
    return {"chunk": {"bytes": json.dumps(payload).encode()}}


class _IterableList(list):
    """List that accepts attribute assignment (StreamingWrapper proxies setattr)."""


class _AsyncEvents:
    def __init__(self, events):
        self._events = events

    async def __aiter__(self):
        for event in self._events:
            yield event


REQUEST_BODY = {
    "messages": [{"role": "user", "content": "Reply with just: pong"}],
    "max_completion_tokens": 64,
}

RESPONSE_BODY = {
    "choices": [{
        "finish_reason": "stop",
        "index": 0,
        "message": {"annotations": [], "content": "pong", "refusal": None, "role": "assistant"},
    }],
    "id": "chatcmpl-1",
    "model": "us.openai.gpt-6-sol",
    "object": "chat.completion",
    "usage": {"completion_tokens": 5, "prompt_tokens": 11, "total_tokens": 16},
}

TOOL_RESPONSE_BODY = {
    "choices": [{
        "finish_reason": "tool_calls",
        "index": 0,
        "message": {
            "content": None,
            "role": "assistant",
            "tool_calls": [{
                "function": {"arguments": '{"city":"Paris"}', "name": "get_weather"},
                "id": "call_0",
                "type": "function",
            }],
        },
    }],
    "id": "chatcmpl-2",
    "model": "us.openai.gpt-6-sol",
    "object": "chat.completion",
    "usage": {"completion_tokens": 18, "prompt_tokens": 52, "total_tokens": 70},
}


def _stream_chunk(delta, finish_reason=None, **extra):
    return _chunk({
        "choices": [{"delta": delta, "finish_reason": finish_reason, "index": 0}],
        "id": "chatcmpl-3",
        "model": "us.openai.gpt-6-sol",
        "object": "chat.completion.chunk",
        "usage": None,
        **extra,
    })


TEXT_STREAM = [
    _stream_chunk({"content": "", "role": "assistant", "refusal": None}),
    _stream_chunk({"content": "pong"}),
    _stream_chunk({}, "stop", usage={"completion_tokens": 5, "prompt_tokens": 11, "total_tokens": 16}),
    _chunk({
        "choices": [],
        "id": "chatcmpl-3",
        "model": "us.openai.gpt-6-sol",
        "object": "chat.completion.chunk",
        "usage": {"completion_tokens": 5, "prompt_tokens": 11, "total_tokens": 16},
        "amazon-bedrock-invocationMetrics": {"inputTokenCount": 11, "outputTokenCount": 5},
    }),
]


def _collect(events):
    result = {}

    def done(body):
        result["body"] = body

    for _ in StreamingWrapper(_IterableList(events), stream_done_callback=done):
        pass
    return result["body"]


class TestOpenAIVendor:
    def test_gpt_6_profile_resolves_to_openai_vendor(self):
        assert _get_vendor_model("us.openai.gpt-6-sol") == ("aws.bedrock", "openai", "gpt-6-sol")


class TestOpenAIInvokeModelSpanAttributes:
    """invoke_model spans for openai.* models carry usage, finish reason and messages."""

    def test_usage_and_request_params(self):
        span = _mock_span()
        set_model_span_attributes(
            "aws.bedrock", "openai", "gpt-6-sol", span,
            REQUEST_BODY, RESPONSE_BODY, {}, _mock_metric_params(), {},
        )
        assert span._attrs[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS] == 11
        assert span._attrs[GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS] == 5
        assert span._attrs[SpanAttributes.GEN_AI_USAGE_TOTAL_TOKENS] == 16
        assert span._attrs[GenAIAttributes.GEN_AI_REQUEST_MAX_TOKENS] == 64
        assert span._attrs[GenAIAttributes.GEN_AI_RESPONSE_MODEL] == "us.openai.gpt-6-sol"

    @patch("opentelemetry.instrumentation.bedrock.span_utils.should_send_prompts", return_value=True)
    def test_output_message_and_finish_reason(self, _mock):
        span = _mock_span()
        set_model_choice_span_attributes("openai", span, RESPONSE_BODY)
        assert span._attrs[GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS] == ("stop",)
        assert json.loads(span._attrs[GenAIAttributes.GEN_AI_OUTPUT_MESSAGES]) == [{
            "role": "assistant",
            "parts": [{"type": "text", "content": "pong"}],
            "finish_reason": "stop",
        }]

    @patch("opentelemetry.instrumentation.bedrock.span_utils.should_send_prompts", return_value=True)
    def test_tool_call_output(self, _mock):
        span = _mock_span()
        set_model_choice_span_attributes("openai", span, TOOL_RESPONSE_BODY)
        assert span._attrs[GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS] == ("tool_call",)
        output = json.loads(span._attrs[GenAIAttributes.GEN_AI_OUTPUT_MESSAGES])
        assert output[0]["parts"] == [{
            "type": "tool_call",
            "name": "get_weather",
            "id": "call_0",
            "arguments": {"city": "Paris"},
        }]
        assert output[0]["finish_reason"] == "tool_call"

    @patch("opentelemetry.instrumentation.bedrock.span_utils.should_send_prompts", return_value=True)
    def test_input_messages(self, _mock):
        span = _mock_span()
        request_body = {"messages": [
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": [{"type": "text", "text": "Weather in Paris?"}]},
            {"role": "assistant", "content": None, "tool_calls": [{
                "id": "call_0",
                "type": "function",
                "function": {"name": "get_weather", "arguments": '{"city":"Paris"}'},
            }]},
            {"role": "tool", "tool_call_id": "call_0", "content": "18C, sunny"},
        ]}
        set_model_message_span_attributes("openai", span, request_body)
        assert json.loads(span._attrs[GenAIAttributes.GEN_AI_INPUT_MESSAGES]) == [
            {"role": "system", "parts": [{"type": "text", "content": "Be brief."}]},
            {"role": "user", "parts": [{"type": "text", "content": "Weather in Paris?"}]},
            {"role": "assistant", "parts": [{
                "type": "tool_call",
                "name": "get_weather",
                "id": "call_0",
                "arguments": {"city": "Paris"},
            }]},
            {"role": "tool", "parts": [
                {"type": "tool_call_response", "id": "call_0", "response": "18C, sunny"},
            ]},
        ]


class TestOpenAIStreamingAccumulation:
    """invoke_model_with_response_stream: chat.completion.chunk events are folded
    into a chat.completion-shaped body instead of string-concatenated."""

    def test_text_stream(self):
        body = _collect(TEXT_STREAM)
        assert body["model"] == "us.openai.gpt-6-sol"
        assert body["id"] == "chatcmpl-3"
        assert body["choices"][0]["message"]["content"] == "pong"
        assert body["choices"][0]["finish_reason"] == "stop"
        assert body["usage"]["prompt_tokens"] == 11

        span = _mock_span()
        set_model_span_attributes(
            "aws.bedrock", "openai", "gpt-6-sol", span,
            REQUEST_BODY, body, {}, _mock_metric_params(), {},
        )
        assert span._attrs[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS] == 11
        assert span._attrs[GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS] == 5
        assert span._attrs[GenAIAttributes.GEN_AI_RESPONSE_MODEL] == "us.openai.gpt-6-sol"

    @patch("opentelemetry.instrumentation.bedrock.span_utils.should_send_prompts", return_value=True)
    def test_tool_call_stream(self, _mock):
        body = _collect([
            _stream_chunk({"content": "", "role": "assistant"}),
            _stream_chunk({"tool_calls": [{
                "function": {"arguments": "", "name": "get_weather"},
                "id": "call_0", "index": 0, "type": "function",
            }]}),
            _stream_chunk({"tool_calls": [{"function": {"arguments": '{"city":'}, "index": 0}]}),
            _stream_chunk({"tool_calls": [{"function": {"arguments": '"Paris"}'}, "index": 0}]}),
            _stream_chunk({}, "tool_calls", usage={"completion_tokens": 18, "prompt_tokens": 52}),
        ])
        span = _mock_span()
        set_model_choice_span_attributes("openai", span, body)
        assert span._attrs[GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS] == ("tool_call",)
        output = json.loads(span._attrs[GenAIAttributes.GEN_AI_OUTPUT_MESSAGES])
        assert output[0]["parts"] == [{
            "type": "tool_call",
            "name": "get_weather",
            "id": "call_0",
            "arguments": {"city": "Paris"},
        }]

    @patch("opentelemetry.instrumentation.bedrock.span_utils.should_send_prompts", return_value=True)
    def test_interleaved_tool_call_stream_is_routed_by_index(self, _mock):
        body = _collect([
            _stream_chunk({"tool_calls": [{
                "function": {"arguments": "", "name": "get_weather"},
                "id": "call_0", "index": 0, "type": "function",
            }]}),
            _stream_chunk({"tool_calls": [{
                "function": {"arguments": "", "name": "get_weather"},
                "id": "call_1", "index": 1, "type": "function",
            }]}),
            _stream_chunk({"tool_calls": [{"function": {"arguments": '{"city":"Paris"}'}, "index": 0}]}),
            _stream_chunk({"tool_calls": [{"function": {"arguments": '{"city":"Tokyo"}'}, "index": 1}]}),
            _stream_chunk({}, "tool_calls"),
        ])
        span = _mock_span()
        set_model_choice_span_attributes("openai", span, body)
        output = json.loads(span._attrs[GenAIAttributes.GEN_AI_OUTPUT_MESSAGES])
        assert [(p["id"], p["arguments"]) for p in output[0]["parts"]] == [
            ("call_0", {"city": "Paris"}),
            ("call_1", {"city": "Tokyo"}),
        ]

    def test_usage_falls_back_to_invocation_metrics(self):
        """gpt-oss streams send no usage object, only amazon-bedrock-invocationMetrics."""
        body = _collect([
            _chunk({
                "choices": [{"delta": {"content": "pong"}, "finish_reason": None, "index": 0}],
                "model": "openai.gpt-oss-120b-1:0",
                "object": "chat.completion.chunk",
            }),
            _chunk({
                "choices": [{"delta": {}, "finish_reason": "stop", "index": 0}],
                "model": "openai.gpt-oss-120b-1:0",
                "object": "chat.completion.chunk",
                "amazon-bedrock-invocationMetrics": {"inputTokenCount": 72, "outputTokenCount": 27},
            }),
        ])
        span = _mock_span()
        set_model_span_attributes(
            "aws.bedrock", "openai", "gpt-oss-120b-1:0", span,
            REQUEST_BODY, body, {}, _mock_metric_params(), {},
        )
        assert span._attrs[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS] == 72
        assert span._attrs[GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS] == 27

    def test_guardrail_fields_kept_for_guardrail_handling(self):
        """Guardrail action and trace ride on the last chunk (gpt-oss with a guardrail)."""
        guardrail_trace = {"guardrail": {"input": {"gr-1": {
            "wordPolicy": {"customWords": [{"match": "zebracorn", "action": "BLOCKED", "detected": True}]},
        }}}}
        body = _collect([
            _chunk({
                "choices": [{"delta": {"content": "blocked input"}, "finish_reason": None, "index": 0}],
                "model": "openai.gpt-oss-120b-1:0",
                "object": "chat.completion.chunk",
            }),
            _chunk({
                "choices": [{"delta": {}, "finish_reason": "stop", "index": 0}],
                "model": "openai.gpt-oss-120b-1:0",
                "object": "chat.completion.chunk",
                "amazon-bedrock-guardrailAction": "INTERVENED",
                "amazon-bedrock-trace": guardrail_trace,
            }),
        ])
        span = _mock_span()
        guardrail_handling(span, body, "aws.bedrock", "gpt-oss-120b-1:0", MagicMock())
        assert json.loads(span._attrs[BEDROCK_GUARDRAIL_INPUT_FILTER])["words"] == ["zebracorn"]

    async def test_async_text_stream(self):
        result = {}

        def done(body):
            result["body"] = body

        async for _ in AsyncStreamingWrapper(_AsyncEvents(TEXT_STREAM), stream_done_callback=done):
            pass
        assert result["body"]["model"] == "us.openai.gpt-6-sol"
        assert result["body"]["choices"][0]["message"]["content"] == "pong"
        assert result["body"]["usage"]["completion_tokens"] == 5
