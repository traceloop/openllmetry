"""
Tests for reasoning models (reasoning_format="parsed").

Groq returns the model's thinking in `message.reasoning` (or `delta.reasoning` when
streaming) and counts it in `usage.completion_tokens_details.reasoning_tokens`.
The responses are served by an httpx MockTransport, so no network calls, no cassettes.
"""

import json

import httpx
import pytest
from groq import Groq
from opentelemetry.semconv._incubating.attributes import gen_ai_attributes as GenAIAttributes
from opentelemetry.semconv_ai import SpanAttributes

MODEL = "qwen/qwen3-32b"
REASONING = "The user wants a joke about opentelemetry. Spans and traces are the obvious material."
ANSWER = "Why did the span refuse to end? It had unresolved attributes."
USAGE = {
    "prompt_tokens": 18,
    "completion_tokens": 40,
    "total_tokens": 58,
    "completion_tokens_details": {"reasoning_tokens": 25},
}


def _completion_body() -> dict:
    return {
        "id": "chatcmpl-reasoning",
        "object": "chat.completion",
        "created": 1758000000,
        "model": MODEL,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": ANSWER, "reasoning": REASONING},
                "finish_reason": "stop",
            }
        ],
        "usage": USAGE,
    }


def _stream_body() -> str:
    def chunk(delta: dict, finish_reason=None, usage=None) -> str:
        event = {
            "id": "chatcmpl-reasoning",
            "object": "chat.completion.chunk",
            "created": 1758000000,
            "model": MODEL,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
        }
        if usage:
            event["x_groq"] = {"id": "req_1", "usage": usage}
        return "data: " + json.dumps(event) + "\n\n"

    # Groq streams the reasoning deltas first, then the answer, then usage on the last chunk.
    head, tail = REASONING[:40], REASONING[40:]
    return "".join(
        [
            chunk({"role": "assistant", "content": ""}),
            chunk({"reasoning": head}),
            chunk({"reasoning": tail}),
            chunk({"content": ANSWER[:20]}),
            chunk({"content": ANSWER[20:]}),
            chunk({}, finish_reason="stop", usage=USAGE),
            "data: [DONE]\n\n",
        ]
    )


def _handler(request: httpx.Request) -> httpx.Response:
    if json.loads(request.content).get("stream"):
        return httpx.Response(200, text=_stream_body(), headers={"content-type": "text/event-stream"})
    return httpx.Response(200, json=_completion_body())


@pytest.fixture
def mock_groq_client():
    with httpx.Client(transport=httpx.MockTransport(_handler)) as http_client:
        yield Groq(api_key="api-key", http_client=http_client)


def _output_parts(span) -> list:
    return json.loads(span.attributes[GenAIAttributes.GEN_AI_OUTPUT_MESSAGES])[0]["parts"]


def test_chat_reasoning_is_recorded(instrument_legacy, mock_groq_client, span_exporter):
    mock_groq_client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "Tell me a joke about opentelemetry"}],
        reasoning_format="parsed",
    )

    (span,) = span_exporter.get_finished_spans()
    assert _output_parts(span) == [
        {"type": "text", "content": ANSWER},
        {"type": "reasoning", "content": REASONING},
    ]
    assert span.attributes[GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS] == 40
    assert span.attributes[SpanAttributes.GEN_AI_USAGE_REASONING_TOKENS] == 25


def test_chat_streaming_reasoning_is_accumulated(instrument_legacy, mock_groq_client, span_exporter):
    response = mock_groq_client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "Tell me a joke about opentelemetry"}],
        reasoning_format="parsed",
        stream=True,
    )
    chunks = list(response)
    assert len(chunks) == 6

    (span,) = span_exporter.get_finished_spans()
    assert _output_parts(span) == [
        {"type": "text", "content": ANSWER},
        {"type": "reasoning", "content": REASONING},
    ]
    assert span.attributes[GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS] == 40
    assert span.attributes[SpanAttributes.GEN_AI_USAGE_REASONING_TOKENS] == 25


def test_chat_without_reasoning_has_no_reasoning_part(instrument_legacy, span_exporter):
    def handler(request: httpx.Request) -> httpx.Response:
        body = _completion_body()
        body["choices"][0]["message"].pop("reasoning")
        body["usage"].pop("completion_tokens_details")
        return httpx.Response(200, json=body)

    with httpx.Client(transport=httpx.MockTransport(handler)) as http_client:
        Groq(api_key="api-key", http_client=http_client).chat.completions.create(
            model=MODEL, messages=[{"role": "user", "content": "Hi"}]
        )

    (span,) = span_exporter.get_finished_spans()
    assert _output_parts(span) == [{"type": "text", "content": ANSWER}]
    assert SpanAttributes.GEN_AI_USAGE_REASONING_TOKENS not in span.attributes
