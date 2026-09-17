import json

import pytest
from mistralai.models import ChatCompletionResponse
from opentelemetry.instrumentation.mistralai import _set_response_attributes
from opentelemetry.instrumentation.mistralai.utils import TRACELOOP_TRACE_CONTENT
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.semconv_ai import LLMRequestTypeValues

REASONING = "The user asks for the capital of France. It is Paris."
ANSWER = "Paris."

# The shape magistral models return: the assistant content is a list of chunks,
# a thinking chunk (a list of text chunks) followed by the answer.
MAGISTRAL_RESPONSE = {
    "id": "b3f1d1f0e8d94f8f9d4b8d7e5a1c2b3d",
    "object": "chat.completion",
    "created": 1758000000,
    "model": "magistral-medium-latest",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": [{"type": "text", "text": REASONING}]},
                    {"type": "text", "text": ANSWER},
                ],
                "tool_calls": None,
            },
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 20, "total_tokens": 61, "completion_tokens": 41},
}


@pytest.fixture
def trace_content(monkeypatch):
    monkeypatch.setenv(TRACELOOP_TRACE_CONTENT, "true")


def test_chunked_content_is_recorded_as_json(tracer_provider, span_exporter, trace_content):
    response = ChatCompletionResponse.model_validate(MAGISTRAL_RESPONSE)
    tracer = tracer_provider.get_tracer("test")
    with tracer.start_as_current_span("mistralai.chat") as span:
        _set_response_attributes(span, LLMRequestTypeValues.CHAT, response)

    attributes = span_exporter.get_finished_spans()[0].attributes
    content = attributes.get(f"{GenAIAttributes.GEN_AI_COMPLETION}.0.content")
    assert content is not None, "the completion content was dropped"
    chunks = json.loads(content)
    assert chunks[0]["type"] == "thinking"
    assert chunks[0]["thinking"][0]["text"] == REASONING
    assert chunks[1] == {"type": "text", "text": ANSWER}
    assert attributes.get(f"{GenAIAttributes.GEN_AI_COMPLETION}.0.role") == "assistant"
    assert attributes.get(f"{GenAIAttributes.GEN_AI_COMPLETION}.0.finish_reason") == "stop"


def test_string_content_is_unchanged(tracer_provider, span_exporter, trace_content):
    plain = dict(MAGISTRAL_RESPONSE)
    plain["choices"] = [
        {
            "index": 0,
            "message": {"role": "assistant", "content": ANSWER, "tool_calls": None},
            "finish_reason": "stop",
        }
    ]
    response = ChatCompletionResponse.model_validate(plain)
    tracer = tracer_provider.get_tracer("test")
    with tracer.start_as_current_span("mistralai.chat") as span:
        _set_response_attributes(span, LLMRequestTypeValues.CHAT, response)

    attributes = span_exporter.get_finished_spans()[0].attributes
    assert attributes.get(f"{GenAIAttributes.GEN_AI_COMPLETION}.0.content") == ANSWER


def _magistral_stream():
    from mistralai.models import (
        CompletionChunk,
        CompletionEvent,
        CompletionResponseStreamChoice,
        DeltaMessage,
    )

    def event(content, finish_reason=None):
        return CompletionEvent(
            data=CompletionChunk(
                id="b3f1d1f0e8d94f8f9d4b8d7e5a1c2b3d",
                model="magistral-medium-latest",
                choices=[
                    CompletionResponseStreamChoice(
                        index=0,
                        delta=DeltaMessage(role="assistant", content=content),
                        finish_reason=finish_reason,
                    )
                ],
            )
        )

    # A streamed reasoning answer: the thinking arrives in pieces, then the text.
    return [
        event([{"type": "thinking", "thinking": [{"type": "text", "text": "The user asks "}]}]),
        event([{"type": "thinking", "thinking": [{"type": "text", "text": "for Paris."}]}]),
        event([{"type": "text", "text": ANSWER}]),
        event(None, finish_reason="stop"),
    ]


def _assert_streamed_completion(span_exporter):
    attributes = span_exporter.get_finished_spans()[0].attributes
    content = attributes.get(f"{GenAIAttributes.GEN_AI_COMPLETION}.0.content")
    assert content is not None, "the streamed completion was dropped"
    chunks = json.loads(content)
    assert [c["type"] for c in chunks] == ["thinking", "thinking", "text"]
    assert chunks[2]["text"] == ANSWER
    assert attributes.get(f"{GenAIAttributes.GEN_AI_COMPLETION}.0.finish_reason") == "stop"


def test_streamed_chunked_content_is_accumulated(tracer_provider, span_exporter, trace_content):
    from opentelemetry.instrumentation.mistralai import _accumulate_streaming_response

    span = tracer_provider.get_tracer("test").start_span("mistralai.chat")
    events = list(
        _accumulate_streaming_response(span, None, LLMRequestTypeValues.CHAT, iter(_magistral_stream()))
    )
    assert len(events) == 4, "every event must still reach the caller"
    _assert_streamed_completion(span_exporter)


@pytest.mark.asyncio
async def test_async_streamed_chunked_content_is_accumulated(
    tracer_provider, span_exporter, trace_content
):
    from opentelemetry.instrumentation.mistralai import _aaccumulate_streaming_response

    async def stream():
        for event in _magistral_stream():
            yield event

    span = tracer_provider.get_tracer("test").start_span("mistralai.chat")
    events = [
        e
        async for e in _aaccumulate_streaming_response(
            span, None, LLMRequestTypeValues.CHAT, stream()
        )
    ]
    assert len(events) == 4
    _assert_streamed_completion(span_exporter)
