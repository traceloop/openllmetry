"""Regression tests for issue #4583.

`_set_response_attributes` crashed with `TypeError: 'NoneType' object is not
iterable` (swallowed by `@dont_throw`) whenever a response's `usage` carried
`prompt_tokens_details=None` — which the openai SDK sets for every response
that omits the optional field. As a result `gen_ai.usage.cache_read.input_tokens`
was never recorded. Self-contained: mock transports, no network.
"""

import httpx
import pytest
from openai import AsyncOpenAI, OpenAI
from opentelemetry.instrumentation.openai import OpenAIInstrumentor
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

CACHE_READ_ATTR = "gen_ai.usage.cache_read.input_tokens"
INPUT_TOKENS_ATTR = "gen_ai.usage.input_tokens"


def _completion_response(prompt_tokens_details):
    usage = {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10}
    if prompt_tokens_details is not None:
        usage["prompt_tokens_details"] = prompt_tokens_details
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 0,
        "model": "gpt-4o-mini",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "hi"},
                "finish_reason": "stop",
            }
        ],
        "usage": usage,
    }


def _stream_sse(prompt_tokens_details):
    usage = {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10}
    if prompt_tokens_details is not None:
        usage["prompt_tokens_details"] = prompt_tokens_details
    chunk = (
        '{"id":"chatcmpl-1","object":"chat.completion.chunk","created":0,'
        '"model":"gpt-4o-mini","choices":[{"index":0,"delta":{"role":"assistant",'
        '"content":"hi"},"finish_reason":null}]}'
    )
    usage_chunk = (
        '{"id":"chatcmpl-1","object":"chat.completion.chunk","created":0,'
        '"model":"gpt-4o-mini","choices":[],"usage":' + _json(usage) + "}"
    )
    return ("data: " + chunk + "\n\ndata: " + usage_chunk + "\n\ndata: [DONE]\n").encode()


def _json(obj):
    import json

    return json.dumps(obj)


@pytest.fixture()
def span_exporter():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    instrumentor = OpenAIInstrumentor()
    instrumentor.instrument(tracer_provider=provider)
    yield exporter
    instrumentor.uninstrument()


@pytest.fixture()
def sync_client(span_exporter):
    def respond(request):
        return httpx.Response(200, json=_completion_response(None))

    return OpenAI(
        api_key="test",
        http_client=httpx.Client(transport=httpx.MockTransport(respond)),
    )


def _chat_span(span_exporter):
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    return spans[0]


def test_missing_prompt_tokens_details_records_zero_cache_read_tokens(
    span_exporter, sync_client
):
    # The SDK sets prompt_tokens_details=None when the response omits the
    # optional field; the attribute must still be recorded (as 0), not dropped.
    sync_client.chat.completions.create(
        model="gpt-4o-mini", messages=[{"role": "user", "content": "hello"}]
    )

    span = _chat_span(span_exporter)
    assert span.attributes[INPUT_TOKENS_ATTR] == 7
    assert span.attributes[CACHE_READ_ATTR] == 0


def test_present_prompt_tokens_details_records_cached_tokens(span_exporter):
    def respond(request):
        return httpx.Response(
            200, json=_completion_response({"cached_tokens": 4})
        )

    client = OpenAI(
        api_key="test",
        http_client=httpx.Client(transport=httpx.MockTransport(respond)),
    )
    client.chat.completions.create(
        model="gpt-4o-mini", messages=[{"role": "user", "content": "hello"}]
    )

    span = _chat_span(span_exporter)
    assert span.attributes[INPUT_TOKENS_ATTR] == 7
    assert span.attributes[CACHE_READ_ATTR] == 4


def test_streaming_missing_prompt_tokens_details_records_zero_cache_read_tokens(
    span_exporter,
):
    def respond(request):
        return httpx.Response(
            200,
            content=_stream_sse(None),
            headers={"content-type": "text/event-stream"},
        )

    client = OpenAI(
        api_key="test",
        http_client=httpx.Client(transport=httpx.MockTransport(respond)),
    )
    stream = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "hello"}],
        stream=True,
        stream_options={"include_usage": True},
    )
    for _ in stream:
        pass

    span = _chat_span(span_exporter)
    assert span.attributes[INPUT_TOKENS_ATTR] == 7
    assert span.attributes[CACHE_READ_ATTR] == 0


@pytest.mark.asyncio
async def test_async_missing_prompt_tokens_details_records_zero_cache_read_tokens(
    span_exporter,
):
    def respond(request):
        return httpx.Response(200, json=_completion_response(None))

    client = AsyncOpenAI(
        api_key="test",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)),
    )
    await client.chat.completions.create(
        model="gpt-4o-mini", messages=[{"role": "user", "content": "hello"}]
    )

    span = _chat_span(span_exporter)
    assert span.attributes[INPUT_TOKENS_ATTR] == 7
    assert span.attributes[CACHE_READ_ATTR] == 0
