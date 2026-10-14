"""Regression: cached with_raw_response / with_streaming_response stay traced after uninstrument.

See https://github.com/traceloop/openllmetry/issues/4619
"""

from __future__ import annotations

import httpx
import pytest
from anthropic import Anthropic, AsyncAnthropic
from opentelemetry.instrumentation.anthropic import AnthropicInstrumentor
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter


def _message_response(request: httpx.Request) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": "claude-x",
            "content": [{"type": "text", "text": "Hello"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 3, "output_tokens": 1},
        },
    )


CREATE_KW = dict(
    model="claude-x",
    max_tokens=10,
    messages=[{"role": "user", "content": "hi"}],
)


def _sync_client() -> Anthropic:
    return Anthropic(
        api_key="x",
        http_client=httpx.Client(transport=httpx.MockTransport(_message_response)),
    )


def _async_client() -> AsyncAnthropic:
    return AsyncAnthropic(
        api_key="x",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(_message_response)),
    )


def _call_sync(client: Anthropic, accessor: str) -> None:
    if accessor == "plain":
        client.messages.create(**CREATE_KW)
    elif accessor == "with_raw_response":
        client.messages.with_raw_response.create(**CREATE_KW)
    elif accessor == "with_streaming_response":
        with client.messages.with_streaming_response.create(**CREATE_KW) as response:
            response.parse()
    else:
        raise AssertionError(accessor)


async def _call_async(client: AsyncAnthropic, accessor: str) -> None:
    if accessor == "plain":
        await client.messages.create(**CREATE_KW)
    elif accessor == "with_raw_response":
        await client.messages.with_raw_response.create(**CREATE_KW)
    elif accessor == "with_streaming_response":
        async with client.messages.with_streaming_response.create(**CREATE_KW) as response:
            await response.parse()
    else:
        raise AssertionError(accessor)


@pytest.mark.parametrize(
    "accessor",
    ["plain", "with_raw_response", "with_streaming_response"],
)
def test_uninstrument_stops_cached_sync_accessors(accessor: str) -> None:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))

    instrumentor = AnthropicInstrumentor()
    instrumentor.instrument(tracer_provider=provider)
    client = _sync_client()
    try:
        _call_sync(client, accessor)  # populate cached accessor while instrumented
        assert len(exporter.get_finished_spans()) == 1
        instrumentor.uninstrument()
        exporter.clear()
        _call_sync(client, accessor)
        assert len(exporter.get_finished_spans()) == 0
    finally:
        instrumentor.uninstrument()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "accessor",
    ["plain", "with_raw_response", "with_streaming_response"],
)
async def test_uninstrument_stops_cached_async_accessors(accessor: str) -> None:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))

    instrumentor = AnthropicInstrumentor()
    instrumentor.instrument(tracer_provider=provider)
    client = _async_client()
    try:
        await _call_async(client, accessor)
        assert len(exporter.get_finished_spans()) == 1
        instrumentor.uninstrument()
        exporter.clear()
        await _call_async(client, accessor)
        assert len(exporter.get_finished_spans()) == 0
    finally:
        instrumentor.uninstrument()
        await client.close()
