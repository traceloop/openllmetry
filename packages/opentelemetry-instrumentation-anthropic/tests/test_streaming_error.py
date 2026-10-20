"""Regression tests for spans left open when a stream fails mid-way."""

import json

import anthropic
import httpx2 as httpx
import pytest
from opentelemetry.trace.status import StatusCode


def _sse(event, data):
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


# one text delta, then the API's mid-stream error event
ERROR_BODY = "".join(
    [
        _sse(
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": "msg_1",
                    "type": "message",
                    "role": "assistant",
                    "model": "claude-x",
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": 11, "output_tokens": 1},
                },
            },
        ),
        _sse(
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            },
        ),
        _sse(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "Hel"},
            },
        ),
        _sse(
            "error",
            {
                "type": "error",
                "error": {"type": "overloaded_error", "message": "Overloaded"},
            },
        ),
    ]
).encode()

STREAM_KWARGS = dict(
    model="claude-x",
    max_tokens=16,
    messages=[{"role": "user", "content": "hi"}],
    stream=True,
)


def test_sync_stream_error_ends_span_with_error_status(instrument_legacy, span_exporter):
    transport = httpx.MockTransport(lambda request: httpx.Response(200, content=ERROR_BODY))
    client = anthropic.Anthropic(api_key="test-key", http_client=httpx.Client(transport=transport))

    with pytest.raises(anthropic.APIError):
        for _ in client.messages.create(**STREAM_KWARGS):
            pass

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].status.status_code == StatusCode.ERROR


@pytest.mark.asyncio
async def test_async_stream_error_ends_span_with_error_status(instrument_legacy, span_exporter):
    async def handler(request):
        return httpx.Response(200, content=ERROR_BODY)

    transport = httpx.MockTransport(handler)
    client = anthropic.AsyncAnthropic(api_key="test-key", http_client=httpx.AsyncClient(transport=transport))

    with pytest.raises(anthropic.APIError):
        async for _ in await client.messages.create(**STREAM_KWARGS):
            pass

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].status.status_code == StatusCode.ERROR
