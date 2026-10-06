"""Regression tests for issue #4555: generate_content_stream spans must end on
early exit (break, close()/aclose(), or a dropped stream), not only when the
stream is fully consumed.

Self-contained: uses httpx.MockTransport (no API key, no network, no VCR).
"""

import gc
import json

import httpx
import pytest
from google import genai
from google.genai import types
from opentelemetry.instrumentation.google_generativeai import (
    GoogleGenerativeAiInstrumentor,
)
from opentelemetry.sdk.trace import SpanProcessor, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)

PIECES = ["one ", "two ", "three"]


def _sse_handler(request):
    body = "".join(
        "data: "
        + json.dumps(
            {
                "candidates": [
                    {
                        "content": {"role": "model", "parts": [{"text": text}]},
                        **({"finishReason": "STOP"} if i == 2 else {}),
                    }
                ]
            }
        )
        + "\n\n"
        for i, text in enumerate(PIECES)
    )
    return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})


class _OpenSpans(SpanProcessor):
    """Tracks spans that have started but not yet ended."""

    def __init__(self):
        self.open = set()

    def on_start(self, span, parent_context=None):
        self.open.add(span.context.span_id)

    def on_end(self, span):
        self.open.discard(span.context.span_id)

    def shutdown(self):
        pass

    def force_flush(self, timeout_millis=30000):
        return True


@pytest.fixture()
def instrumented():
    provider = TracerProvider()
    open_spans = _OpenSpans()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(open_spans)
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    instrumentor = GoogleGenerativeAiInstrumentor()
    instrumentor.instrument(tracer_provider=provider)
    yield open_spans, exporter
    instrumentor.uninstrument()
    provider.shutdown()


def _sync_client():
    return genai.Client(
        api_key="x",
        http_options=types.HttpOptions(
            base_url="http://mock",
            httpx_client=httpx.Client(transport=httpx.MockTransport(_sse_handler)),
        ),
    )


def _async_client():
    return genai.Client(
        api_key="x",
        http_options=types.HttpOptions(
            base_url="http://mock",
            httpx_async_client=httpx.AsyncClient(transport=httpx.MockTransport(_sse_handler)),
        ),
    )


def _finished_stream_span(exporter):
    spans = exporter.get_finished_spans()
    assert len(spans) == 1, f"expected 1 finished span, got {len(spans)}"
    return spans[0]


# ---------------------------------------------------------------------------
# Sync generate_content_stream
# ---------------------------------------------------------------------------


def test_sync_full_consumption_ends_span(instrumented):
    open_spans, exporter = instrumented
    client = _sync_client()
    for _ in client.models.generate_content_stream(model="m", contents="hi"):
        pass
    gc.collect()
    assert len(open_spans.open) == 0
    span = _finished_stream_span(exporter)
    assert span.attributes[GenAIAttributes.GEN_AI_OUTPUT_MESSAGES] is not None


def test_sync_break_ends_span_with_partial_output(instrumented):
    open_spans, exporter = instrumented
    client = _sync_client()
    for _ in client.models.generate_content_stream(model="m", contents="hi"):
        break
    gc.collect()
    assert len(open_spans.open) == 0
    span = _finished_stream_span(exporter)
    output = span.attributes[GenAIAttributes.GEN_AI_OUTPUT_MESSAGES]
    assert "one " in output
    assert "three" not in output


def test_sync_close_ends_span(instrumented):
    open_spans, exporter = instrumented
    client = _sync_client()
    stream = client.models.generate_content_stream(model="m", contents="hi")
    next(stream)
    stream.close()
    gc.collect()
    assert len(open_spans.open) == 0
    _finished_stream_span(exporter)


def test_sync_dropped_stream_ends_span_after_gc(instrumented):
    open_spans, exporter = instrumented
    client = _sync_client()
    stream = client.models.generate_content_stream(model="m", contents="hi")
    next(stream)
    del stream
    gc.collect()
    assert len(open_spans.open) == 0
    _finished_stream_span(exporter)


# ---------------------------------------------------------------------------
# Async generate_content_stream
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_async_full_consumption_ends_span(instrumented):
    open_spans, exporter = instrumented
    client = _async_client()
    stream = await client.aio.models.generate_content_stream(model="m", contents="hi")
    async for _ in stream:
        pass
    gc.collect()
    assert len(open_spans.open) == 0
    span = _finished_stream_span(exporter)
    assert span.attributes[GenAIAttributes.GEN_AI_OUTPUT_MESSAGES] is not None


@pytest.mark.asyncio
async def test_async_break_then_aclose_ends_span(instrumented):
    open_spans, exporter = instrumented
    client = _async_client()
    stream = await client.aio.models.generate_content_stream(model="m", contents="hi")
    async for _ in stream:
        break
    await stream.aclose()
    gc.collect()
    assert len(open_spans.open) == 0
    span = _finished_stream_span(exporter)
    output = span.attributes[GenAIAttributes.GEN_AI_OUTPUT_MESSAGES]
    assert "one " in output
    assert "three" not in output


@pytest.mark.asyncio
async def test_async_aclose_ends_span(instrumented):
    open_spans, exporter = instrumented
    client = _async_client()
    stream = await client.aio.models.generate_content_stream(model="m", contents="hi")
    await anext(stream)
    await stream.aclose()
    gc.collect()
    assert len(open_spans.open) == 0
    _finished_stream_span(exporter)
