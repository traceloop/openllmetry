"""Regression tests for https://github.com/traceloop/openllmetry/issues/4532.

ChatStream must end the LLM span when the stream is closed early (sync or
async) instead of leaving it open until garbage collection. The wrapped
stream is mocked so no network access is needed.
"""

from unittest.mock import MagicMock

import pytest
from opentelemetry.instrumentation.openai.shared.chat_wrappers import ChatStream
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.trace import StatusCode


@pytest.fixture
def tracer_and_exporter():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider.get_tracer(__name__), exporter


def _make_stream(tracer, wrapped):
    return ChatStream(tracer.start_span("openai.chat"), wrapped, instance=None)


def test_sync_close_ends_span(tracer_and_exporter):
    tracer, exporter = tracer_and_exporter
    wrapped = MagicMock()
    wrapped.close = MagicMock()

    stream = _make_stream(tracer, wrapped)
    stream.close()

    wrapped.close.assert_called_once_with()
    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "openai.chat"
    assert spans[0].end_time is not None
    assert spans[0].status.status_code == StatusCode.OK


async def test_async_close_ends_span(tracer_and_exporter):
    tracer, exporter = tracer_and_exporter
    closed = []

    async def aclose():
        closed.append(True)

    wrapped = MagicMock()
    wrapped.close = aclose

    stream = _make_stream(tracer, wrapped)
    await stream.close()

    assert closed == [True]
    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].end_time is not None
    assert spans[0].status.status_code == StatusCode.OK


def test_close_is_idempotent(tracer_and_exporter):
    tracer, exporter = tracer_and_exporter
    wrapped = MagicMock()
    wrapped.close = MagicMock()

    stream = _make_stream(tracer, wrapped)
    stream.close()
    stream.close()

    # underlying close still delegates, but the span is ended exactly once
    assert wrapped.close.call_count == 2
    spans = exporter.get_finished_spans()
    assert len(spans) == 1


def test_close_after_full_consumption_does_not_double_end(tracer_and_exporter):
    tracer, exporter = tracer_and_exporter
    chunk = {"model": "gpt-4", "id": "chatcmpl-x", "choices": []}
    wrapped = MagicMock()
    wrapped.__next__ = MagicMock(side_effect=[chunk, StopIteration])
    wrapped.close = MagicMock()

    stream = _make_stream(tracer, wrapped)
    with pytest.raises(StopIteration):
        while True:
            next(stream)

    # span already ended by normal completion; close() must stay safe
    stream.close()
    wrapped.close.assert_called_once_with()
    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].status.status_code == StatusCode.OK


def test_sync_context_manager_exit_ends_span(tracer_and_exporter):
    tracer, exporter = tracer_and_exporter
    wrapped = MagicMock()

    stream = _make_stream(tracer, wrapped)
    with stream:
        pass

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].end_time is not None


async def test_async_context_manager_exit_ends_span(tracer_and_exporter):
    tracer, exporter = tracer_and_exporter
    wrapped = MagicMock()

    stream = _make_stream(tracer, wrapped)
    async with stream:
        pass

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].end_time is not None
    assert spans[0].status.status_code == StatusCode.OK
