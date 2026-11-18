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


async def test_async_aclose_ends_span(tracer_and_exporter):
    """Regression: SDK 3.x ``aclose()`` must also end the span.

    OpenAI SDK 3.x added ``AsyncStream.aclose()`` (2.x has no such method).
    Before the fix, ``await stream.aclose()`` delegated straight through the
    wrapt proxy to the wrapped stream, so the span stayed open until GC.
    """
    tracer, exporter = tracer_and_exporter
    closed = []

    async def aclose():
        closed.append(True)

    wrapped = MagicMock()
    wrapped.aclose = aclose

    stream = _make_stream(tracer, wrapped)
    await stream.aclose()

    assert closed == [True]
    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].end_time is not None
    assert spans[0].status.status_code == StatusCode.OK


async def test_aclose_without_wrapped_aclose_still_ends_span(tracer_and_exporter):
    """SDK 2.x async streams have no ``aclose``; span cleanup must still run."""
    tracer, exporter = tracer_and_exporter
    wrapped = MagicMock()
    del wrapped.aclose

    stream = _make_stream(tracer, wrapped)
    await stream.aclose()

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].end_time is not None
    assert spans[0].status.status_code == StatusCode.OK


def test_close_does_not_swallow_base_exception_from_cleanup(tracer_and_exporter):
    """Regression: no ``return`` inside the ``finally`` of ``close()``.

    A ``return`` in ``finally`` swallows any BaseException (e.g.
    KeyboardInterrupt) escaping ``_ensure_cleanup()``. ``close()`` must let
    it propagate while still delegating to the wrapped stream's close.
    """
    from unittest.mock import patch

    tracer, exporter = tracer_and_exporter
    wrapped = MagicMock()
    wrapped.close = MagicMock(return_value="closed")

    stream = _make_stream(tracer, wrapped)
    with patch.object(
        ChatStream, "_ensure_cleanup", side_effect=KeyboardInterrupt
    ):
        with pytest.raises(KeyboardInterrupt):
            stream.close()

    # the wrapped close still ran in the finally block before propagating
    wrapped.close.assert_called_once_with()


def test_close_returns_wrapped_close_result(tracer_and_exporter):
    """The finally restructure must preserve the delegated return value."""
    tracer, exporter = tracer_and_exporter
    wrapped = MagicMock()
    wrapped.close = MagicMock(return_value="closed")

    stream = _make_stream(tracer, wrapped)
    assert stream.close() == "closed"
    wrapped.close.assert_called_once_with()


def test_close_without_wrapped_close_returns_none(tracer_and_exporter):
    tracer, exporter = tracer_and_exporter
    wrapped = MagicMock()
    del wrapped.close

    stream = _make_stream(tracer, wrapped)
    assert stream.close() is None
