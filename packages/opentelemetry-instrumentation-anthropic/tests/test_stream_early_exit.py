"""Regression tests for https://github.com/traceloop/openllmetry/issues/4536.

Anthropic streaming spans were never ended on early exit: ``AnthropicStream``
and ``AnthropicAsyncStream`` only completed instrumentation on stream
exhaustion, and the ``WrappedMessageStreamManager`` wrappers never ran
cleanup on ``__exit__``/``__aexit__``. The wrapped streams are mocked so no
network access or API key is needed.
"""

import gc
import time
from unittest.mock import AsyncMock, MagicMock

import pytest
from opentelemetry.instrumentation.anthropic.streaming import (
    AnthropicAsyncStream,
    AnthropicStream,
    WrappedAsyncMessageStreamManager,
    WrappedMessageStreamManager,
)
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


def _ping_event():
    # An event type the instrumentation ignores, so iteration is side-effect
    # free apart from span completion.
    event = MagicMock()
    event.type = "ping"
    return event


def _make_sync_stream(tracer, wrapped):
    return AnthropicStream(
        tracer.start_span("anthropic.chat"), wrapped, instance=None,
        start_time=time.time(),
    )


def _make_async_stream(tracer, wrapped):
    return AnthropicAsyncStream(
        tracer.start_span("anthropic.chat"), wrapped, instance=None,
        start_time=time.time(),
    )


def _assert_single_finished_span(exporter):
    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "anthropic.chat"
    assert spans[0].end_time is not None
    assert spans[0].status.status_code == StatusCode.OK


def test_sync_close_ends_span(tracer_and_exporter):
    tracer, exporter = tracer_and_exporter
    wrapped = MagicMock()
    wrapped.__next__ = MagicMock(side_effect=[_ping_event()])
    wrapped.close = MagicMock()

    stream = _make_sync_stream(tracer, wrapped)
    next(stream)  # partial consumption
    stream.close()

    wrapped.close.assert_called_once_with()
    _assert_single_finished_span(exporter)


def test_sync_exit_ends_span(tracer_and_exporter):
    tracer, exporter = tracer_and_exporter
    wrapped = MagicMock()
    wrapped.__next__ = MagicMock(side_effect=[_ping_event()])

    stream = _make_sync_stream(tracer, wrapped)
    with stream:
        next(stream)  # leave the with-block early

    wrapped.__exit__.assert_called_once()
    _assert_single_finished_span(exporter)


def test_sync_del_ends_span(tracer_and_exporter):
    tracer, exporter = tracer_and_exporter
    wrapped = MagicMock()
    wrapped.__next__ = MagicMock(side_effect=[_ping_event()])

    stream = _make_sync_stream(tracer, wrapped)
    next(stream)
    del stream
    gc.collect()

    _assert_single_finished_span(exporter)


def test_sync_close_is_idempotent(tracer_and_exporter):
    tracer, exporter = tracer_and_exporter
    wrapped = MagicMock()
    wrapped.close = MagicMock()

    stream = _make_sync_stream(tracer, wrapped)
    stream.close()
    stream.close()

    # underlying close still delegates, but the span is ended exactly once
    assert wrapped.close.call_count == 2
    _assert_single_finished_span(exporter)


def test_sync_close_after_full_consumption_does_not_double_end(
    tracer_and_exporter,
):
    tracer, exporter = tracer_and_exporter
    wrapped = MagicMock()
    wrapped.__next__ = MagicMock(side_effect=[_ping_event(), StopIteration])
    wrapped.close = MagicMock()

    stream = _make_sync_stream(tracer, wrapped)
    with pytest.raises(StopIteration):
        while True:
            next(stream)

    # span already ended by normal completion; close() must stay safe
    stream.close()
    wrapped.close.assert_called_once_with()
    _assert_single_finished_span(exporter)


@pytest.mark.asyncio
async def test_async_close_ends_span(tracer_and_exporter):
    tracer, exporter = tracer_and_exporter
    closed = []

    async def aclose():
        closed.append(True)

    wrapped = MagicMock()
    wrapped.__anext__ = AsyncMock(side_effect=[_ping_event()])
    wrapped.close = aclose

    stream = _make_async_stream(tracer, wrapped)
    await stream.__anext__()  # partial consumption
    await stream.close()

    assert closed == [True]
    _assert_single_finished_span(exporter)


@pytest.mark.asyncio
async def test_async_aexit_ends_span(tracer_and_exporter):
    tracer, exporter = tracer_and_exporter
    wrapped = MagicMock()
    wrapped.__anext__ = AsyncMock(side_effect=[_ping_event()])
    wrapped.__aexit__ = AsyncMock(return_value=False)

    stream = _make_async_stream(tracer, wrapped)
    async with stream:
        await stream.__anext__()  # leave the async with-block early

    wrapped.__aexit__.assert_awaited_once()
    _assert_single_finished_span(exporter)


def test_message_stream_manager_exit_ends_span(tracer_and_exporter):
    """`with client.messages.stream(...)` early exit ends the span."""
    tracer, exporter = tracer_and_exporter
    stream_mock = MagicMock()
    stream_mock.__next__ = MagicMock(side_effect=[_ping_event()])
    manager_mock = MagicMock()
    manager_mock.__enter__ = MagicMock(return_value=stream_mock)

    manager = WrappedMessageStreamManager(
        manager_mock, tracer.start_span("anthropic.chat"), instance=None,
        start_time=time.time(), token_histogram=None, choice_counter=None,
        duration_histogram=None, exception_counter=None, event_logger=None,
        kwargs={},
    )
    with manager as stream:
        next(stream)  # break out of text_stream early

    manager_mock.__exit__.assert_called_once()
    _assert_single_finished_span(exporter)


@pytest.mark.asyncio
async def test_async_message_stream_manager_aexit_ends_span(tracer_and_exporter):
    tracer, exporter = tracer_and_exporter
    stream_mock = MagicMock()
    stream_mock.__anext__ = AsyncMock(side_effect=[_ping_event()])
    manager_mock = MagicMock()
    manager_mock.__aenter__ = AsyncMock(return_value=stream_mock)
    manager_mock.__aexit__ = AsyncMock(return_value=False)

    manager = WrappedAsyncMessageStreamManager(
        manager_mock, tracer.start_span("anthropic.chat"), instance=None,
        start_time=time.time(), token_histogram=None, choice_counter=None,
        duration_histogram=None, exception_counter=None, event_logger=None,
        kwargs={},
    )
    async with manager as stream:
        await stream.__anext__()

    manager_mock.__aexit__.assert_awaited_once()
    _assert_single_finished_span(exporter)


def _assert_single_error_span(exporter):
    """Exactly one finished span, ERROR status, exception recorded on it."""
    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "anthropic.chat"
    assert span.end_time is not None
    assert span.status.status_code == StatusCode.ERROR
    assert span.status.description == "boom"
    exc_events = [e for e in span.events if e.name == "exception"]
    assert len(exc_events) == 1
    assert exc_events[0].attributes["exception.type"] == "ValueError"
    assert exc_events[0].attributes["exception.message"] == "boom"


def test_sync_exit_with_exception_marks_span_error(tracer_and_exporter):
    """An exception escaping the with-block must not be recorded as OK."""
    tracer, exporter = tracer_and_exporter
    wrapped = MagicMock()
    wrapped.__next__ = MagicMock(side_effect=[_ping_event()])
    wrapped.__exit__ = MagicMock(return_value=False)

    stream = _make_sync_stream(tracer, wrapped)
    with pytest.raises(ValueError, match="boom"):
        with stream:
            next(stream)
            raise ValueError("boom")

    wrapped.__exit__.assert_called_once()
    _assert_single_error_span(exporter)

    # cleanup is idempotent: a later close() must not touch the ended span
    stream.close()
    assert len(exporter.get_finished_spans()) == 1


@pytest.mark.asyncio
async def test_async_aexit_with_exception_marks_span_error(tracer_and_exporter):
    tracer, exporter = tracer_and_exporter
    wrapped = MagicMock()
    wrapped.__anext__ = AsyncMock(side_effect=[_ping_event()])
    wrapped.__aexit__ = AsyncMock(return_value=False)

    stream = _make_async_stream(tracer, wrapped)
    with pytest.raises(ValueError, match="boom"):
        async with stream:
            await stream.__anext__()
            raise ValueError("boom")

    wrapped.__aexit__.assert_awaited_once()
    _assert_single_error_span(exporter)


def test_message_stream_manager_exit_with_exception_marks_span_error(
    tracer_and_exporter,
):
    """`with client.messages.stream(...)` raising inside marks the span ERROR."""
    tracer, exporter = tracer_and_exporter
    stream_mock = MagicMock()
    stream_mock.__next__ = MagicMock(side_effect=[_ping_event()])
    manager_mock = MagicMock()
    manager_mock.__enter__ = MagicMock(return_value=stream_mock)
    manager_mock.__exit__ = MagicMock(return_value=False)

    manager = WrappedMessageStreamManager(
        manager_mock, tracer.start_span("anthropic.chat"), instance=None,
        start_time=time.time(), token_histogram=None, choice_counter=None,
        duration_histogram=None, exception_counter=None, event_logger=None,
        kwargs={},
    )
    with pytest.raises(ValueError, match="boom"):
        with manager as stream:
            next(stream)
            raise ValueError("boom")

    manager_mock.__exit__.assert_called_once()
    _assert_single_error_span(exporter)


@pytest.mark.asyncio
async def test_async_message_stream_manager_aexit_with_exception_marks_span_error(
    tracer_and_exporter,
):
    tracer, exporter = tracer_and_exporter
    stream_mock = MagicMock()
    stream_mock.__anext__ = AsyncMock(side_effect=[_ping_event()])
    manager_mock = MagicMock()
    manager_mock.__aenter__ = AsyncMock(return_value=stream_mock)
    manager_mock.__aexit__ = AsyncMock(return_value=False)

    manager = WrappedAsyncMessageStreamManager(
        manager_mock, tracer.start_span("anthropic.chat"), instance=None,
        start_time=time.time(), token_histogram=None, choice_counter=None,
        duration_histogram=None, exception_counter=None, event_logger=None,
        kwargs={},
    )
    with pytest.raises(ValueError, match="boom"):
        async with manager as stream:
            await stream.__anext__()
            raise ValueError("boom")

    manager_mock.__aexit__.assert_awaited_once()
    _assert_single_error_span(exporter)
