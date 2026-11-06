"""A stream the caller stops reading early still ends its span."""

import time

import pytest
from opentelemetry.instrumentation.agno.streaming import AgnoAsyncStream, AgnoStream
from opentelemetry.trace import StatusCode


def _events():
    yield "first"
    yield "second"


async def _aevents():
    yield "first"
    yield "second"


def test_sync_stream_break_ends_span(tracer_provider, span_exporter):
    span = tracer_provider.get_tracer(__name__).start_span("agno.agent.test")
    stream = AgnoStream(span, _events(), None, time.time())

    for _ in stream:
        break

    assert [s.name for s in span_exporter.get_finished_spans()] == ["agno.agent.test"]


def test_sync_stream_close_ends_span(tracer_provider, span_exporter):
    span = tracer_provider.get_tracer(__name__).start_span("agno.agent.test")
    events = _events()
    stream = AgnoStream(span, events, None, time.time())

    next(stream)
    stream.close()

    assert [s.name for s in span_exporter.get_finished_spans()] == ["agno.agent.test"]
    assert list(events) == []


@pytest.mark.asyncio
async def test_async_stream_aclose_ends_span(tracer_provider, span_exporter):
    span = tracer_provider.get_tracer(__name__).start_span("agno.agent.test")
    events = _aevents()
    stream = AgnoAsyncStream(span, events, None, time.time())

    async for _ in stream:
        break
    await stream.aclose()

    assert [s.name for s in span_exporter.get_finished_spans()] == ["agno.agent.test"]


def _failing_cleanup():
    try:
        yield "first"
    finally:
        raise RuntimeError("cleanup failed")


async def _afailing_cleanup():
    try:
        yield "first"
    finally:
        raise RuntimeError("cleanup failed")


def test_sync_stream_close_error_marks_span(tracer_provider, span_exporter):
    span = tracer_provider.get_tracer(__name__).start_span("agno.agent.test")
    stream = AgnoStream(span, _failing_cleanup(), None, time.time())

    next(stream)
    with pytest.raises(RuntimeError, match="cleanup failed"):
        stream.close()

    (finished,) = span_exporter.get_finished_spans()
    assert finished.status.status_code == StatusCode.ERROR
    assert [e.name for e in finished.events] == ["exception"]


@pytest.mark.asyncio
async def test_async_stream_aclose_error_marks_span(tracer_provider, span_exporter):
    span = tracer_provider.get_tracer(__name__).start_span("agno.agent.test")
    stream = AgnoAsyncStream(span, _afailing_cleanup(), None, time.time())

    await stream.__anext__()
    with pytest.raises(RuntimeError, match="cleanup failed"):
        await stream.aclose()

    (finished,) = span_exporter.get_finished_spans()
    assert finished.status.status_code == StatusCode.ERROR
    assert [e.name for e in finished.events] == ["exception"]
