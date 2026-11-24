import httpx
import pytest
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from writerai.types.chat_completion_chunk import (
    ChatCompletionChunk,
    Choice,
    ChoiceDelta,
)

from opentelemetry.instrumentation.writer import (
    _create_async_stream_processor,
    _create_stream_processor,
)

FINISH_REASONS = GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS
RESPONSE_ID = GenAIAttributes.GEN_AI_RESPONSE_ID


class FakeStream:
    def __init__(self, chunks, error=None):
        self._chunks = chunks
        self._error = error
        self.response = httpx.Response(
            200,
            request=httpx.Request("POST", "https://api.writer.com/v1/chat"),
        )

    def __iter__(self):
        yield from self._chunks
        if self._error is not None:
            raise self._error


class FakeAsyncStream:
    def __init__(self, chunks, error=None):
        self._chunks = chunks
        self._error = error
        self.response = httpx.Response(
            200,
            request=httpx.Request("POST", "https://api.writer.com/v1/chat"),
        )

    def __aiter__(self):
        return self._iterate()

    async def _iterate(self):
        for chunk in self._chunks:
            yield chunk
        if self._error is not None:
            raise self._error

    async def aclose(self):
        return None


def make_chunk(content=None, finish_reason=None, index=0):
    return ChatCompletionChunk(
        id="chunk-id-1",
        created=1,
        model="palmyra-x4",
        object="chat.completion.chunk",
        choices=[
            Choice(
                index=index,
                finish_reason=finish_reason,
                delta=ChoiceDelta(content=content, role="assistant"),
            )
        ],
    )


def make_processor(tracer_provider, chunks, error=None, event_logger=None):
    span = tracer_provider.get_tracer(__name__).start_span("writerai.chat")
    processor = _create_stream_processor(
        FakeStream(chunks, error),
        span,
        event_logger,
        0.0,
        None,
        None,
        None,
        None,
        "chat",
    )
    return processor


def make_async_processor(tracer_provider, chunks, error=None, event_logger=None):
    span = tracer_provider.get_tracer(__name__).start_span("writerai.chat")
    return _create_async_stream_processor(
        FakeAsyncStream(chunks, error),
        span,
        event_logger,
        0.0,
        None,
        None,
        None,
        None,
        "chat",
    )


def choice_events(log_exporter):
    return [
        log.log_record
        for log in log_exporter.get_finished_logs()
        if log.log_record.event_name == "gen_ai.choice"
    ]


def finished_span(span_exporter):
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    return spans[0]


def test_stream_with_length_finish_reason(tracer_provider, span_exporter):
    chunks = [make_chunk("Hel"), make_chunk("lo", finish_reason="length")]
    assert len(list(make_processor(tracer_provider, chunks))) == 2

    span = finished_span(span_exporter)
    assert span.attributes.get(FINISH_REASONS) == ("length",)
    assert span.attributes.get(RESPONSE_ID) == "chunk-id-1"


def test_stream_with_tool_calls_finish_reason(tracer_provider, span_exporter):
    chunks = [make_chunk("a"), make_chunk(None, finish_reason="tool_calls")]
    list(make_processor(tracer_provider, chunks))

    span = finished_span(span_exporter)
    assert span.attributes.get(FINISH_REASONS) == ("tool_calls",)


def test_stream_without_finish_reason(tracer_provider, span_exporter):
    chunks = [make_chunk("Hel"), make_chunk("lo")]
    list(make_processor(tracer_provider, chunks))

    span = finished_span(span_exporter)
    assert FINISH_REASONS not in span.attributes
    assert span.attributes.get(RESPONSE_ID) == "chunk-id-1"


def test_stream_cut_off_by_exception(tracer_provider, span_exporter):
    chunks = [make_chunk("Hel")]
    processor = make_processor(tracer_provider, chunks, error=RuntimeError("boom"))

    with pytest.raises(RuntimeError):
        list(processor)

    span = finished_span(span_exporter)
    assert FINISH_REASONS not in span.attributes


def test_stream_closed_early(tracer_provider, span_exporter):
    chunks = [make_chunk("Hel"), make_chunk("lo", finish_reason="length")]
    processor = make_processor(tracer_provider, chunks)

    assert next(processor).choices[0].delta.content == "Hel"
    processor.close()

    span = finished_span(span_exporter)
    assert FINISH_REASONS not in span.attributes


def test_stream_with_no_chunks(tracer_provider, span_exporter):
    list(make_processor(tracer_provider, []))

    span = finished_span(span_exporter)
    assert FINISH_REASONS not in span.attributes
    assert RESPONSE_ID not in span.attributes


def test_stream_first_chunk_with_nonzero_index(tracer_provider, span_exporter):
    chunks = [make_chunk("x", index=1, finish_reason="stop")]
    list(make_processor(tracer_provider, chunks))

    span = finished_span(span_exporter)
    assert span.attributes.get(FINISH_REASONS) == ("stop",)


def test_stream_without_finish_reason_legacy_completion_attribute(
    instrument_legacy, tracer_provider, span_exporter
):
    chunks = [make_chunk("Hel"), make_chunk("lo")]
    list(make_processor(tracer_provider, chunks))

    span = finished_span(span_exporter)
    assert f"{GenAIAttributes.GEN_AI_COMPLETION}.0.finish_reason" not in span.attributes
    assert FINISH_REASONS not in span.attributes
    assert span.attributes.get(f"{GenAIAttributes.GEN_AI_COMPLETION}.0.content") == "Hello"


def test_stream_without_finish_reason_events_mode(
    instrument_with_content, tracer_provider, logger_provider, log_exporter
):
    chunks = [make_chunk("Hel"), make_chunk("lo")]
    event_logger = logger_provider.get_logger(__name__)
    list(make_processor(tracer_provider, chunks, event_logger=event_logger))

    events = choice_events(log_exporter)
    assert len(events) == 1
    assert events[0].body["finish_reason"] == "unknown"


@pytest.mark.parametrize("reason", ["stop", "length"])
def test_stream_with_finish_reason_events_mode(
    reason, instrument_with_content, tracer_provider, logger_provider, log_exporter
):
    chunks = [make_chunk("Hel"), make_chunk("lo", finish_reason=reason)]
    event_logger = logger_provider.get_logger(__name__)
    list(make_processor(tracer_provider, chunks, event_logger=event_logger))

    events = choice_events(log_exporter)
    assert len(events) == 1
    assert events[0].body["finish_reason"] == reason


@pytest.mark.asyncio
async def test_async_stream_without_finish_reason(
    instrument_legacy, tracer_provider, span_exporter
):
    chunks = [make_chunk("Hel"), make_chunk("lo")]
    received = [chunk async for chunk in make_async_processor(tracer_provider, chunks)]
    assert len(received) == 2

    span = finished_span(span_exporter)
    assert FINISH_REASONS not in span.attributes
    assert f"{GenAIAttributes.GEN_AI_COMPLETION}.0.finish_reason" not in span.attributes
    assert span.attributes.get(RESPONSE_ID) == "chunk-id-1"


@pytest.mark.asyncio
async def test_async_stream_with_length_finish_reason(
    instrument_legacy, tracer_provider, span_exporter
):
    chunks = [make_chunk("Hel"), make_chunk("lo", finish_reason="length")]
    received = [chunk async for chunk in make_async_processor(tracer_provider, chunks)]
    assert len(received) == 2

    span = finished_span(span_exporter)
    assert span.attributes.get(FINISH_REASONS) == ("length",)
    assert span.attributes.get(f"{GenAIAttributes.GEN_AI_COMPLETION}.0.finish_reason") == "length"
    assert span.attributes.get(RESPONSE_ID) == "chunk-id-1"


@pytest.mark.asyncio
async def test_async_stream_without_finish_reason_events_mode(
    instrument_with_content, tracer_provider, logger_provider, log_exporter
):
    chunks = [make_chunk("Hel"), make_chunk("lo")]
    event_logger = logger_provider.get_logger(__name__)
    async for _ in make_async_processor(tracer_provider, chunks, event_logger=event_logger):
        pass

    events = choice_events(log_exporter)
    assert len(events) == 1
    assert events[0].body["finish_reason"] == "unknown"
