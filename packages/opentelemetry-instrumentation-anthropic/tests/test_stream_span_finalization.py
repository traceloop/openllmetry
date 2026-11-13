"""Every streamed call must end its span exactly once, whatever stops it.

Covers the two gaps in #4536 (early exit) and #4643 (a sync stream failing
mid-way) for both the sync and the async client, plus the normal path and the
SDK cleanup that must still run.
"""

import gc
import itertools

import pytest
from opentelemetry.instrumentation.anthropic import AnthropicInstrumentor
from opentelemetry.instrumentation.anthropic.event_emitter import emit_streaming_response_events
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk._logs.export import InMemoryLogRecordExporter, SimpleLogRecordProcessor
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.trace import StatusCode

import anthropic

try:
    import httpx2 as httpx
except ImportError:  # anthropic < 1.14 ships on httpx
    import httpx

MESSAGE_START = (
    "event: message_start\n"
    'data: {"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant",'
    '"model":"claude-sonnet-5-5","content":[],"stop_reason":null,"stop_sequence":null,'
    '"usage":{"input_tokens":1000,"output_tokens":0}}}\n\n'
)
CONTENT = (
    "event: content_block_start\n"
    'data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}\n\n'
    "event: content_block_delta\n"
    'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hi"}}\n\n'
    "event: content_block_stop\n"
    'data: {"type":"content_block_stop","index":0}\n\n'
)
MESSAGE_END = (
    "event: message_delta\n"
    'data: {"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},'
    '"usage":{"output_tokens":200}}\n\n'
    "event: message_stop\n"
    'data: {"type":"message_stop"}\n\n'
)
PARTIAL_TOOL = (
    MESSAGE_START
    + "event: content_block_start\n"
    'data: {"type":"content_block_start","index":0,"content_block":{"type":"tool_use",'
    '"id":"tu_1","name":"add","input":{}}}\n\n'
    + "event: content_block_delta\n"
    'data: {"type":"content_block_delta","index":0,'
    '"delta":{"type":"input_json_delta","partial_json":"{\\"a\\": 1"}}\n\n'
    # The block stops before the message does: the accumulated tool input is
    # the partial JSON above, which the event emitter must tolerate.
    + "event: content_block_stop\n"
    'data: {"type":"content_block_stop","index":0}\n\n'
)

STREAM_ERROR = 'event: error\ndata: {"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}\n\n'

COMPLETE = MESSAGE_START + CONTENT + MESSAGE_END
FAILED = MESSAGE_START + CONTENT + STREAM_ERROR

CALL = {
    "model": "claude-sonnet-5-5",
    "max_tokens": 16,
    "messages": [{"role": "user", "content": "hi"}],
}


class Served:
    """A mock transport that keeps the responses it hands to the SDK."""

    def __init__(self, body):
        self._body = body
        self.responses = []

    def __call__(self, request):
        response = httpx.Response(200, headers={"content-type": "text/event-stream"}, content=self._body)
        self.responses.append(response)
        return response


@pytest.fixture
def exporter():
    span_exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(span_exporter))
    # Event emission does the tool-input json.loads; without a logger it is
    # skipped, and the early-exit path would not be exercised at all.
    logger_provider = LoggerProvider()
    logger_provider.add_log_record_processor(SimpleLogRecordProcessor(InMemoryLogRecordExporter()))
    instrumentor = AnthropicInstrumentor()
    instrumentor.instrument(tracer_provider=provider, logger_provider=logger_provider)
    yield span_exporter
    instrumentor.uninstrument()
    span_exporter.clear()


def sync_client(served):
    return anthropic.Anthropic(api_key="test", http_client=httpx.Client(transport=httpx.MockTransport(served)))


def async_client(served):
    return anthropic.AsyncAnthropic(
        api_key="test",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(served)),
    )


def only_span(exporter):
    spans = exporter.get_finished_spans()
    assert len(spans) == 1, [span.name for span in spans]
    return spans[0]


def test_sync_stream_ends_span_on_early_exit(exporter):
    served = Served(COMPLETE)
    with sync_client(served).messages.stream(**CALL) as stream:
        next(iter(stream))

    assert only_span(exporter).status.status_code is StatusCode.OK
    assert served.responses[0].is_closed, "the SDK must still close the response"


def test_sync_stream_ends_span_when_the_stream_fails(exporter):
    served = Served(FAILED)
    with pytest.raises(anthropic.APIStatusError):
        with sync_client(served).messages.stream(**CALL) as stream:
            for _ in stream:
                pass

    assert only_span(exporter).status.status_code is StatusCode.ERROR
    assert served.responses[0].is_closed, "the SDK must still close the response"


def test_sync_stream_ends_span_on_completion(exporter):
    served = Served(COMPLETE)
    with sync_client(served).messages.stream(**CALL) as stream:
        for _ in stream:
            pass

    span = only_span(exporter)
    assert span.status.status_code is StatusCode.OK
    assert span.attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS] == 1000
    assert span.attributes[GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS] == 200


def test_sync_stream_ends_span_on_close(exporter):
    served = Served(COMPLETE)
    stream = sync_client(served).messages.create(stream=True, **CALL)
    next(iter(stream))
    stream.close()

    assert only_span(exporter).status.status_code is StatusCode.OK


def test_sync_stream_ends_span_when_dropped(exporter):
    served = Served(COMPLETE)
    stream = sync_client(served).messages.create(stream=True, **CALL)
    next(iter(stream))
    del stream
    gc.collect()

    assert only_span(exporter).status.status_code is StatusCode.OK


@pytest.mark.asyncio
async def test_async_stream_ends_span_on_early_exit(exporter):
    served = Served(COMPLETE)
    async with async_client(served).messages.stream(**CALL) as stream:
        await stream.__anext__()

    assert only_span(exporter).status.status_code is StatusCode.OK
    assert served.responses[0].is_closed, "the SDK must still close the response"


@pytest.mark.asyncio
async def test_async_stream_ends_span_when_the_stream_fails(exporter):
    served = Served(FAILED)
    with pytest.raises(anthropic.APIStatusError):
        async with async_client(served).messages.stream(**CALL) as stream:
            async for _ in stream:
                pass

    assert only_span(exporter).status.status_code is StatusCode.ERROR
    assert served.responses[0].is_closed, "the SDK must still close the response"


@pytest.mark.asyncio
async def test_async_stream_ends_span_on_completion(exporter):
    served = Served(COMPLETE)
    async with async_client(served).messages.stream(**CALL) as stream:
        async for _ in stream:
            pass

    span = only_span(exporter)
    assert span.status.status_code is StatusCode.OK
    assert span.attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS] == 1000
    assert span.attributes[GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS] == 200


def test_sync_stream_closed_mid_tool_use_ends_span(exporter):
    """Closing while a tool_use block streams must not raise on the partial input."""
    served = Served(PARTIAL_TOOL)
    with sync_client(served).messages.stream(**CALL) as stream:
        # Consume through content_block_stop, then leave before the message ends.
        for _ in itertools.islice(stream, 4):
            pass

    assert only_span(exporter).status.status_code is StatusCode.OK


@pytest.mark.asyncio
async def test_async_stream_closed_mid_tool_use_ends_span(exporter):
    served = Served(PARTIAL_TOOL)
    async with async_client(served).messages.stream(**CALL) as stream:
        consumed = 0
        async for _ in stream:
            consumed += 1
            if consumed == 4:
                break

    assert only_span(exporter).status.status_code is StatusCode.OK


def test_emit_streaming_response_events_tolerates_partial_tool_input():
    emit_streaming_response_events(None, {"events": [{"type": "tool_use", "input": '{"a": 1'}]})


def test_direct_stream_context_manager_keeps_the_proxy(exporter):
    served = Served(COMPLETE)
    with sync_client(served).messages.create(stream=True, **CALL) as stream:
        assert hasattr(stream, "_complete_instrumentation")
        for _ in stream:
            pass

    assert only_span(exporter).status.status_code is StatusCode.OK
