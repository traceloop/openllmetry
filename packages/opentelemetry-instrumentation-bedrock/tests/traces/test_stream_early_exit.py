"""Regression tests for https://github.com/traceloop/openllmetry/issues/4556.

The spans for `converse_stream` and `invoke_model_with_response_stream` must
be ended when the caller stops consuming the event stream early (`break`,
`close()`, or dropping the stream) — not only when the stream is read to the
end. Previously the span stayed open (and was never exported) in all three
early-exit cases.

Self-contained: a botocore `before-send` hook answers the requests with real
AWS event-stream framing, so no AWS account, network access, or VCR cassette
is needed.
"""

import base64
import gc
import json
import struct
import zlib

import boto3
import pytest
from botocore.awsrequest import AWSResponse
from botocore.config import Config

MODEL = "anthropic.claude-3-haiku-20240307-v1:0"


def _frame(event_type, payload):
    def _header(name, value):
        name, value = name.encode(), value.encode()
        return struct.pack("B", len(name)) + name + b"\x07" + struct.pack(">H", len(value)) + value

    headers = (
        _header(":event-type", event_type)
        + _header(":content-type", "application/json")
        + _header(":message-type", "event")
    )
    body = json.dumps(payload).encode()
    prelude = struct.pack(">II", 12 + len(headers) + len(body) + 4, len(headers))
    prelude += struct.pack(">I", zlib.crc32(prelude))
    message = prelude + headers + body
    return message + struct.pack(">I", zlib.crc32(message))


CONVERSE_EVENTS = [
    ("messageStart", {"role": "assistant"}),
    ("contentBlockDelta", {"contentBlockIndex": 0, "delta": {"text": "one "}}),
    ("contentBlockDelta", {"contentBlockIndex": 0, "delta": {"text": "two"}}),
    ("contentBlockStop", {"contentBlockIndex": 0}),
    ("messageStop", {"stopReason": "end_turn"}),
    (
        "metadata",
        {
            "usage": {"inputTokens": 3, "outputTokens": 2, "totalTokens": 5},
            "metrics": {"latencyMs": 1},
        },
    ),
]

INVOKE_EVENTS = [
    {
        "type": "message_start",
        "message": {
            "id": "m",
            "type": "message",
            "role": "assistant",
            "model": "m",
            "content": [],
            "usage": {"input_tokens": 3, "output_tokens": 0},
        },
    },
    {
        "type": "content_block_start",
        "index": 0,
        "content_block": {"type": "text", "text": ""},
    },
    {
        "type": "content_block_delta",
        "index": 0,
        "delta": {"type": "text_delta", "text": "hello"},
    },
    {"type": "content_block_stop", "index": 0},
    {
        "type": "message_delta",
        "delta": {"stop_reason": "end_turn"},
        "usage": {"output_tokens": 1},
    },
    {
        "type": "message_stop",
        "amazon-bedrock-invocationMetrics": {
            "inputTokenCount": 3,
            "outputTokenCount": 1,
            "invocationLatency": 1,
            "firstByteLatency": 1,
        },
    },
]


class _RawStream:
    """Minimal stand-in for botocore's raw HTTP stream."""

    def __init__(self, frames):
        self.frames = frames

    def stream(self, *args, **kwargs):
        yield from self.frames

    def read(self, amt=None):
        return b"".join(self.frames)

    def close(self):
        pass

    release_conn = close


def _before_send(request, **kwargs):
    if request.url.endswith("/converse-stream"):
        frames = [_frame(event_type, payload) for event_type, payload in CONVERSE_EVENTS]
    else:
        frames = [
            _frame(
                "chunk",
                {"bytes": base64.b64encode(json.dumps(event).encode()).decode()},
            )
            for event in INVOKE_EVENTS
        ]
    return AWSResponse(
        request.url,
        200,
        {"content-type": "application/vnd.amazon.eventstream"},
        _RawStream(frames),
    )


@pytest.fixture
def instrumented_brt(span_exporter, tracer_provider):
    """Bedrock client with the before-send mock, instrumented with only a
    tracer provider (no logger provider), so response content lands on span
    attributes rather than log events."""
    from opentelemetry.instrumentation.bedrock import BedrockInstrumentor

    instrumentor = BedrockInstrumentor()
    instrumentor.instrument(tracer_provider=tracer_provider)
    client = boto3.client(
        "bedrock-runtime",
        region_name="us-east-1",
        aws_access_key_id="test",
        aws_secret_access_key="test",
        config=Config(retries={"total_max_attempts": 1}),
    )
    client.meta.events.register("before-send.bedrock-runtime.*", _before_send)
    yield client
    instrumentor.uninstrument()


MESSAGES = [{"role": "user", "content": [{"text": "hi"}]}]
INVOKE_BODY = json.dumps(
    {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 50,
        "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
    }
)


def _converse_stream(instrumented_brt):
    return instrumented_brt.converse_stream(modelId=MODEL, messages=MESSAGES)["stream"]


def _invoke_stream(instrumented_brt):
    return instrumented_brt.invoke_model_with_response_stream(modelId=MODEL, body=INVOKE_BODY)["body"]


def _finished_spans(span_exporter):
    return span_exporter.get_finished_spans()


def _assert_single_ended_span(span_exporter):
    spans = _finished_spans(span_exporter)
    assert len(spans) == 1
    return spans[0]


# --- invoke_model_with_response_stream ---


def test_invoke_model_stream_full_consumption_ends_span_once(instrumented_brt, span_exporter):
    for _ in _invoke_stream(instrumented_brt):
        pass
    gc.collect()
    _assert_single_ended_span(span_exporter)


def test_invoke_model_stream_break_ends_span(instrumented_brt, span_exporter):
    for _ in _invoke_stream(instrumented_brt):
        break
    gc.collect()
    _assert_single_ended_span(span_exporter)


def test_invoke_model_stream_close_ends_span(instrumented_brt, span_exporter):
    stream = _invoke_stream(instrumented_brt)
    next(iter(stream))
    stream.close()
    gc.collect()
    _assert_single_ended_span(span_exporter)


def test_invoke_model_stream_drop_ends_span(instrumented_brt, span_exporter):
    stream = _invoke_stream(instrumented_brt)
    next(iter(stream))
    del stream
    gc.collect()
    _assert_single_ended_span(span_exporter)


# --- converse_stream ---


def test_converse_stream_full_consumption_ends_span_once(instrumented_brt, span_exporter):
    for _ in _converse_stream(instrumented_brt):
        pass
    gc.collect()
    span = _assert_single_ended_span(span_exporter)
    output = span.attributes.get("gen_ai.output.messages")
    assert output is not None
    assert "one two" in output


def test_converse_stream_break_ends_span_and_records_partial_output(instrumented_brt, span_exporter):
    stream = _converse_stream(instrumented_brt)
    it = iter(stream)
    next(it)  # messageStart
    next(it)  # contentBlockDelta carrying "one "
    # abandon the iterator mid-stream
    del it
    gc.collect()
    span = _assert_single_ended_span(span_exporter)
    # The early-exit fallback flushes the partial response captured so far.
    output = span.attributes.get("gen_ai.output.messages")
    assert output is not None
    assert "one " in output


def test_converse_stream_close_ends_span(instrumented_brt, span_exporter):
    stream = _converse_stream(instrumented_brt)
    next(iter(stream))
    stream.close()
    gc.collect()
    _assert_single_ended_span(span_exporter)


def test_converse_stream_drop_ends_span(instrumented_brt, span_exporter):
    stream = _converse_stream(instrumented_brt)
    next(iter(stream))
    del stream
    gc.collect()
    _assert_single_ended_span(span_exporter)
