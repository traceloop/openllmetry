"""A stream that fails mid-iteration is a failed call, not a successful one.

The OpenAI SDK re-raises the error out of the iterator, so the caller sees the
failure; the span has to agree. ChatStream ends its span from the cleanup path
whenever iteration stops, and that path marks the span OK -- on an error that
must not be the last word.
"""

import time

import pytest
from opentelemetry.instrumentation.openai.shared.chat_wrappers import ChatStream
from opentelemetry.trace import StatusCode


def _first_chunk():
    return {
        "id": "chatcmpl-broken",
        "model": "gpt-4",
        "choices": [
            {"index": 0, "delta": {"content": "partial"}, "finish_reason": None}
        ],
    }


class _BrokenStream:
    """Yields one chunk, then fails the way a dropped connection does."""

    def __init__(self):
        self._delivered = False

    def __next__(self):
        if not self._delivered:
            self._delivered = True
            return _first_chunk()
        raise RuntimeError("connection reset mid-stream")

    def __aiter__(self):
        return self

    async def __anext__(self):
        return self.__next__()


def _broken_chat_stream(tracer):
    span = tracer.start_span("openai.chat")
    return ChatStream(
        span,
        _BrokenStream(),
        None,
        None,
        None,
        None,
        None,
        None,
        time.time(),
        {"model": "gpt-4"},
    )


def _assert_failure_recorded(span_exporter):
    spans = [s for s in span_exporter.get_finished_spans() if s.name == "openai.chat"]
    assert len(spans) == 1, f"expected one chat span, got {[s.name for s in spans]}"
    span = spans[0]

    assert span.status.status_code is StatusCode.ERROR, (
        "a stream that raised was exported as "
        f"{span.status.status_code}: the failed call reads as a success"
    )
    assert span.attributes.get("error.type") == "RuntimeError"
    assert any(event.name == "exception" for event in span.events), (
        "the failure left no exception event"
    )


def test_sync_stream_failure_is_recorded_as_error(span_exporter, tracer_provider):
    stream = _broken_chat_stream(tracer_provider.get_tracer(__name__))

    with pytest.raises(RuntimeError):
        for _ in stream:
            pass

    _assert_failure_recorded(span_exporter)


@pytest.mark.asyncio
async def test_async_stream_failure_is_recorded_as_error(
    span_exporter, tracer_provider
):
    stream = _broken_chat_stream(tracer_provider.get_tracer(__name__))

    with pytest.raises(RuntimeError):
        async for _ in stream:
            pass

    _assert_failure_recorded(span_exporter)
