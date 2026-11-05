"""A stream that fails mid-iteration is a failed call, not a successful one.

The OpenAI SDK re-raises the error out of the iterator, so the caller sees the
failure; the span has to agree. ChatStream ends its span from the cleanup path
whenever iteration stops, and that path marks the span OK -- on an error that
must not be the last word.
"""

import time

import pytest
from opentelemetry.instrumentation.openai.shared.chat_wrappers import ChatStream
from opentelemetry.instrumentation.openai.utils import TRACELOOP_TRACE_CONTENT
from opentelemetry.trace import StatusCode


_ERROR_MESSAGE = "connection reset mid-stream: private prompt content"


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

    def __init__(self, error):
        self._delivered = False
        self._error = error

    def __next__(self):
        if not self._delivered:
            self._delivered = True
            return _first_chunk()
        raise self._error

    def __aiter__(self):
        return self

    async def __anext__(self):
        return self.__next__()


def _broken_chat_stream(tracer, error):
    span = tracer.start_span("openai.chat")
    return ChatStream(
        span,
        _BrokenStream(error),
        None,
        None,
        None,
        None,
        None,
        None,
        time.time(),
        {"model": "gpt-4"},
    )


def _assert_failure_recorded(span_exporter, content_enabled):
    spans = [s for s in span_exporter.get_finished_spans() if s.name == "openai.chat"]
    assert len(spans) == 1, f"expected one chat span, got {[s.name for s in spans]}"
    span = spans[0]

    assert span.status.status_code is StatusCode.ERROR, (
        "a stream that raised was exported as "
        f"{span.status.status_code}: the failed call reads as a success"
    )
    assert span.attributes.get("error.type") == "RuntimeError"
    exception_events = [event for event in span.events if event.name == "exception"]
    if content_enabled:
        assert len(exception_events) == 1, "the failure left no exception event"
        assert exception_events[0].attributes["exception.message"] == _ERROR_MESSAGE
        assert span.status.description == _ERROR_MESSAGE
    else:
        assert not exception_events, "exception details bypassed content tracing"
        assert span.status.description is None
        assert _ERROR_MESSAGE not in span.to_json()
        assert "partial" not in span.to_json()


@pytest.fixture(params=[None, "true", "false"], ids=["default", "enabled", "disabled"])
def content_enabled(request, monkeypatch):
    if request.param is None:
        monkeypatch.delenv(TRACELOOP_TRACE_CONTENT, raising=False)
    else:
        monkeypatch.setenv(TRACELOOP_TRACE_CONTENT, request.param)
    return request.param != "false"


def test_sync_stream_failure_is_recorded_as_error(
    span_exporter, tracer_provider, content_enabled
):
    error = RuntimeError(_ERROR_MESSAGE)
    stream = _broken_chat_stream(tracer_provider.get_tracer(__name__), error)

    with pytest.raises(RuntimeError) as exc_info:
        for _ in stream:
            pass

    assert exc_info.value is error
    _assert_failure_recorded(span_exporter, content_enabled)


@pytest.mark.asyncio
async def test_async_stream_failure_is_recorded_as_error(
    span_exporter, tracer_provider, content_enabled
):
    error = RuntimeError(_ERROR_MESSAGE)
    stream = _broken_chat_stream(tracer_provider.get_tracer(__name__), error)

    with pytest.raises(RuntimeError) as exc_info:
        async for _ in stream:
            pass

    assert exc_info.value is error
    _assert_failure_recorded(span_exporter, content_enabled)
