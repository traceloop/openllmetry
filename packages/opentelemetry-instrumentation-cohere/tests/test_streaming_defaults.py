"""A stream must not write into the shared default message."""

from unittest.mock import Mock

import pytest

from opentelemetry.instrumentation.cohere.streaming import (
    DEFAULT_MESSAGE,
    process_chat_v2_streaming_response,
)
from opentelemetry.instrumentation.cohere.utils import TRACELOOP_TRACE_CONTENT

COMPLETION_CONTENT = "gen_ai.completion.0.content"



def _stream(chunks):
    """The v2 stream items a chat with one text answer produces.

    A ``message-start`` carrying no ``delta.message`` is the fallback path: the
    processor has to build the message itself rather than adopt the one the
    server sent.
    """

    return list(chunks)


def _chunks(text):
    return [
        {"type": "message-start", "id": "abc", "index": 0, "delta": {}},
        {"type": "content-start", "index": 1, "delta": {"message": {"content": {}}}},
        {"type": "content-delta", "index": 1, "delta": {"message": {"content": {"text": text}}}},
        {"type": "content-end", "index": 1, "delta": {}},
        {"type": "message-end", "index": 4, "delta": {"finish_reason": "COMPLETE", "usage": {}}},
    ]


@pytest.fixture(autouse=True)
def _no_shared_default():
    """The module default has to be pristine before and after each case."""

    assert DEFAULT_MESSAGE["content"] == []
    assert DEFAULT_MESSAGE["tool_calls"] == []
    yield
    assert DEFAULT_MESSAGE["content"] == []
    assert DEFAULT_MESSAGE["tool_calls"] == []


@pytest.fixture(autouse=True)
def _content_is_traced(monkeypatch):
    """The content has to be emitted for the isolation check to mean anything.

    ``should_send_prompts`` reads the variable at call time, and an environment that
    sets it to ``false`` leaves every span without a completion attribute. The
    "first answer is absent" assertion would then hold for a stream that never wrote
    it -- a test that passes because nothing was compared.
    """

    monkeypatch.setenv(TRACELOOP_TRACE_CONTENT, "true")


def _completion_content(span):
    """The completion content the processor set on *span*, or None if it set none."""

    values = [
        call.args[1]
        for call in span.set_attribute.call_args_list
        if call.args and call.args[0] == COMPLETION_CONTENT
    ]
    return values[0] if values else None



def test_a_stream_does_not_write_into_the_module_default():
    """The fallback message is per stream: ``content-end`` appends into it."""

    span = Mock()
    list(process_chat_v2_streaming_response(span, Mock(), None, _chunks("hello")))

    assert DEFAULT_MESSAGE["content"] == []


def test_the_next_stream_does_not_inherit_the_previous_answer():
    """Two streams on one process: the second span must not carry the first text."""

    first_span = Mock()
    list(process_chat_v2_streaming_response(first_span, Mock(), None, _chunks("first answer")))

    second_span = Mock()
    list(process_chat_v2_streaming_response(second_span, Mock(), None, _chunks("second answer")))

    # The premise comes first: an absent completion attribute would satisfy the
    # assertion below without either answer having been observed.
    completion = _completion_content(second_span)
    assert completion is not None, "no completion content was emitted, so nothing was compared"
    assert "second answer" in str(completion)
    assert "first answer" not in str(completion)

