"""A stream must not write into the shared default message."""

from unittest.mock import Mock

import pytest

from opentelemetry.instrumentation.cohere.streaming import (
    DEFAULT_MESSAGE,
    process_chat_v2_streaming_response,
)


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

    response_attributes = [
        call.args[1]
        for call in second_span.set_attribute.call_args_list
        if call.args and "first answer" in str(call.args[1])
    ]
    assert response_attributes == []
