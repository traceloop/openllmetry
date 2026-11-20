import logging

import pytest

from opentelemetry.instrumentation.haystack.utils import (
    dont_throw,
    process_request,
    process_response,
)

LOGGER_NAME = "opentelemetry.instrumentation.haystack.utils"


class _ExplodingSpan:
    def set_attribute(self, *args, **kwargs):
        raise RuntimeError("span attribute serialization exploded")


@pytest.fixture(autouse=True)
def enable_content_capture(monkeypatch):
    monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", "true")


def test_dont_throw_logs_the_full_traceback(caplog):
    @dont_throw
    def boom():
        raise ValueError("tracing exploded")

    # dont_throw logs to the wrapped function's module logger; boom lives here.
    with caplog.at_level(logging.DEBUG):
        boom()

    assert "Traceback (most recent call last)" in caplog.text
    assert "tracing exploded" in caplog.text


def test_process_request_logs_the_full_traceback(caplog):
    with caplog.at_level(logging.DEBUG, logger=LOGGER_NAME):
        process_request(_ExplodingSpan(), (), {})

    assert "Traceback (most recent call last)" in caplog.text


def test_process_response_logs_the_full_traceback(caplog):
    with caplog.at_level(logging.DEBUG, logger=LOGGER_NAME):
        process_response(_ExplodingSpan(), {"replies": ["hi"]})

    assert "Traceback (most recent call last)" in caplog.text
