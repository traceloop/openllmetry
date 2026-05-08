from unittest.mock import MagicMock

import pytest
from opentelemetry.trace.status import StatusCode

from opentelemetry.instrumentation.together import _wrap


def _make_wrapper():
    tracer = MagicMock()
    span = MagicMock()
    span.is_recording.return_value = True
    tracer.start_span.return_value = span

    wrapper = _wrap(tracer, None, {"method": "chat.completions.ChatCompletions.create", "span_name": "together.chat"})
    return span, wrapper


def test_chat_api_error_marks_span_failed():
    span, wrapper = _make_wrapper()
    wrapped = MagicMock(side_effect=RuntimeError("401 Unauthorized"))

    with pytest.raises(RuntimeError, match="401 Unauthorized"):
        wrapper(wrapped, None, [], {"model": "test-model"})

    exc_arg = span.record_exception.call_args.args[0]
    assert isinstance(exc_arg, RuntimeError)
    assert "401 Unauthorized" in str(exc_arg)
    span.set_status.assert_called_once()
    span.end.assert_called_once()


def test_chat_api_error_records_exception_message():
    span, wrapper = _make_wrapper()
    wrapped = MagicMock(side_effect=ValueError("bad request"))

    with pytest.raises(ValueError, match="bad request"):
        wrapper(wrapped, None, [], {"model": "test-model"})

    status_arg = span.set_status.call_args.args[0]
    assert status_arg.status_code == StatusCode.ERROR
    assert status_arg.description == "bad request"
    span.end.assert_called_once()
