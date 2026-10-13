"""Failure handling of `_buffer_request_body`, which buffers a file-like
`body` for invoke_model*. The success paths are covered by the nova tests."""

import io

import pytest

from opentelemetry.instrumentation.bedrock import _buffer_request_body


class _FailingBody(io.BytesIO):
    """Advances its cursor, then fails."""

    def read(self, *args):
        super().read(5)
        raise OSError("read failed")


def test_failed_read_restores_position_and_raises():
    body = _FailingBody(b"0123456789")
    kwargs = {"body": body}

    with pytest.raises(OSError):
        _buffer_request_body(kwargs)

    assert kwargs["body"] is body
    assert body.tell() == 0


def test_failed_read_on_unseekable_body_raises():
    class Unseekable:
        def read(self):
            raise OSError("read failed")

    with pytest.raises(OSError):
        _buffer_request_body({"body": Unseekable()})
