import pytest

from traceloop.sdk.evaluator.stream_client import SSEClient


class _FakeResponse:
    def __init__(self, status_code: int, body: bytes):
        self.status_code = status_code
        self._body = body

    async def aread(self) -> bytes:
        return self._body


@pytest.mark.asyncio
async def test_error_body_is_decoded_in_exception_message():
    client = SSEClient.__new__(SSEClient)
    response = _FakeResponse(500, b'{"error":"boom"}')

    with pytest.raises(Exception) as exc_info:
        await client._handle_sse_response(response)  # type: ignore[arg-type]

    message = str(exc_info.value)
    assert '{"error":"boom"}' in message
    assert "b'" not in message
    assert "500" in message


@pytest.mark.asyncio
async def test_error_body_with_invalid_utf8_does_not_raise():
    client = SSEClient.__new__(SSEClient)
    response = _FakeResponse(502, b"\xff\xfeinvalid")

    with pytest.raises(Exception) as exc_info:
        await client._handle_sse_response(response)  # type: ignore[arg-type]

    assert "502" in str(exc_info.value)
