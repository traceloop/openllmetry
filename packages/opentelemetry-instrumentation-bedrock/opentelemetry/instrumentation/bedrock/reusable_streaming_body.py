from botocore.response import StreamingBody
from botocore.exceptions import (
    ReadTimeoutError,
    ResponseStreamingError,
)
from urllib3.exceptions import ProtocolError as URLLib3ProtocolError
from urllib3.exceptions import ReadTimeoutError as URLLib3ReadTimeoutError


class ReusableStreamingBody(StreamingBody):
    """Wrapper around StreamingBody that allows the body to be read multiple times."""

    def __init__(self, raw_stream, content_length):
        super().__init__(raw_stream, content_length)
        self._buffer = None
        self._buffer_cursor = 0

    def read(self, amt=None):
        """Read at most amt bytes from the stream.

        If the amt argument is omitted, read all data.
        """
        if self._buffer is None:
            try:
                self._buffer = self._raw_stream.read()
            except URLLib3ReadTimeoutError as e:
                # TODO: the url will be None as urllib3 isn't setting it yet
                raise ReadTimeoutError(endpoint_url=e.url, error=e)
            except URLLib3ProtocolError as e:
                raise ResponseStreamingError(error=e)

            self._amount_read += len(self._buffer)
            if amt is None or (not self._buffer and amt > 0):
                # If the server sends empty contents or
                # we ask to read all of the contents, then we know
                # we need to verify the content length.
                self._verify_content_length()

        if amt is None:
            return self._buffer[self._buffer_cursor:]
        else:
            self._buffer_cursor += amt
            return self._buffer[self._buffer_cursor-amt:self._buffer_cursor]


class BufferedAsyncBody:
    """Async-readable replacement for aiobotocore StreamingBody whose contents
    have already been read. Allows the user to `await body.read()` after our
    instrumentation already consumed the bytes once.

    Uses an immutable buffer + cursor (matching ReusableStreamingBody's sync
    semantics) so a partial `read(n)` followed by `read()` returns the rest,
    and the buffer can be re-read by resetting the cursor."""

    def __init__(self, buffer: bytes):
        self._buffer = buffer
        self._cursor = 0

    async def read(self, amt=None):
        if amt is None:
            return self._buffer[self._cursor:]
        out = self._buffer[self._cursor:self._cursor + amt]
        self._cursor += amt
        return out

    async def close(self):
        # No-op: the bytes are already buffered, no underlying resource to release.
        return None
