"""The transport wrapper must not change what the traced code raises."""

import asyncio
from contextlib import asynccontextmanager

import pytest
from opentelemetry.instrumentation.mcp.instrumentation import (
    InstrumentedStreamReader,
    InstrumentedStreamWriter,
    McpInstrumentor,
)
from opentelemetry.sdk.trace import TracerProvider


class _Stream:
    """Stands in for a transport's read or write stream."""


@asynccontextmanager
async def _two_streams():
    yield _Stream(), _Stream()


@asynccontextmanager
async def _streams_with_session_id():
    yield _Stream(), _Stream(), lambda: "session-id"


async def _raise_inside(wrapper, transport, error):
    async with wrapper(transport, None, (), {}):
        raise error


@pytest.mark.parametrize("transport", [_two_streams, _streams_with_session_id])
def test_application_exception_reaches_the_caller_unchanged(transport):
    """A failure inside the transport block is not an unpacking failure.

    Retrying the shape detection from the ``except`` clause would yield a
    second time, and contextlib would replace this error with
    ``RuntimeError: generator didn't stop after athrow()``.
    """
    wrapper = McpInstrumentor()._transport_wrapper(
        TracerProvider().get_tracer(__name__)
    )
    error = ValueError("application failure")

    with pytest.raises(ValueError) as caught:
        asyncio.run(_raise_inside(wrapper, transport, error))

    assert caught.value is error


_OPAQUE_RESULT = _Stream()
_ONE_STREAM_RESULT = (_Stream(),)
_FOUR_ELEMENT_RESULT = (_Stream(), _Stream(), lambda: "session-id", "extra")


@asynccontextmanager
async def _opaque_result():
    yield _OPAQUE_RESULT


@asynccontextmanager
async def _one_stream_result():
    yield _ONE_STREAM_RESULT


@asynccontextmanager
async def _four_element_result():
    yield _FOUR_ELEMENT_RESULT


@pytest.mark.parametrize(
    "transport, expected",
    [
        (_opaque_result, _OPAQUE_RESULT),
        (_one_stream_result, _ONE_STREAM_RESULT),
        (_four_element_result, _FOUR_ELEMENT_RESULT),
    ],
)
def test_unknown_transport_shapes_fall_back_to_the_raw_result(transport, expected):
    """A transport whose shape the wrapper cannot handle is passed through unchanged.

    That covers a non-iterable, a one-element tuple, and a tuple with more than
    the two or three elements the MCP transports define. The shape fallback is
    also inside the ``yield``, so a failure raised by the caller must still reach
    it rather than turning into a second yield.
    """
    wrapper = McpInstrumentor()._transport_wrapper(
        TracerProvider().get_tracer(__name__)
    )
    seen = []
    error = ValueError("application failure")

    async def consume():
        async with wrapper(transport, None, (), {}) as result:
            seen.append(result)
            raise error

    with pytest.raises(ValueError) as caught:
        asyncio.run(consume())

    assert caught.value is error
    # Identity, not equality: wrapt proxies compare equal to their wrapped
    # stream, so only `is` proves the raw result was passed through untouched.
    assert len(seen) == 1
    assert seen[0] is expected


@pytest.mark.parametrize(
    "transport, expected_length",
    [(_two_streams, 2), (_streams_with_session_id, 3)],
)
def test_both_transport_shapes_are_instrumented(transport, expected_length):
    wrapper = McpInstrumentor()._transport_wrapper(
        TracerProvider().get_tracer(__name__)
    )

    async def consume():
        async with wrapper(transport, None, (), {}) as streams:
            assert len(streams) == expected_length
            assert isinstance(streams[0], InstrumentedStreamReader)
            assert isinstance(streams[1], InstrumentedStreamWriter)

    asyncio.run(consume())
