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
