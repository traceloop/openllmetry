import pytest
from unittest.mock import AsyncMock, MagicMock
from opentelemetry.instrumentation.mcp.instrumentation import (
    InstrumentedStreamReader,
    InstrumentedStreamWriter,
)
from opentelemetry.trace import get_tracer


@pytest.mark.asyncio
async def test_writer_sends_non_iterable_result():
    # Mock underlying stream writer
    mock_wrapped = AsyncMock()
    mock_wrapped.send = AsyncMock(return_value=None)

    tracer = get_tracer("test")
    writer = InstrumentedStreamWriter(mock_wrapped, tracer)

    # Mock a result that is a non-iterable class (no __contains__ / __iter__)
    class NonIterableResult:
        pass

    class CustomRequest:
        def __init__(self):
            self.result = NonIterableResult()
            self.id = 123
            self.params = None

    class CustomMessage:
        def __init__(self):
            self.root = CustomRequest()

    # Test that InstrumentedStreamWriter.send does not raise TypeError and successfully sends
    await writer.send(CustomMessage())

    # Assert send was called
    mock_wrapped.send.assert_called_once()


@pytest.mark.asyncio
async def test_writer_propagates_exceptions():
    # Mock underlying stream writer that raises a transport exception
    mock_wrapped = AsyncMock()
    mock_wrapped.send = AsyncMock(side_effect=ConnectionResetError("Connection lost"))

    tracer = get_tracer("test")
    writer = InstrumentedStreamWriter(mock_wrapped, tracer)

    class CustomRequest:
        def __init__(self):
            self.id = 123
            self.params = None

    class CustomMessage:
        def __init__(self):
            self.root = CustomRequest()

    # Test that InstrumentedStreamWriter propagates the transport exception instead of swallowing it
    with pytest.raises(ConnectionResetError):
        await writer.send(CustomMessage())


@pytest.mark.asyncio
async def test_reader_handles_list_params():
    # Create an async iterable mock
    mock_wrapped = MagicMock()

    class CustomRequest:
        def __init__(self):
            # List parameters, which do not support .get()
            self.params = [1, 2, 3]

    class CustomMessage:
        def __init__(self):
            self.root = CustomRequest()

    messages = [CustomMessage()]

    async def mock_aiter(*args, **kwargs):
        for msg in messages:
            yield msg

    mock_wrapped.__aiter__ = mock_aiter

    tracer = get_tracer("test")
    reader = InstrumentedStreamReader(mock_wrapped, tracer)

    # Test that InstrumentedStreamReader.__aiter__ does not crash on non-dict params
    results = []
    async for item in reader:
        results.append(item)

    assert len(results) == 1
    assert results[0] == messages[0]
