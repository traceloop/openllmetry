from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from mcp.types import (
    CallToolRequest,
    CallToolRequestParams,
    ClientRequest,
    JSONRPCMessage,
    JSONRPCRequest,
    RequestParams,
)
from opentelemetry import propagate, trace
from opentelemetry.instrumentation.mcp import instrumentation
from opentelemetry.instrumentation.mcp.instrumentation import (
    InstrumentedStreamReader,
    InstrumentedStreamWriter,
    McpInstrumentor,
)


@pytest.mark.parametrize(
    ("meta", "expected_custom_value"),
    [
        (None, None),
        (RequestParams.Meta(custom_value="preserved"), "preserved"),
    ],
)
async def test_client_trace_context_is_injected_without_replacing_metadata(
    monkeypatch, meta, expected_custom_value
):
    """Verify ordinary tool calls propagate trace context and preserve metadata."""
    propagator = AsyncMock()
    propagator.inject = lambda carrier: carrier.update(
        {
            "traceparent": "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01",
            "tracestate": "vendor=value",
        }
    )
    monkeypatch.setattr(instrumentation.propagate, "get_global_textmap", lambda: propagator)

    # BaseInstrumentor is a singleton. Bypass it so this unit test cannot mutate
    # the session-scoped instrumentor used by the integration tests.
    instrumentor = object.__new__(McpInstrumentor)
    instrumentor._handle_tool_call = AsyncMock(return_value="sent")
    params = CallToolRequestParams(name="test", arguments={}, _meta=meta)
    message = ClientRequest(CallToolRequest(params=params))

    wrapped = AsyncMock()
    result = await instrumentor.patch_mcp_client(AsyncMock())(wrapped, None, (message,), {})

    assert result == "sent"
    assert params.meta is not None
    assert params.meta.traceparent == "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"
    assert params.meta.tracestate == "vendor=value"
    assert getattr(params.meta, "custom_value", None) == expected_custom_value
    wire_meta = message.model_dump(by_alias=True, exclude_none=True)["params"]["_meta"]
    assert wire_meta["traceparent"] == "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"
    assert wire_meta["tracestate"] == "vendor=value"


async def test_client_trace_context_leaves_meta_none_without_active_span():
    """Without an active span, ordinary calls with meta=None must keep params.meta as None."""
    instrumentor = object.__new__(McpInstrumentor)
    instrumentor._handle_tool_call = AsyncMock(return_value="sent")
    params = CallToolRequestParams(name="test", arguments={}, _meta=None)
    message = SimpleNamespace(root=SimpleNamespace(method="tools/call", params=params))

    wrapped = AsyncMock()
    result = await instrumentor.patch_mcp_client(AsyncMock())(wrapped, None, (message,), {})

    assert result == "sent"
    assert params.meta is None


async def test_streamable_http_client_wraps_both_current_and_deprecated_entrypoints(
    tracer_provider,
):
    """Both streamable_http_client and deprecated streamablehttp_client must wrap streams once."""
    instrumentor = object.__new__(McpInstrumentor)
    tracer = tracer_provider.get_tracer(__name__)

    sent_messages = []

    class _FakeReader:
        pass

    class _FakeWriter:
        async def send(self, item):
            sent_messages.append(item)

    @asynccontextmanager
    async def fake_streamable_http_client(*args, **kwargs):
        yield _FakeReader(), _FakeWriter(), lambda: "session-1"

    @asynccontextmanager
    async def fake_streamablehttp_client(*args, **kwargs):
        async with fake_module.streamable_http_client(*args, **kwargs) as streams:
            yield streams

    fake_module = SimpleNamespace(
        streamable_http_client=fake_streamable_http_client,
        streamablehttp_client=fake_streamablehttp_client,
    )
    instrumentor._wrap_streamable_http_client(fake_module, tracer)

    async with fake_module.streamable_http_client("http://localhost:8000/mcp") as (
        read_stream,
        write_stream,
        get_session_id,
    ):
        assert isinstance(read_stream, InstrumentedStreamReader)
        assert isinstance(write_stream, InstrumentedStreamWriter)
        assert not isinstance(write_stream.__wrapped__, InstrumentedStreamWriter)
        assert get_session_id() == "session-1"

        rpc_request = JSONRPCMessage(
            JSONRPCRequest(jsonrpc="2.0", id=1, method="tools/call", params={"name": "echo"})
        )
        with tracer.start_as_current_span("caller") as caller_span:
            await write_stream.send(rpc_request)

        meta = sent_messages[0].root.params["_meta"]
        extracted = trace.get_current_span(propagate.extract(meta))
        assert (
            extracted.get_span_context().trace_id
            == caller_span.get_span_context().trace_id
        )

    # Deprecated streamablehttp_client delegates to streamable_http_client; verify no double-wrap.
    async with fake_module.streamablehttp_client("http://localhost:8000/mcp") as (
        legacy_reader,
        legacy_writer,
        _,
    ):
        assert isinstance(legacy_reader, InstrumentedStreamReader)
        assert not isinstance(legacy_reader.__wrapped__, InstrumentedStreamReader)
        assert isinstance(legacy_writer, InstrumentedStreamWriter)
        assert not isinstance(legacy_writer.__wrapped__, InstrumentedStreamWriter)

    # Verify caller exceptions inside the context manager propagate unchanged.
    with pytest.raises(ValueError, match="caller failure"):
        async with fake_module.streamable_http_client("http://localhost:8000/mcp"):
            raise ValueError("caller failure")
