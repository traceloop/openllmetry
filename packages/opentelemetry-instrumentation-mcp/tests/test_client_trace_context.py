from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from mcp.types import CallToolRequestParams, RequestParams
from opentelemetry.instrumentation.mcp import instrumentation
from opentelemetry.instrumentation.mcp.instrumentation import McpInstrumentor


@pytest.mark.parametrize(
    ("meta", "expected_custom_value"),
    [
        (None, None),
        (RequestParams.Meta(custom_value="preserved"), "preserved"),
    ],
)
async def test_client_trace_context_is_injected_without_replacing_metadata(monkeypatch, meta, expected_custom_value):
    """Verify ordinary tool calls propagate trace context and preserve metadata."""
    propagator = AsyncMock()
    propagator.inject = lambda carrier: carrier.update(
        {
            "traceparent": "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01",
            "tracestate": "vendor=value",
        }
    )
    monkeypatch.setattr(instrumentation, "TraceContextTextMapPropagator", lambda: propagator)

    # BaseInstrumentor is a singleton. Bypass it so this unit test cannot mutate
    # the session-scoped instrumentor used by the integration tests.
    instrumentor = object.__new__(McpInstrumentor)
    instrumentor._handle_tool_call = AsyncMock(return_value="sent")
    params = CallToolRequestParams(name="test", arguments={}, _meta=meta)
    message = SimpleNamespace(root=SimpleNamespace(method="tools/call", params=params))

    wrapped = AsyncMock()
    result = await instrumentor.patch_mcp_client(AsyncMock())(wrapped, None, (message,), {})

    assert result == "sent"
    assert params.meta is not None
    assert params.meta.traceparent == "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"
    assert params.meta.tracestate == "vendor=value"
    assert getattr(params.meta, "custom_value", None) == expected_custom_value
