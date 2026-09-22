import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from mcp.types import CallToolResult, ImageContent, TextContent, EmbeddedResource, TextResourceContents
from opentelemetry import context
from opentelemetry.instrumentation.mcp import McpInstrumentor
from opentelemetry.trace import StatusCode


def request(method="tools/call"):
    return SimpleNamespace(root=SimpleNamespace(
        method=method, params=SimpleNamespace(name="test", arguments={}, meta=SimpleNamespace())
    ))


@pytest.mark.parametrize("content", [
    [],
    [TextContent(type="text", text="failed")],
    [ImageContent(type="image", data="aGVsbG8=", mimeType="image/png")],
    [EmbeddedResource(type="resource", resource=TextResourceContents(uri="file:///test", text="failed"))],
])
@pytest.mark.asyncio
async def test_error_content_preserves_result(content, tracer_provider, span_exporter):
    result = CallToolResult(content=content, isError=True)
    wrapped = AsyncMock(return_value=result)
    wrapper = McpInstrumentor().patch_mcp_client(tracer_provider.get_tracer(__name__))
    assert await wrapper(wrapped, None, (request(),), {}) is result
    wrapped.assert_awaited_once()
    span = span_exporter.get_finished_spans()[-1]
    assert span.status.status_code == StatusCode.ERROR
    assert span.attributes["error.type"] == "tool_error"


@pytest.mark.parametrize("failure", ["inject", "start_span", "set_attribute", "set_status", "end"])
@pytest.mark.parametrize("method", ["tools/call", "tools/list"])
@pytest.mark.asyncio
async def test_telemetry_failure_preserves_result(failure, method, monkeypatch):
    tracer = Mock()
    if failure == "inject":
        monkeypatch.setattr(
            "opentelemetry.instrumentation.mcp.instrumentation.TraceContextTextMapPropagator.inject",
            Mock(side_effect=RuntimeError("telemetry")),
        )
    elif failure == "start_span":
        tracer.start_span.side_effect = RuntimeError("telemetry")
    else:
        getattr(tracer.start_span.return_value, failure).side_effect = RuntimeError("telemetry")
    result = CallToolResult(content=[])
    wrapped = AsyncMock(return_value=result)
    before = context.get_current()
    wrapper = McpInstrumentor().patch_mcp_client(tracer)
    assert await wrapper(wrapped, None, (), {"request": request(method)}) is result
    wrapped.assert_awaited_once()
    assert context.get_current() is before


@pytest.mark.parametrize("error", [ValueError("tool failed"), asyncio.CancelledError()])
@pytest.mark.asyncio
async def test_application_exception_is_preserved(error):
    tracer = Mock()
    tracer.start_span.return_value.record_exception.side_effect = RuntimeError("telemetry")
    tracer.start_span.return_value.end.side_effect = RuntimeError("cleanup")
    wrapped = AsyncMock(side_effect=error)
    before = context.get_current()
    wrapper = McpInstrumentor().patch_mcp_client(tracer)
    with pytest.raises(type(error)) as caught:
        await wrapper(wrapped, None, (request(),), {})
    assert caught.value is error
    wrapped.assert_awaited_once()
    assert context.get_current() is before
