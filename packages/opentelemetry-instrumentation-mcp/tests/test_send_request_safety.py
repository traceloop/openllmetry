"""Tests for #4463: instrumentation failures must never change the outcome of
the traced MCP call. BaseSession.send_request returns the RPC result, so a
wrapper that swallows an instrumentation error and returns None replaces a
real result (e.g. CallToolResult) with None and crashes the caller."""

import pytest
from opentelemetry.trace.status import StatusCode


@pytest.mark.asyncio
async def test_tool_call_outside_active_span_returns_real_result(
    span_exporter, tracer_provider
):
    """
    #4463: when send_request runs outside any span (first call after
    instrumenting, a detached task, a non-recording context), the
    TraceContextTextMapPropagator writes no traceparent header. The old
    unconditional `carrier["traceparent"]` read then raised KeyError, dont_throw
    swallowed it and returned None, and ClientSession.call_tool crashed with
    AttributeError: 'NoneType' object has no attribute 'isError'.
    """
    from fastmcp import Client, FastMCP
    from opentelemetry import context

    server = FastMCP("no-active-span")

    @server.tool()
    async def ping() -> str:
        return "pong"

    async with Client(server) as client:
        # Client.__aenter__ makes the session span current. Push an empty
        # context so the tool call happens outside any span, as reported in
        # production for #4463. The meta dict makes the request carry an MCP
        # Meta object, which is what forces the trace-context injection branch
        # in patch_mcp_client to run.
        token = context.attach(context.Context())
        try:
            result = await client.call_tool("ping", {}, meta={"clientID": "x"})
        finally:
            context.detach(token)

    assert result.content[0].text == "pong"

    # The call was still traced as a root span, but with no errors.
    ping_spans = [
        s for s in span_exporter.get_finished_spans() if s.name == "ping.tool"
    ]
    assert ping_spans, "expected a ping.tool span for the traced call"
    assert all(
        s.status.status_code != StatusCode.ERROR for s in ping_spans
    ), "tracing failure was recorded as a span error"


@pytest.mark.asyncio
async def test_error_result_with_non_text_content_block_is_returned(
    span_exporter, tracer_provider
):
    """
    #4463: post-call span decoration read result.content[0].text unguarded.
    Error results can carry non-text content blocks (image, audio, embedded
    resource), which have no .text attribute. The AttributeError was raised
    after the RPC had already succeeded, swallowed by dont_throw, and the real
    result was thrown away in favour of None.
    """
    from mcp.types import CallToolResult, ImageContent

    from opentelemetry.instrumentation.mcp.instrumentation import McpInstrumentor

    instrumentor = McpInstrumentor()
    tracer = tracer_provider.get_tracer("test-send-request-safety")
    span = tracer.start_span("image-error.tool")

    image_error = CallToolResult(
        content=[
            ImageContent(
                type="image", data="iVBORw0KGgo=", mimeType="image/png"
            )
        ],
        isError=True,
    )

    async def fake_wrapped(*args, **kwargs):
        return image_error

    result = await instrumentor._execute_and_handle_result(
        span, "tools/call", [], {}, fake_wrapped, clean_output=True
    )
    span.end()

    assert result is image_error, "the real RPC result must be returned"

    # The error is still recorded on the span (error.type + ERROR status).
    error_spans = [
        s for s in span_exporter.get_finished_spans() if s.name == "image-error.tool"
    ]
    assert len(error_spans) == 1
    assert error_spans[0].status.status_code == StatusCode.ERROR
    assert error_spans[0].attributes.get("error.type") == "tool_error"
