"""Tests for error.type attribute on protocol-level MCP tool errors (issue #4037)."""
import pytest
from opentelemetry.trace.status import StatusCode


@pytest.mark.asyncio
async def test_tool_error_sets_error_type_on_client_span(span_exporter, tracer_provider):
    """
    When a FastMCP tool raises an exception, the client-side span must have
    error.type set. The client-side span wraps BaseSession.send_request and
    hits the isError=True branch in _execute_and_handle_result
    (instrumentation.py:344) — that's the bug location this test pins.
    """
    from fastmcp import FastMCP, Client

    server = FastMCP("test-error-type")

    @server.tool()
    async def fail_tool(x: int) -> str:
        raise ValueError("intentional tool error")

    try:
        async with Client(server) as client:
            await client.call_tool("fail_tool", {"x": 1})
    except Exception:
        pass  # client raises ToolError or re-raises ValueError — expected

    spans = span_exporter.get_finished_spans()

    # Both client- and server-side produce a fail_tool.tool ERROR span (same name,
    # different code paths). The client-side one — created by _execute_and_handle_result
    # on isError=True — is the path this PR fixes.
    error_tool_spans = [
        s for s in spans
        if s.name == "fail_tool.tool" and s.status.status_code == StatusCode.ERROR
    ]

    assert len(error_tool_spans) >= 1, (
        f"Expected at least 1 ERROR fail_tool.tool span, got: {[s.name for s in spans]}"
    )

    # Every ERROR tool span should carry error.type
    for span in error_tool_spans:
        assert span.attributes.get("error.type") is not None, (
            f"error.type missing on ERROR span '{span.name}'"
        )

    # The client-side span specifically carries "tool_error" (isError=True path).
    # The server-side span carries type(e).__name__ (e.g. "ValueError") from
    # fastmcp_instrumentation.py's exception handler.
    tool_error_values = [
        s.attributes.get("error.type") for s in error_tool_spans
    ]
    assert "tool_error" in tool_error_values, (
        f"Expected 'tool_error' in error.type values, got: {tool_error_values}"
    )


@pytest.mark.asyncio
async def test_tool_error_with_non_text_content_sets_error_type(
    span_exporter, tracer_provider
):
    """A non-text MCP error result must produce a tool_error span without raising."""
    from mcp.server.lowlevel import Server
    from mcp.shared.memory import create_connected_server_and_client_session
    from mcp.types import CallToolResult, ImageContent

    server = Server("test-nontext-error")

    @server.call_tool()
    async def img_error_tool(tool_name: str, arguments: dict):
        return CallToolResult(
            content=[
                ImageContent(
                    type="image",
                    data="aGVsbG8=",
                    mimeType="image/png",
                )
            ],
            isError=True,
        )

    async with create_connected_server_and_client_session(server) as session:
        await session.call_tool("img_error_tool", {})

    tool_spans = [
        s for s in span_exporter.get_finished_spans() if s.name.endswith(".tool")
    ]
    assert tool_spans, "expected a client-side tool span"

    client_error_spans = [
        s for s in tool_spans if s.status.status_code == StatusCode.ERROR
    ]
    assert client_error_spans, "expected the tool span to have ERROR status"

    assert len(client_error_spans) == 1, (
        f"Expected exactly 1 client-side ERROR tool span, "
        f"got {[s.name for s in client_error_spans]}"
    )

    assert client_error_spans[0].attributes.get("error.type") == "tool_error", (
        f"Expected error.type='tool_error', "
        f"got {client_error_spans[0].attributes.get('error.type')!r}"
    )
