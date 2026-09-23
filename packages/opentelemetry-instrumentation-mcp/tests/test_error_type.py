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
