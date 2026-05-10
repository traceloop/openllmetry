"""Tests for error.type attribute on protocol-level MCP tool errors (issue #4037)."""
import pytest
from opentelemetry.trace.status import StatusCode


@pytest.mark.asyncio
async def test_tool_error_sets_error_type_on_server_span(span_exporter, tracer_provider):
    """
    When a FastMCP tool raises an exception, the server-side span must have
    error.type set. The server-side span hits the isError=True code path in
    _execute_and_handle_result — that's the bug location.
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

    # The server-side tool span is the one that goes through _execute_and_handle_result
    # with isError=True. It should have error.type set.
    server_tool_spans = [
        s for s in spans
        if s.name == "fail_tool.tool" and s.status.status_code == StatusCode.ERROR
    ]

    assert len(server_tool_spans) >= 1, (
        f"Expected at least 1 ERROR fail_tool.tool span, got: {[s.name for s in spans]}"
    )

    # All ERROR tool spans should have error.type set
    for span in server_tool_spans:
        assert span.attributes.get("error.type") is not None, (
            f"error.type missing on ERROR span '{span.name}'"
        )

    # The server-side span specifically should have "tool_error" (isError=True path)
    tool_error_values = [
        s.attributes.get("error.type") for s in server_tool_spans
    ]
    assert "tool_error" in tool_error_values, (
        f"Expected 'tool_error' in error.type values, got: {tool_error_values}"
    )
