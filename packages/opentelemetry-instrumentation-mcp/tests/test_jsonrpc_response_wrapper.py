"""Test cases for _jsonrpc_response_init_wrapper method demonstrating stdio MCP use cases.

These tests demonstrate how the JSONRPC response wrapper works with MCP communication,
which is the same pattern used by external clients like Claude and Copilot when they
communicate with MCP servers via stdio (standard input/output).

Note: Due to stdio communication complexity in test environments, these tests use
FastMCP which implements the same JSONRPC patterns that stdio mode uses.
"""

from opentelemetry.semconv_ai import SpanAttributes
from opentelemetry.trace.status import StatusCode


async def test_jsonrpc_response_stdio_success(span_exporter, tracer_provider) -> None:
    """Test JSONRPC response wrapper with successful MCP tool execution.

    This demonstrates the same JSONRPC communication pattern that external clients
    like Claude and Copilot use when communicating with MCP servers via stdio.
    The _jsonrpc_response_init_wrapper captures these responses and creates spans.
    """
    from fastmcp import FastMCP, Client

    # Create a FastMCP server (implements JSONRPC patterns as stdio mode)
    server = FastMCP("test-server")

    @server.tool()
    async def test_tool(message: str) -> str:
        """A test tool that returns a message.

        In stdio mode, this would be called by external clients
        like Claude/Copilot via JSONRPC over stdin/stdout.
        """
        return f"Tool executed successfully: {message}"

    # Use in-memory client (same JSONRPC patterns as stdio_client)
    async with Client(server) as client:
        # This simulates the JSONRPC request/response cycle that happens
        # when external clients call MCP tools via stdio
        result = await client.call_tool("test_tool", {"message": "Hello from stdio client"})
        assert len(result) == 1
        assert "Tool executed successfully: Hello from stdio client" in result[0].text

    # Get the finished spans
    spans = span_exporter.get_finished_spans()

    # Look for MCP_Tool_Response spans created by the JSONRPC response wrapper
    mcp_response_spans = [span for span in spans if span.name == "MCP_Tool_Response"]

    assert len(mcp_response_spans) >= 1, f"Expected at least 1 MCP_Tool_Response span, found {len(mcp_response_spans)}"

    response_span = mcp_response_spans[0]
    assert response_span.status.status_code == StatusCode.OK

    assert SpanAttributes.MCP_RESPONSE_VALUE in response_span.attributes
    response_value = response_span.attributes[SpanAttributes.MCP_RESPONSE_VALUE]
    assert "Tool executed successfully" in response_value

    assert SpanAttributes.MCP_REQUEST_ID in response_span.attributes
    request_id = response_span.attributes[SpanAttributes.MCP_REQUEST_ID]
    assert request_id is not None


async def test_jsonrpc_response_stdio_error(span_exporter, tracer_provider) -> None:
    """Test JSONRPC response wrapper with MCP error response.

    This demonstrates error handling in the same JSONRPC communication pattern
    that external clients like Claude and Copilot use when communicating with
    MCP servers via stdio.
    """
    from fastmcp import FastMCP, Client

    # Create a FastMCP server (implements same JSONRPC patterns as stdio mode)
    server = FastMCP("test-server")

    @server.tool()
    async def failing_tool(should_fail: bool = True) -> str:
        """A tool that can fail for testing error handling.

        In stdio mode, this error would be captured and sent back
        to external clients like Claude/Copilot via JSONRPC error response.
        """
        if should_fail:
            raise ValueError("Intentional test error - simulating stdio error response")
        return "Success!"

    # Use in-memory client (same JSONRPC patterns as stdio_client)
    async with Client(server) as client:
        # This simulates the JSONRPC error response cycle that happens
        # when external clients call MCP tools via stdio and they fail
        try:
            await client.call_tool("failing_tool", {"should_fail": True})
        except Exception:
            pass  # Expected to fail - this simulates stdio error handling

    # Get the finished spans
    spans = span_exporter.get_finished_spans()

    # Look for MCP_Tool_Response spans created by the JSONRPC response wrapper
    mcp_response_spans = [span for span in spans if span.name == "MCP_Tool_Response"]

    # Should have at least one MCP_Tool_Response span
    assert len(mcp_response_spans) >= 1, f"Expected at least 1 MCP_Tool_Response span, found {len(mcp_response_spans)}"

    # Check the first MCP_Tool_Response span
    response_span = mcp_response_spans[0]
    assert response_span.status.status_code == StatusCode.ERROR
    # The error description should now contain the actual error message from the response
    assert response_span.status.description is not None
    assert "error" in response_span.status.description.lower() or "NoneType" in response_span.status.description

    assert SpanAttributes.MCP_RESPONSE_VALUE in response_span.attributes
    response_value = response_span.attributes[SpanAttributes.MCP_RESPONSE_VALUE]
    assert "error" in response_value.lower() or "Intentional test error" in response_value

    assert SpanAttributes.MCP_REQUEST_ID in response_span.attributes
