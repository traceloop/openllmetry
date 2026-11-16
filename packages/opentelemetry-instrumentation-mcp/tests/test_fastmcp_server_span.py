import pytest


async def test_fastmcp_server_mcp_parent_span(span_exporter, tracer_provider) -> None:
    """Test that FastMCP tool calls have mcp.server as parent span."""
    from fastmcp import FastMCP, Client

    # Create a simple FastMCP server
    server = FastMCP("test-server")

    @server.tool()
    async def test_tool(x: int) -> int:
        """A simple test tool."""
        return x * 2

    # Use in-memory client to connect to the server
    async with Client(server) as client:
        # Test tool calling
        result = await client.call_tool("test_tool", {"x": 5})
        assert len(result.content) == 1
        assert result.content[0].text == "10"

    # Get the finished spans
    spans = span_exporter.get_finished_spans()

    # Debug: Print span details with parent info
    print(f"\nTotal spans: {len(spans)}")
    for i, span in enumerate(spans):
        parent_id = span.parent.span_id if span.parent else "None"
        print(f"Span {i}: name='{span.name}', span_id={span.get_span_context().span_id}, "
              f"parent_id={parent_id}, trace_id={span.get_span_context().trace_id}")

    # Look specifically for mcp.server and tool spans
    server_mcp_spans = [span for span in spans if span.name == 'mcp.server']
    tool_spans = [span for span in spans if span.name.endswith('.tool')]

    print(f"\nMCP Server spans: {len(server_mcp_spans)}")
    print(f"Tool spans: {len(tool_spans)}")

    # Check if we have the expected spans
    assert len(server_mcp_spans) >= 1, f"Expected at least 1 mcp.server span, found {len(server_mcp_spans)}"
    assert len(tool_spans) >= 1, f"Expected at least 1 tool span, found {len(tool_spans)}"

    # Find server-side spans (should be in same trace)
    server_side_spans = []
    for server_span in server_mcp_spans:
        for tool_span in tool_spans:
            if (server_span.get_span_context().trace_id == tool_span.get_span_context().trace_id and
                    tool_span.parent and
                    tool_span.parent.span_id == server_span.get_span_context().span_id):
                server_side_spans.append((server_span, tool_span))
                break

    print(f"\nFound {len(server_side_spans)} server-side span pairs")

    # Verify we found at least one proper parent-child relationship
    assert len(server_side_spans) >= 1, "Expected at least one mcp.server span to be parent of a tool span"

    # Check the specific parent-child relationship
    server_span, tool_span = server_side_spans[0]
    assert tool_span.parent.span_id == server_span.get_span_context().span_id, \
        "Tool span should be child of mcp.server span"
    assert server_span.get_span_context().trace_id == tool_span.get_span_context().trace_id, \
        "Parent and child should be in same trace"

    # Verify MCP server span attributes
    assert server_span.attributes.get('traceloop.span.kind') == 'server', \
        "Server span should have server span kind"
    assert server_span.attributes.get('traceloop.entity.name') == 'mcp.server', \
        "Server span should have mcp.server entity name"

    # Verify tool span attributes
    assert tool_span.attributes.get('traceloop.span.kind') == 'tool', \
        "Tool span should have tool span kind"
    assert tool_span.attributes.get('traceloop.entity.name') == 'test_tool', \
        "Tool span should have correct entity name"



@pytest.mark.parametrize(
    "call",
    [
        lambda s: s.call_tool("double", {"x": 5}, run_middleware=False),
        lambda s: s.call_tool(name="double", arguments={"x": 5}, run_middleware=False),
        lambda s: s.call_tool("double", arguments={"x": 5}, run_middleware=False),
    ],
    ids=["positional", "keywords", "mixed"],
)
async def test_fastmcp3_server_span_reads_name_and_arguments(call, span_exporter) -> None:
    """FastMCP.call_tool(run_middleware=False) is what the server span wraps on
    fastmcp 3; its name and arguments may come positionally or by keyword."""
    import json

    from fastmcp import FastMCP

    if not hasattr(FastMCP, "call_tool"):
        pytest.skip("FastMCP.call_tool is fastmcp 3+")

    server = FastMCP("test-server")

    @server.tool()
    async def double(x: int) -> int:
        return x * 2

    await call(server)

    tool_spans = [s for s in span_exporter.get_finished_spans() if s.name.endswith(".tool")]
    assert [s.name for s in tool_spans] == ["double.tool"]
    assert tool_spans[0].attributes["traceloop.entity.name"] == "double"
    tool_input = json.loads(tool_spans[0].attributes["traceloop.entity.input"])
    assert tool_input == {"tool_name": "double", "arguments": {"x": 5}}
