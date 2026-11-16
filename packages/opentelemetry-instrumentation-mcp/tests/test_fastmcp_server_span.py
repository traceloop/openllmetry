import pytest
from opentelemetry.trace import StatusCode


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


def _fastmcp_with_mount():
    from fastmcp import FastMCP

    if not hasattr(FastMCP, "mount") or not hasattr(FastMCP, "call_tool"):
        pytest.skip("needs FastMCP.mount and FastMCP.call_tool")
    return FastMCP


def _server_spans(span_exporter):
    """The server-side span pairs: mcp.server spans and the tool spans under them."""
    spans = span_exporter.get_finished_spans()
    server_spans = [s for s in spans if s.name == "mcp.server"]
    server_ids = {s.get_span_context().span_id for s in server_spans}
    tool_spans = [s for s in spans if s.parent and s.parent.span_id in server_ids]
    return server_spans, tool_spans


async def test_fastmcp_mounted_server_traces_each_tool_run_once(span_exporter) -> None:
    """A mounted server's tool runs through the parent's provider and the
    child's call_tool; it is still one execution, so one span pair, attributed
    to the child server."""
    FastMCP = _fastmcp_with_mount()
    from fastmcp import Client

    child = FastMCP("child")
    parent = FastMCP("parent")

    @child.tool()
    async def double(x: int) -> int:
        return x * 2

    parent.mount(child, namespace="c")

    async with Client(parent) as client:
        result = await client.call_tool("c_double", {"x": 5})
    assert result.content[0].text == "10"

    server_spans, tool_spans = _server_spans(span_exporter)
    assert [s.name for s in tool_spans] == ["double.tool"]
    assert len(server_spans) == 1
    assert tool_spans[0].parent.span_id == server_spans[0].get_span_context().span_id
    for span in (server_spans[0], tool_spans[0]):
        assert span.attributes["traceloop.workflow.name"] == "child.mcp"


async def test_fastmcp_error_result_marks_server_spans_as_error(span_exporter) -> None:
    """A tool can fail without raising, by returning ToolResult(is_error=True)."""
    FastMCP = _fastmcp_with_mount()
    from fastmcp import Client
    from fastmcp.tools.base import ToolResult
    from mcp.types import TextContent

    server = FastMCP("test-server")

    @server.tool()
    async def bad() -> ToolResult:
        return ToolResult(content=[TextContent(type="text", text="nope")], is_error=True)

    async with Client(server) as client:
        result = await client.call_tool("bad", {}, raise_on_error=False)
    assert result.is_error

    server_spans, tool_spans = _server_spans(span_exporter)
    assert len(server_spans) == 1 and len(tool_spans) == 1
    for span in (server_spans[0], tool_spans[0]):
        assert span.status.status_code is StatusCode.ERROR
        assert span.attributes["error.type"] == "tool_error"
        assert "nope" in span.status.description


async def test_fastmcp_task_submission_is_not_traced_as_a_tool_run(span_exporter) -> None:
    """With task_meta, Tool._run only submits a background task and returns a
    CreateTaskResult; the tool has not run, so no server span pair is opened.
    (This tool forbids tasks, so the submission is refused before reaching
    docket, which keeps the test free of a task backend.)"""
    FastMCP = _fastmcp_with_mount()
    from fastmcp.utilities.tasks import TaskMeta

    server = FastMCP("test-server")

    @server.tool()
    async def double(x: int) -> int:
        return x * 2

    tool = await server.get_tool("double")
    with pytest.raises(Exception, match="does not support task-augmented execution"):
        await tool._run({"x": 5}, task_meta=TaskMeta())

    assert _server_spans(span_exporter) == ([], [])
