async def test_fastmcp_instrumentor(span_exporter, tracer_provider) -> None:
    from fastmcp import FastMCP, Client

    # Create a simple FastMCP server
    server = FastMCP("test-server")

    @server.tool()
    async def add_numbers(a: int, b: int) -> int:
        """Add two numbers together."""
        return a + b

    @server.resource("test://greeting")
    def get_greeting() -> str:
        """Get a test greeting."""
        return "Hello from FastMCP!"

    # Use in-memory client to connect to the server
    async with Client(server) as client:
        # Test tool listing
        tools_res = await client.list_tools()
        assert len(tools_res) == 1
        assert tools_res[0].name == "add_numbers"

        # Test tool calling
        result = await client.call_tool("add_numbers", {"a": 5, "b": 3})
        assert len(result.content) == 1
        assert result.content[0].text == "8"

        # Test resource listing
        resources_res = await client.list_resources()
        assert len(resources_res) == 1
        assert str(resources_res[0].uri) == "test://greeting"

        # Test resource reading
        resource_result = await client.read_resource("test://greeting")
        assert len(resource_result) == 1
        assert resource_result[0].text == "Hello from FastMCP!"

    # Get the finished spans
    spans = span_exporter.get_finished_spans()

    # Verify spans were created
    assert len(spans) > 0, "No spans were captured"

    # Debug: Print span details
    print(f"\nTotal spans: {len(spans)}")
    for i, span in enumerate(spans):
        print(f"Span {i}: name='{span.name}', trace_id={span.get_span_context().trace_id}")

    # Verify all spans belong to the same trace
    trace_ids = set(span.get_span_context().trace_id for span in spans)
    assert len(trace_ids) == 1, (
        f"Expected all spans in same trace, found {len(trace_ids)} different traces: {trace_ids}"
    )

    # Assert specific span details
    span_names = [span.name for span in spans]
    # Verify expected MCP operation spans are present
    # Only tools/call should have .tool suffix, others should have .mcp suffix
    expected_spans = [
        'initialize.mcp', 'tools/list.mcp', 'add_numbers.tool',
        'resources/list.mcp', 'resources/read.mcp'
    ]

    for expected_span_name in expected_spans:
        matching_spans = [span for span in spans if span.name == expected_span_name]
        assert len(matching_spans) >= 1, (
            f"Expected span '{expected_span_name}' not found. All spans: {span_names}"
        )

    # Verify specific operations (we should have both client-side and server-side spans)
    tool_call_spans = [span for span in spans if span.name == 'add_numbers.tool']
    assert len(tool_call_spans) == 2, (
        f"Expected exactly 2 add_numbers.tool spans (client + server), found {len(tool_call_spans)}"
    )

    resource_read_spans = [span for span in spans if span.name == 'resources/read.mcp']
    assert len(resource_read_spans) >= 1, (
        f"Expected at least 1 resources/read.mcp span, found {len(resource_read_spans)}"
    )

    for i, span in enumerate(tool_call_spans):
        # Verify span metadata
        assert span.attributes.get('traceloop.span.kind') == 'tool', (
            f"Span {i} should have tool span kind"
        )
        assert span.attributes.get('traceloop.entity.name') == 'add_numbers', (
            f"Span {i} should have correct entity name"
        )

        # Verify actual input content
        input_attr = span.attributes.get('traceloop.entity.input', '')
        assert '"tool_name": "add_numbers"' in input_attr, (
            f"Span {i} input should contain tool_name: {input_attr}"
        )
        assert '"a": 5' in input_attr, f"Span {i} input should contain a=5: {input_attr}"
        assert '"b": 3' in input_attr, f"Span {i} input should contain b=3: {input_attr}"

        # Verify actual output content
        output_attr = span.attributes.get('traceloop.entity.output', '')
        assert '8' in output_attr, f"Span {i} output should contain result 8: {output_attr}"

    # Verify non-tool operations have correct attributes with actual content
    resource_read_span = resource_read_spans[0]

    # Verify resource read input contains the URI being read
    resource_input = resource_read_span.attributes.get('traceloop.entity.input', '')
    assert 'test://greeting' in resource_input, (
        f"Expected 'test://greeting' in resource read input: {resource_input}"
    )

    # Verify resource read output contains the expected content
    resource_output = resource_read_span.attributes.get('traceloop.entity.output', '')
    assert 'Hello from FastMCP!' in resource_output, (
        f"Expected 'Hello from FastMCP!' in resource read output: {resource_output}"
    )

    # Verify RequestStreamWriter spans were removed (as requested)
    request_writer_spans = [span for span in spans if span.name == 'RequestStreamWriter']
    assert len(request_writer_spans) == 0, (
        f"RequestStreamWriter spans should be removed, found {len(request_writer_spans)}"
    )

    # Verify TRACELOOP_WORKFLOW_NAME is set correctly on server spans
    mcp_server_spans = [span for span in spans if span.name == 'mcp.server']
    assert len(mcp_server_spans) >= 1, (
        f"Expected at least 1 mcp.server span, found {len(mcp_server_spans)}"
    )

    for server_span in mcp_server_spans:
        workflow_name = server_span.attributes.get('traceloop.workflow.name')
        assert workflow_name == 'test-server.mcp', (
            f"Expected workflow name 'test-server.mcp', got '{workflow_name}'"
        )

    # Verify TRACELOOP_WORKFLOW_NAME is also set on tool spans
    server_tool_spans = [span for span in spans if span.name == 'add_numbers.tool'
                         and span.attributes.get('traceloop.span.kind') == 'tool'
                         and 'traceloop.workflow.name' in span.attributes]
    assert len(server_tool_spans) >= 1, (
        f"Expected at least 1 server-side tool span with workflow name, found {len(server_tool_spans)}"
    )

    for tool_span in server_tool_spans:
        workflow_name = tool_span.attributes.get('traceloop.workflow.name')
        assert workflow_name == 'test-server.mcp', (
            f"Expected workflow name 'test-server.mcp' on tool span, got '{workflow_name}'"
        )
