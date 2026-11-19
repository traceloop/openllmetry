"""Each FastMCP server's spans carry that server's own workflow name."""

import pytest


@pytest.mark.asyncio
async def test_each_server_keeps_its_own_name(span_exporter):
    from fastmcp import Client, FastMCP

    first = FastMCP("first-server")

    @first.tool()
    async def ping() -> str:
        return "pong"

    # Constructing another server must not rename the first one's spans.
    FastMCP("second-server")

    async with Client(first) as client:
        await client.call_tool("ping", {})

    server_spans = [
        span
        for span in span_exporter.get_finished_spans()
        if span.name == "mcp.server"
    ]
    assert server_spans, "expected a server-side mcp.server span"
    assert {
        span.attributes.get("traceloop.workflow.name") for span in server_spans
    } == {"first-server.mcp"}


@pytest.mark.asyncio
async def test_generated_name_is_not_replaced_by_a_previous_server(span_exporter):
    from fastmcp import Client, FastMCP

    FastMCP("named-server")
    unnamed = FastMCP()

    @unnamed.tool()
    async def ping() -> str:
        return "pong"

    async with Client(unnamed) as client:
        await client.call_tool("ping", {})

    server_spans = [
        span
        for span in span_exporter.get_finished_spans()
        if span.name == "mcp.server"
    ]
    assert server_spans, "expected a server-side mcp.server span"
    assert {
        span.attributes.get("traceloop.workflow.name") for span in server_spans
    } == {f"{unnamed.name}.mcp"}
