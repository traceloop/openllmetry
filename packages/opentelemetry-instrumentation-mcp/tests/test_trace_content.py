"""Tests for TRACELOOP_TRACE_CONTENT env var in MCP and FastMCP instrumentors."""


async def test_trace_content_false_suppresses_tool_spans(
    span_exporter, tracer_provider, monkeypatch
) -> None:
    """With TRACELOOP_TRACE_CONTENT=false, no input/output on any tool span."""
    monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", "false")

    from fastmcp import FastMCP, Client

    server = FastMCP("no-content-server")

    @server.tool()
    async def echo(message: str) -> str:
        return f"Echo: {message}"

    async with Client(server) as client:
        await client.call_tool("echo", {"message": "secret"})

    spans = span_exporter.get_finished_spans()
    tool_spans = [s for s in spans if s.name == "echo.tool"]
    assert len(tool_spans) >= 1, f"Expected echo.tool spans, got: {[s.name for s in spans]}"

    for span in tool_spans:
        assert "traceloop.entity.input" not in span.attributes, (
            f"Input must be absent when TRACELOOP_TRACE_CONTENT=false, got: {span.attributes}"
        )
        assert "traceloop.entity.output" not in span.attributes, (
            f"Output must be absent when TRACELOOP_TRACE_CONTENT=false, got: {span.attributes}"
        )
        # Span identity attributes must still be present
        assert span.attributes.get("traceloop.span.kind") == "tool"
        assert span.attributes.get("traceloop.entity.name") == "echo"


async def test_trace_content_false_suppresses_mcp_method_spans(
    span_exporter, tracer_provider, monkeypatch
) -> None:
    """With TRACELOOP_TRACE_CONTENT=false, no input/output on non-tool MCP method spans."""
    monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", "false")

    from fastmcp import FastMCP, Client

    server = FastMCP("no-content-mcp-server")

    @server.tool()
    async def dummy() -> str:
        return "ok"

    async with Client(server) as client:
        await client.list_tools()

    spans = span_exporter.get_finished_spans()
    mcp_spans = [s for s in spans if s.name.endswith(".mcp")]
    assert len(mcp_spans) >= 1, f"Expected .mcp spans, got: {[s.name for s in spans]}"

    for span in mcp_spans:
        assert "traceloop.entity.input" not in span.attributes, (
            f"Input must be absent on {span.name} when TRACELOOP_TRACE_CONTENT=false"
        )
        assert "traceloop.entity.output" not in span.attributes, (
            f"Output must be absent on {span.name} when TRACELOOP_TRACE_CONTENT=false"
        )


async def test_trace_content_true_includes_tool_input_output(
    span_exporter, tracer_provider, monkeypatch
) -> None:
    """With TRACELOOP_TRACE_CONTENT=true (default), input/output are present on tool spans."""
    monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", "true")

    from fastmcp import FastMCP, Client

    server = FastMCP("with-content-server")

    @server.tool()
    async def greet(name: str) -> str:
        return f"Hello, {name}!"

    async with Client(server) as client:
        await client.call_tool("greet", {"name": "world"})

    spans = span_exporter.get_finished_spans()
    tool_spans = [s for s in spans if s.name == "greet.tool"]
    assert len(tool_spans) >= 1

    for span in tool_spans:
        assert "traceloop.entity.input" in span.attributes, (
            "Input must be present when TRACELOOP_TRACE_CONTENT=true"
        )
        assert "traceloop.entity.output" in span.attributes, (
            "Output must be present when TRACELOOP_TRACE_CONTENT=true"
        )

async def test_trace_content_false_suppresses_error_response_text(
    span_exporter, tracer_provider, monkeypatch
) -> None:
    """With TRACELOOP_TRACE_CONTENT=false, error response text must not leak into
    the span status description or recorded exception events, but the span must
    still be marked ERROR."""
    monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", "false")

    from fastmcp import FastMCP, Client

    secret_ticker = "SPCX"
    secret = f"super-secret-error-detail. {secret_ticker} is not valid ticker. Did you mean SPCE?"

    server = FastMCP("error-no-content-server")

    @server.tool()
    async def fail_tool(ticker: str) -> str:
        raise ValueError(secret)

    try:
        async with Client(server) as client:
            await client.call_tool("fail_tool", {"ticker": secret_ticker})
    except Exception:
        pass  # client re-raises the tool error — expected

    spans = span_exporter.get_finished_spans()

    from opentelemetry.trace.status import StatusCode

    error_spans = [s for s in spans if s.status.status_code == StatusCode.ERROR]
    assert len(error_spans) >= 1, (
        f"Expected at least one ERROR span, got: {[s.name for s in spans]}"
    )

    # secret_ticker is both the call argument and part of the tool's error
    # message, so asserting its absence covers both leak vectors: arguments
    # leaking from the call and the return/error value leaking from the tool.
    for span in spans:
        # It must not leak into the status description...
        description = span.status.description or ""
        assert secret_ticker not in description, (
            f"Error response text leaked into status description of '{span.name}' "
            f"when TRACELOOP_TRACE_CONTENT=false: {description!r}"
        )

        # ...nor into any recorded exception event (message/stacktrace). The SDK's
        # automatic exception recording would otherwise re-leak the content even
        # though the explicit set_status was suppressed.
        for event in span.events:
            for attr_key, attr_value in event.attributes.items():
                assert secret_ticker not in str(attr_value), (
                    f"Error content leaked into event '{event.name}' "
                    f"attribute '{attr_key}' of span '{span.name}' "
                    f"when TRACELOOP_TRACE_CONTENT=false: {attr_value!r}"
                )
