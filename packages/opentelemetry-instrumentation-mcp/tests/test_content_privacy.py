"""Tests that MCP instrumentation honors TRACELOOP_TRACE_CONTENT.

Tool call arguments and results flowing through the MCP transport are
third-party content (they may contain end-user PII). Operators disable capture
of that content by setting TRACELOOP_TRACE_CONTENT=false, and every other
OpenLLMetry instrumentation (including the FastMCP wrapper in this same package)
respects it. These tests guard the transport-level instrumentation against
leaking that content onto spans when capture is disabled.
"""

import pytest
from opentelemetry.semconv_ai import SpanAttributes

CONTENT_ATTRIBUTES = (
    SpanAttributes.TRACELOOP_ENTITY_INPUT,
    SpanAttributes.TRACELOOP_ENTITY_OUTPUT,
    SpanAttributes.MCP_RESPONSE_VALUE,
)


async def _call_tool(secret_value):
    from fastmcp import FastMCP, Client

    server = FastMCP("privacy-test-server")

    @server.tool()
    async def echo(secret: str) -> str:
        """Echo back the provided secret string."""
        return f"echoed:{secret}"

    async with Client(server) as client:
        result = await client.call_tool("echo", {"secret": secret_value})
        assert result.content[0].text == f"echoed:{secret_value}"


@pytest.mark.asyncio
async def test_content_not_captured_when_disabled(
    span_exporter, tracer_provider, monkeypatch
) -> None:
    monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", "false")
    secret = "pii-9c3f-should-not-leak"

    await _call_tool(secret)

    spans = span_exporter.get_finished_spans()
    assert len(spans) > 0, "No spans were captured"

    for span in spans:
        for attr in CONTENT_ATTRIBUTES:
            value = span.attributes.get(attr)
            if value is not None:
                assert secret not in str(value), (
                    f"Sensitive content leaked into span '{span.name}' "
                    f"attribute '{attr}' despite TRACELOOP_TRACE_CONTENT=false"
                )


@pytest.mark.asyncio
async def test_content_captured_when_enabled(
    span_exporter, tracer_provider, monkeypatch
) -> None:
    monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", "true")
    secret = "visible-when-enabled"

    await _call_tool(secret)

    spans = span_exporter.get_finished_spans()
    captured = any(
        secret in str(span.attributes.get(attr))
        for span in spans
        for attr in CONTENT_ATTRIBUTES
        if span.attributes.get(attr) is not None
    )
    assert captured, (
        "Expected tool content to be captured on a span when "
        "TRACELOOP_TRACE_CONTENT=true"
    )
