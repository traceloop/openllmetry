"""TRACELOOP_TRACE_CONTENT must gate the MCP client path, not only FastMCP.

The package documents TRACELOOP_TRACE_CONTENT as the switch that disables content
logging. Before this test, only the FastMCP server-side wrapper consulted it: the
client path (tools/call arguments, non-tool request bodies, response bodies) recorded
content regardless, so an operator who turned the switch off still got request and
response payloads on their spans.

Each test drives the real client wrapper with a marker value and asserts the marker
is absent from every span attribute when content capture is off, and present when it
is on, so the test fails if either the gate or the capture itself regresses.

"Every span attribute" includes the two places content reaches a span without being
an attribute: a status description, and the message and stacktrace that
``record_exception`` writes as event attributes.
"""

import json

import pytest
from fastmcp import Client, FastMCP
from mcp.types import JSONRPCMessage, JSONRPCResponse
from opentelemetry.instrumentation.mcp.instrumentation import InstrumentedStreamWriter
from opentelemetry.trace import StatusCode

MARKER = "content-capture-marker-9f3a"


def _all_recorded_text(span_exporter) -> str:
    """Everything an exported span carries that could hold content, as one string.

    Status descriptions and event attributes count too: ``record_exception``
    puts the message and the whole stacktrace in the latter.
    """
    chunks = []
    for span in span_exporter.get_finished_spans():
        sources = [span.attributes or {}]
        sources.extend(event.attributes or {} for event in span.events)
        for attributes in sources:
            for value in attributes.values():
                if isinstance(value, (list, tuple)):
                    chunks.extend(str(item) for item in value)
                else:
                    chunks.append(str(value))
        if span.status is not None and span.status.description:
            chunks.append(span.status.description)
    return "\n".join(chunks)


def _server() -> FastMCP:
    """Build a server with one tool that echoes a caller-supplied token."""
    server = FastMCP("content-gate-server")

    @server.tool()
    async def echo_secret(token: str) -> str:
        """Echo back a caller-supplied token."""
        return f"received {token}"

    return server


async def test_tool_arguments_suppressed_when_content_capture_off(
    span_exporter, monkeypatch
) -> None:
    """With the switch off, tool arguments must not appear on any span."""
    monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", "false")

    async with Client(_server()) as client:
        await client.call_tool("echo_secret", {"token": MARKER})

    assert span_exporter.get_finished_spans(), "expected the tool call to be traced"
    assert MARKER not in _all_recorded_text(span_exporter)


async def test_tool_arguments_captured_when_content_capture_on(
    span_exporter, monkeypatch
) -> None:
    """With the switch on, tool arguments are still recorded as before."""
    monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", "true")

    async with Client(_server()) as client:
        await client.call_tool("echo_secret", {"token": MARKER})

    # The gate must not silently disable capture altogether: with the switch on,
    # the argument is still recorded.
    assert MARKER in _all_recorded_text(span_exporter)


async def test_non_tool_response_body_suppressed_when_content_capture_off(
    span_exporter, monkeypatch
) -> None:
    """list_tools goes through _handle_mcp_method, which serialized the whole response.

    The marker lives in the registered tool description, so it travels back in the
    list_tools result and exercises the response-serialization path.
    """
    monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", "false")

    server = FastMCP("content-gate-server")

    @server.tool(description=f"A tool whose description carries {MARKER}.")
    async def documented(arg: str) -> str:
        """A tool that exists only to carry the marker in its description."""
        return arg

    async with Client(server) as client:
        tools = await client.list_tools()

    assert any(MARKER in (t.description or "") for t in tools), (
        "the marker must reach the client, otherwise this test proves nothing"
    )
    assert span_exporter.get_finished_spans(), "expected the request to be traced"
    assert MARKER not in _all_recorded_text(span_exporter)


async def test_span_structure_survives_content_capture_off(
    span_exporter, monkeypatch
) -> None:
    """Turning content off must not remove spans or their non-content attributes."""
    monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", "false")

    async with Client(_server()) as client:
        await client.call_tool("echo_secret", {"token": MARKER})

    spans = span_exporter.get_finished_spans()
    tool_spans = [s for s in spans if s.name.endswith(".tool")]
    assert tool_spans, f"expected a tool span, got {[s.name for s in spans]}"

    entity_names = [
        (s.attributes or {}).get("traceloop.entity.name") for s in tool_spans
    ]
    assert "echo_secret" in entity_names

    # Structural attributes stay; only content is withheld.
    for span in tool_spans:
        attributes = span.attributes or {}
        assert "traceloop.span.kind" in attributes
        for key in ("traceloop.entity.input", "traceloop.entity.output"):
            if key in attributes:
                json.loads(attributes[key])  # if present it must still be valid JSON
                assert MARKER not in attributes[key]


def _failing_server() -> FastMCP:
    """Build a server whose tool fails with the marker in its message."""
    server = FastMCP("content-gate-server")

    @server.tool()
    async def boom(token: str) -> str:
        """Fail with the caller's token in the exception message."""
        raise ValueError(f"failure involving {token}")

    return server


async def test_error_text_suppressed_when_content_capture_off(
    span_exporter, monkeypatch
) -> None:
    """A failure's message and stacktrace are the server's text, so they are content."""
    monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", "false")

    async with Client(_failing_server()) as client:
        with pytest.raises(Exception):
            await client.call_tool("boom", {"token": MARKER})

    spans = span_exporter.get_finished_spans()
    assert spans, "expected the failed call to be traced"
    assert MARKER not in _all_recorded_text(span_exporter)

    # The failure itself stays visible; only its text is withheld.
    assert any(s.status.status_code is StatusCode.ERROR for s in spans)
    assert any("error.type" in (s.attributes or {}) for s in spans)


async def test_error_text_captured_when_content_capture_on(
    span_exporter, monkeypatch
) -> None:
    """With the switch on, the failure text is still recorded as before."""
    monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", "true")

    async with Client(_failing_server()) as client:
        with pytest.raises(Exception):
            await client.call_tool("boom", {"token": MARKER})

    assert MARKER in _all_recorded_text(span_exporter)


async def test_non_tool_response_body_captured_when_content_capture_on(
    span_exporter, monkeypatch
) -> None:
    """The paired case: list_tools still serializes its response when the switch is on."""
    monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", "true")

    server = FastMCP("content-gate-server")

    @server.tool(description=f"A tool whose description carries {MARKER}.")
    async def documented(arg: str) -> str:
        """A tool that exists only to carry the marker in its description."""
        return arg

    async with Client(server) as client:
        await client.list_tools()

    assert MARKER in _all_recorded_text(span_exporter)


class _Sink:
    """Stands in for the wrapped stream, recording what was forwarded to it."""

    def __init__(self):
        """Start with nothing sent."""
        self.sent = []

    async def send(self, item):
        """Record the forwarded item."""
        self.sent.append(item)


def _error_response() -> JSONRPCMessage:
    """A tool-error response whose payload text carries the marker."""
    return JSONRPCMessage(
        JSONRPCResponse(
            jsonrpc="2.0",
            id=1,
            result={"isError": True, "content": [{"type": "text", "text": MARKER}]},
        )
    )


@pytest.mark.parametrize(
    ("switch", "expected"), [("false", False), ("true", True)]
)
async def test_stream_writer_error_status_honors_the_switch(
    span_exporter, tracer_provider, monkeypatch, switch, expected
) -> None:
    """The stdio/SSE proxy is not reachable through the in-memory client.

    _handle_mcp_tool_call covers the FastMCP path; InstrumentedStreamWriter is the
    one used for stdio and SSE, and it sets the same payload text as a status
    description. Drive it directly rather than leaving it uncovered.
    """
    monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", switch)
    sink = _Sink()

    await InstrumentedStreamWriter(sink, tracer_provider.get_tracer(__name__)).send(
        _error_response()
    )

    assert sink.sent, "the item must still reach the wrapped stream"
    spans = span_exporter.get_finished_spans()
    assert spans, "expected the response to be traced"
    assert any(s.status.status_code is StatusCode.ERROR for s in spans)
    assert (MARKER in _all_recorded_text(span_exporter)) is expected
