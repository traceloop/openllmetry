"""A failure inside tracing must never change the outcome of the traced call (issue #4463).

``BaseSession.send_request`` returns the RPC result, so the client wrapper's return
value is load-bearing. These tests pin that invariant from both ends: the span
lifecycle (``guarded_span``) cannot break the call, and the wrapper returns the MCP
result it received -- or raises the RPC's own exception -- whatever the
instrumentation does.

Driven directly rather than through a server where possible: the wrapper under test
is what ``McpInstrumentor.patch_mcp_client`` returns, so a fake ``wrapped`` is the
only way to make the instrumentation fail on demand without a real transport.
"""

import asyncio
import logging
from types import SimpleNamespace

import pytest
from mcp.types import (
    AudioContent,
    CallToolResult,
    EmbeddedResource,
    ImageContent,
    JSONRPCMessage,
    JSONRPCResponse,
    TextContent,
    TextResourceContents,
)
from opentelemetry.instrumentation.mcp import McpInstrumentor
from opentelemetry.instrumentation.mcp import instrumentation as instrumentation_module
from opentelemetry.instrumentation.mcp.instrumentation import InstrumentedStreamWriter
from opentelemetry.instrumentation.mcp.utils import Config, guarded_span
from opentelemetry.semconv.attributes.error_attributes import ERROR_TYPE
from opentelemetry.semconv_ai import SpanAttributes
from opentelemetry.trace import INVALID_SPAN, StatusCode

logger = logging.getLogger(__name__)


def _request(method="tools/call", meta=None):
    """Stands in for a ClientRequest: the wrapper only reads .root.method/.params."""
    params = SimpleNamespace(name="mytool", arguments={"a": 1}, meta=meta)
    return SimpleNamespace(root=SimpleNamespace(method=method, params=params))


def _instrumentation_records(caplog):
    """Only this package's own log records: another library's warning is not ours."""
    return [r for r in caplog.records if r.name.startswith("opentelemetry.instrumentation.mcp")]


def _returning(result):
    async def wrapped(*args, **kwargs):
        return result

    return wrapped


def _raising(exc):
    async def wrapped(*args, **kwargs):
        raise exc

    return wrapped


def _ok_result():
    return CallToolResult(content=[TextContent(type="text", text="ok")], isError=False)


def test_guarded_span_yields_a_real_span(tracer_provider, span_exporter):
    span_exporter.clear()
    with guarded_span(logger, "test", tracer_provider.get_tracer(__name__), "s") as span:
        assert span is not INVALID_SPAN
        span.set_attribute("k", "v")
    spans = span_exporter.get_finished_spans()
    assert [s.name for s in spans] == ["s"]
    assert spans[0].attributes["k"] == "v"


def test_guarded_span_yields_invalid_span_when_the_tracer_fails(broken_tracer, caplog):
    with caplog.at_level(logging.DEBUG):
        with guarded_span(logger, "test", broken_tracer, "s") as span:
            assert span is INVALID_SPAN
            span.set_attribute("k", "v")  # must not raise
    assert "tracer exploded" in caplog.text


def test_guarded_span_swallows_a_span_end_failure(end_raises):
    with guarded_span(logger, "test", end_raises, "s") as span:
        span.set_attribute("k", "v")
    # no exception escapes the with-block


def test_guarded_span_ends_the_span_when_the_body_raises(tracer_provider, span_exporter):
    span_exporter.clear()
    with pytest.raises(ValueError, match="body failed"):
        with guarded_span(logger, "test", tracer_provider.get_tracer(__name__), "s"):
            raise ValueError("body failed")
    assert len(span_exporter.get_finished_spans()) == 1


async def test_guarded_span_lets_cancellation_unwind(tracer_provider, span_exporter):
    """Only Exception is ever swallowed: a cancellation has to keep unwinding."""
    span_exporter.clear()
    with pytest.raises(asyncio.CancelledError):
        with guarded_span(logger, "test", tracer_provider.get_tracer(__name__), "s"):
            raise asyncio.CancelledError()
    assert len(span_exporter.get_finished_spans()) == 1


@pytest.mark.parametrize(
    "result",
    [
        CallToolResult(
            content=[ImageContent(type="image", data="aGk=", mimeType="image/png")], isError=True
        ),
        CallToolResult(content=[], isError=True),
        CallToolResult(
            content=[AudioContent(type="audio", data="aGk=", mimeType="audio/wav")],
            isError=True,
        ),
        CallToolResult(
            content=[
                EmbeddedResource(
                    type="resource",
                    resource=TextResourceContents(uri="file:///x.txt", text="hi"),
                )
            ],
            isError=True,
        ),
    ],
    ids=["image", "empty-content", "audio", "embedded-resource"],
)
async def test_error_result_without_text_is_returned_to_the_caller(
    result, tracer_provider, span_exporter
):
    """Only TextContent carries .text, and every one of these is a valid MCP error result."""
    instrumentor = McpInstrumentor()
    returned = await instrumentor.patch_mcp_client(tracer_provider.get_tracer(__name__))(
        _returning(result), None, (_request(),), {}
    )
    assert returned is result
    span = span_exporter.get_finished_spans()[0]
    assert span.status.status_code is StatusCode.ERROR
    assert span.attributes[ERROR_TYPE] == "tool_error"


async def test_text_error_result_still_carries_the_server_message(tracer_provider, span_exporter):
    """The getattr rewrite must not cost the server's error text.

    test_error_type.py::test_tool_error_sets_error_type_on_client_span already drives this
    branch through a real FastMCP client, but asserts only on error.type; this pins the
    status description, which is the part the rewrite touches.
    """
    result = CallToolResult(content=[TextContent(type="text", text="boom")], isError=True)
    instrumentor = McpInstrumentor()
    await instrumentor.patch_mcp_client(tracer_provider.get_tracer(__name__))(
        _returning(result), None, (_request(),), {}
    )
    span = span_exporter.get_finished_spans()[0]
    assert "boom" in span.status.description


async def test_broken_tracer_still_issues_the_call_and_returns_the_result(broken_tracer):
    result = _ok_result()
    instrumentor = McpInstrumentor()
    returned = await instrumentor.patch_mcp_client(broken_tracer)(
        _returning(result), None, (_request(),), {}
    )
    assert returned is result


async def test_span_end_failure_still_returns_the_result(end_raises):
    result = _ok_result()
    instrumentor = McpInstrumentor()
    returned = await instrumentor.patch_mcp_client(end_raises)(
        _returning(result), None, (_request(),), {}
    )
    assert returned is result


async def test_rpc_failure_reaches_the_caller(tracer_provider, span_exporter):
    instrumentor = McpInstrumentor()
    with pytest.raises(RuntimeError, match="transport died"):
        await instrumentor.patch_mcp_client(tracer_provider.get_tracer(__name__))(
            _raising(RuntimeError("transport died")), None, (_request(),), {}
        )
    assert span_exporter.get_finished_spans()[0].status.status_code is StatusCode.ERROR


async def test_rpc_failure_survives_a_span_end_failure(end_raises):
    """An instrumentation failure must never replace the traced call's own failure."""
    instrumentor = McpInstrumentor()
    with pytest.raises(RuntimeError, match="transport died"):
        await instrumentor.patch_mcp_client(end_raises)(
            _raising(RuntimeError("transport died")), None, (_request(),), {}
        )


async def test_rpc_failure_survives_a_record_failure(
    tracer_provider, span_exporter, record_raises, caplog
):
    """The record guard, on the client path: recording the RPC's failure may not replace it.

    Both failures are RuntimeErrors, so the match is what tells them apart -- the
    caller must still see the transport's, never the instrumentation's.
    """
    instrumentor = McpInstrumentor()
    with caplog.at_level(logging.WARNING):
        with pytest.raises(RuntimeError, match="transport died"):
            await instrumentor.patch_mcp_client(tracer_provider.get_tracer(__name__))(
                _raising(RuntimeError("transport died")), None, (_request(),), {}
            )
    assert "record_error exploded" in caplog.text
    assert len(span_exporter.get_finished_spans()) == 1, "the span must still be ended"


async def test_non_tool_method_result_is_returned(tracer_provider):
    result = SimpleNamespace(tools=[])
    instrumentor = McpInstrumentor()
    returned = await instrumentor.patch_mcp_client(tracer_provider.get_tracer(__name__))(
        _returning(result), None, (_request(method="tools/list"),), {}
    )
    assert returned is result


async def test_tool_call_with_meta_outside_any_span_returns_the_result(tracer_provider):
    """inject() writes no traceparent with no span current -- the KeyError path."""
    result = _ok_result()
    meta = SimpleNamespace(traceparent=None)
    request = _request(meta=meta)
    instrumentor = McpInstrumentor()
    returned = await instrumentor.patch_mcp_client(tracer_provider.get_tracer(__name__))(
        _returning(result), None, (request,), {}
    )
    assert returned is result
    assert meta.traceparent is None


async def test_meta_is_still_injected_under_an_active_span(tracer_provider):
    """The crash fix must not disable trace-context propagation."""
    result = _ok_result()
    meta = SimpleNamespace(traceparent=None)
    request = _request(meta=meta)
    tracer = tracer_provider.get_tracer(__name__)
    instrumentor = McpInstrumentor()
    with tracer.start_as_current_span("caller"):
        await instrumentor.patch_mcp_client(tracer)(_returning(result), None, (request,), {})
    assert meta.traceparent is not None
    assert meta.traceparent.startswith("00-")


async def test_malformed_request_still_issues_the_call(tracer_provider):
    """Nothing readable on args[0].root: the pre-call guard must fail open."""
    result = _ok_result()
    instrumentor = McpInstrumentor()
    returned = await instrumentor.patch_mcp_client(tracer_provider.get_tracer(__name__))(
        _returning(result), None, (SimpleNamespace(),), {}
    )
    assert returned is result


async def test_instrumentation_failure_is_logged_above_debug(broken_tracer, caplog):
    """The issue's second complaint: at INFO and above, production saw nothing."""
    result = _ok_result()
    instrumentor = McpInstrumentor()
    with caplog.at_level(logging.INFO):
        returned = await instrumentor.patch_mcp_client(broken_tracer)(
            _returning(result), None, (_request(),), {}
        )
    assert returned is result
    assert "tracer exploded" in caplog.text
    assert any(r.levelno >= logging.WARNING for r in caplog.records)


class _Sink:
    """Stands in for the wrapped stream, recording what was forwarded to it."""

    def __init__(self):
        self.sent = []

    async def send(self, item):
        self.sent.append(item)


async def test_raising_exception_logger_does_not_break_the_call(broken_tracer):
    """An operator's exception_logger is instrumentation too: its failure is not the call's.

    Config.exception_logger is reset by McpInstrumentor.__init__, so it is set after.
    """
    result = _ok_result()
    instrumentor = McpInstrumentor()

    def boom(exc):
        raise ValueError("exception_logger exploded")

    Config.exception_logger = boom
    try:
        returned = await instrumentor.patch_mcp_client(broken_tracer)(
            _returning(result), None, (_request(),), {}
        )
        assert returned is result
    finally:
        Config.exception_logger = None


def _raise_decoration_failure(*args, **kwargs):
    """Replaces whatever renders the output attribute, as an unbound method or a function."""
    raise RuntimeError("decoration exploded")


async def test_error_status_survives_a_decoration_failure(
    tracer_provider, span_exporter, monkeypatch, caplog
):
    """A failure recording the response body must still leave the span's status honest.

    Status and output sit in separate guards for exactly this: under a single
    guard, a raise on the way to the output attribute would skip the status,
    leaving a span that says error.type=tool_error and carries no status at all.
    Decoration is made to fail outright -- _extract_clean_output swallows its
    own exceptions, so a result it cannot read is not enough to drive this.
    """
    monkeypatch.setattr(McpInstrumentor, "_extract_clean_output", _raise_decoration_failure)
    result = CallToolResult(content=[TextContent(type="text", text="boom")], isError=True)
    instrumentor = McpInstrumentor()
    with caplog.at_level(logging.WARNING):
        returned = await instrumentor.patch_mcp_client(tracer_provider.get_tracer(__name__))(
            _returning(result), None, (_request(),), {}
        )
    assert returned is result
    span = span_exporter.get_finished_spans()[0]
    assert span.attributes[ERROR_TYPE] == "tool_error"
    assert span.status.status_code is StatusCode.ERROR
    assert "boom" in span.status.description
    assert SpanAttributes.TRACELOOP_ENTITY_OUTPUT not in span.attributes
    assert "decoration exploded" in caplog.text


async def test_non_tool_result_survives_a_decoration_failure(
    tracer_provider, span_exporter, monkeypatch
):
    """The same guard split on the non-tool path, whose output goes through serialize()."""
    monkeypatch.setattr(instrumentation_module, "serialize", _raise_decoration_failure)
    result = SimpleNamespace(tools=[])
    instrumentor = McpInstrumentor()
    returned = await instrumentor.patch_mcp_client(tracer_provider.get_tracer(__name__))(
        _returning(result), None, (_request(method="tools/list"),), {}
    )
    assert returned is result
    span = span_exporter.get_finished_spans()[0]
    assert span.status.status_code is StatusCode.OK
    assert SpanAttributes.TRACELOOP_ENTITY_OUTPUT not in span.attributes


async def test_cancellation_unwinds_through_the_client_wrapper(tracer_provider, span_exporter):
    instrumentor = McpInstrumentor()
    with pytest.raises(asyncio.CancelledError):
        await instrumentor.patch_mcp_client(tracer_provider.get_tracer(__name__))(
            _raising(asyncio.CancelledError()), None, (_request(),), {}
        )
    assert len(span_exporter.get_finished_spans()) == 1, "the span must still be ended"


async def test_writer_tolerates_non_text_error_content(span_exporter, tracer_provider, caplog):
    """An error response carrying an image block is valid MCP. Reaching for ['text'] there
    costs the span its status and logs a warning with a full traceback for every such
    message.
    """
    sink = _Sink()
    response = JSONRPCMessage(
        JSONRPCResponse(
            jsonrpc="2.0",
            id=1,
            result={
                "isError": True,
                "content": [{"type": "image", "data": "aGk=", "mimeType": "image/png"}],
            },
        )
    )
    with caplog.at_level(logging.WARNING):
        await InstrumentedStreamWriter(sink, tracer_provider.get_tracer(__name__)).send(response)
    assert sink.sent == [response]
    spans = span_exporter.get_finished_spans()
    assert any(s.status.status_code is StatusCode.ERROR for s in spans), (
        "an isError response must leave an ERROR span whatever its content block is"
    )
    assert _instrumentation_records(caplog) == [], (
        f"instrumentation logged on a valid MCP response: {caplog.text}"
    )
