from types import SimpleNamespace

from opentelemetry.instrumentation.mcp import McpInstrumentor
from opentelemetry.instrumentation.mcp.instrumentation import InstrumentedStreamWriter
from opentelemetry.trace.status import StatusCode


async def test_protocol_error_result_sets_error_type(
    span_exporter,
    tracer_provider,
) -> None:
    instrumentor = McpInstrumentor()
    tracer = tracer_provider.get_tracer(__name__)
    result = SimpleNamespace(
        isError=True,
        content=[SimpleNamespace(text="tool failed")],
    )

    async def wrapped():
        return result

    with tracer.start_as_current_span("test.tool") as span:
        actual = await instrumentor._execute_and_handle_result(
            span,
            "tools/call",
            (),
            {},
            wrapped,
            clean_output=True,
        )

    assert actual is result
    error_span = span_exporter.get_finished_spans()[-1]
    assert error_span.status.status_code == StatusCode.ERROR
    assert error_span.status.description == "tool failed"
    assert error_span.attributes["error.type"] == "MCPToolError"


async def test_response_stream_protocol_error_sets_error_type(
    span_exporter,
    tracer_provider,
) -> None:
    tracer = tracer_provider.get_tracer(__name__)
    wrapped_stream = SimpleNamespace(send=_async_noop)
    stream_writer = InstrumentedStreamWriter(wrapped_stream, tracer)
    response = SimpleNamespace(
        id="request-1",
        result={
            "isError": True,
            "content": [{"text": "streamed tool failed"}],
        },
    )
    item = SimpleNamespace(message=SimpleNamespace(root=response))

    await stream_writer.send(item)

    error_span = span_exporter.get_finished_spans()[-1]
    assert error_span.name == "ResponseStreamWriter"
    assert error_span.status.status_code == StatusCode.ERROR
    assert error_span.status.description == "streamed tool failed"
    assert error_span.attributes["error.type"] == "MCPToolError"


async def _async_noop(_item):
    return None
