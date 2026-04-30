from types import SimpleNamespace

import pytest
from opentelemetry.instrumentation.mcp import McpInstrumentor
from opentelemetry.instrumentation.mcp.instrumentation import InstrumentedStreamWriter
from opentelemetry.trace.status import StatusCode


@pytest.mark.parametrize(
    ("result", "expected_description"),
    [
        (
            SimpleNamespace(
                isError=True,
                content=[SimpleNamespace(text="tool failed")],
            ),
            "tool failed",
        ),
        (
            SimpleNamespace(isError=True, content=[]),
            "MCP tool call returned isError=True",
        ),
        (
            SimpleNamespace(isError=True),
            "MCP tool call returned isError=True",
        ),
    ],
)
async def test_protocol_error_result_sets_error_type(
    span_exporter,
    tracer_provider,
    result,
    expected_description,
) -> None:
    instrumentor = McpInstrumentor()
    tracer = tracer_provider.get_tracer(__name__)

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
    assert error_span.status.description == expected_description
    assert error_span.attributes["error.type"] == "MCPToolError"


@pytest.mark.parametrize(
    ("result", "expected_description"),
    [
        (
            {
                "isError": True,
                "content": [{"text": "streamed tool failed"}],
            },
            "streamed tool failed",
        ),
        (
            {
                "isError": True,
                "content": [],
            },
            "MCP response returned isError=True",
        ),
        (
            {
                "isError": True,
            },
            "MCP response returned isError=True",
        ),
    ],
)
async def test_response_stream_protocol_error_sets_error_type(
    span_exporter,
    tracer_provider,
    result,
    expected_description,
) -> None:
    tracer = tracer_provider.get_tracer(__name__)
    sent_items = []

    async def send(item):
        sent_items.append(item)

    wrapped_stream = SimpleNamespace(send=send)
    stream_writer = InstrumentedStreamWriter(wrapped_stream, tracer)
    response = SimpleNamespace(
        id="request-1",
        result=result,
    )
    item = SimpleNamespace(message=SimpleNamespace(root=response))

    await stream_writer.send(item)

    assert sent_items == [item]
    error_span = span_exporter.get_finished_spans()[-1]
    assert error_span.name == "ResponseStreamWriter"
    assert error_span.status.status_code == StatusCode.ERROR
    assert error_span.status.description == expected_description
    assert error_span.attributes["error.type"] == "MCPToolError"
