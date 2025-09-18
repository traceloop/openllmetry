"""Test cases for _jsonrpc_response_init_wrapper method."""

import pytest
from unittest.mock import Mock
from opentelemetry.semconv_ai import SpanAttributes
from opentelemetry.trace.status import StatusCode


async def test_jsonrpc_response_success_with_content(span_exporter, tracer_provider) -> None:
    """Test JSONRPC response wrapper with successful response containing content."""
    from opentelemetry.instrumentation.mcp import McpInstrumentor

    instrumentor = McpInstrumentor()
    tracer = tracer_provider.get_tracer(__name__)
    wrapper = instrumentor._jsonrpc_response_init_wrapper(tracer)

    mock_wrapped = Mock(return_value=None)

    result_data = {
        "content": [
            {
                "type": "text",
                "text": "Tool execution completed successfully"
            }
        ],
        "isError": False
    }

    args = []
    kwargs = {
        "id": "test-request-123",
        "result": result_data
    }

    # Call the wrapper
    wrapper(mock_wrapped, None, args, kwargs)

    # Verify the wrapped function was called
    mock_wrapped.assert_called_once_with(*args, **kwargs)

    # Check spans
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1, f"Expected 1 span, got {len(spans)}"

    span = spans[0]
    assert span.name == "MCP_Tool_Response"
    assert span.status.status_code == StatusCode.OK

    # Check attributes
    assert SpanAttributes.MCP_RESPONSE_VALUE in span.attributes
    response_value = span.attributes[SpanAttributes.MCP_RESPONSE_VALUE]
    assert "Tool execution completed successfully" in response_value

    assert SpanAttributes.MCP_REQUEST_ID in span.attributes
    assert span.attributes[SpanAttributes.MCP_REQUEST_ID] == "test-request-123"


async def test_jsonrpc_response_error_with_content(span_exporter, tracer_provider) -> None:
    """Test JSONRPC response wrapper with error response containing content."""
    from opentelemetry.instrumentation.mcp import McpInstrumentor

    instrumentor = McpInstrumentor()
    tracer = tracer_provider.get_tracer(__name__)
    wrapper = instrumentor._jsonrpc_response_init_wrapper(tracer)

    mock_wrapped = Mock(return_value=None)

    result_data = {
        "content": [
            {
                "type": "text",
                "text": "Tool execution failed: Invalid parameters"
            }
        ],
        "isError": True
    }

    args = ["test-request-456", result_data]
    kwargs = {}

    wrapper(mock_wrapped, None, args, kwargs)

    mock_wrapped.assert_called_once_with(*args, **kwargs)

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1, f"Expected 1 span, got {len(spans)}"

    span = spans[0]
    assert span.name == "MCP_Tool_Response"
    assert span.status.status_code == StatusCode.ERROR
    assert span.status.description == "Tool execution error"

    assert SpanAttributes.MCP_RESPONSE_VALUE in span.attributes
    response_value = span.attributes[SpanAttributes.MCP_RESPONSE_VALUE]
    assert "Tool execution failed: Invalid parameters" in response_value

    assert SpanAttributes.MCP_REQUEST_ID in span.attributes
    assert span.attributes[SpanAttributes.MCP_REQUEST_ID] == "test-request-456"


async def test_jsonrpc_response_no_content(span_exporter, tracer_provider) -> None:
    """Test JSONRPC response wrapper with response that has no content field."""
    from opentelemetry.instrumentation.mcp import McpInstrumentor

    instrumentor = McpInstrumentor()
    tracer = tracer_provider.get_tracer(__name__)
    wrapper = instrumentor._jsonrpc_response_init_wrapper(tracer)

    mock_wrapped = Mock(return_value=None)

    result_data = {
        "someOtherField": "some value",
        "isError": False
    }

    args = []
    kwargs = {
        "id": "test-request-789",
        "result": result_data
    }

    wrapper(mock_wrapped, None, args, kwargs)

    mock_wrapped.assert_called_once_with(*args, **kwargs)

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 0, f"Expected 0 spans, got {len(spans)}"


async def test_jsonrpc_response_no_request_id(span_exporter, tracer_provider) -> None:
    """Test JSONRPC response wrapper without request ID."""
    from opentelemetry.instrumentation.mcp import McpInstrumentor

    # Create instrumentor and get the wrapper
    instrumentor = McpInstrumentor()
    tracer = tracer_provider.get_tracer(__name__)
    wrapper = instrumentor._jsonrpc_response_init_wrapper(tracer)

    # Mock the wrapped function
    mock_wrapped = Mock(return_value=None)

    # Create a successful response with content but no request ID
    result_data = {
        "content": [
            {
                "type": "text",
                "text": "Tool execution completed without request ID"
            }
        ],
        "isError": False
    }

    args = []
    kwargs = {
        "result": result_data
        # No id field
    }

    # Call the wrapper
    wrapper(mock_wrapped, None, args, kwargs)

    # Verify the wrapped function was called
    mock_wrapped.assert_called_once_with(*args, **kwargs)

    # Check spans
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1, f"Expected 1 span, got {len(spans)}"

    span = spans[0]
    assert span.name == "MCP_Tool_Response"
    assert span.status.status_code == StatusCode.OK

    # Check attributes
    assert SpanAttributes.MCP_RESPONSE_VALUE in span.attributes
    response_value = span.attributes[SpanAttributes.MCP_RESPONSE_VALUE]
    assert "Tool execution completed without request ID" in response_value

    # Should not have request ID attribute
    assert SpanAttributes.MCP_REQUEST_ID not in span.attributes


async def test_jsonrpc_response_wrapper_exception_handling(span_exporter, tracer_provider) -> None:
    """Test that the wrapper handles exceptions gracefully using @dont_throw decorator."""
    from opentelemetry.instrumentation.mcp import McpInstrumentor

    # Create instrumentor and get the wrapper
    instrumentor = McpInstrumentor()
    tracer = tracer_provider.get_tracer(__name__)
    wrapper = instrumentor._jsonrpc_response_init_wrapper(tracer)

    # Mock the wrapped function to raise an exception
    mock_wrapped = Mock(side_effect=Exception("Test exception"))

    # Create a valid response
    result_data = {
        "content": [
            {
                "type": "text",
                "text": "This should not create a span due to exception"
            }
        ],
        "isError": False
    }

    args = []
    kwargs = {
        "id": "exception-test-123",
        "result": result_data
    }

    try:
        wrapper(mock_wrapped, None, args, kwargs)
    except Exception:
        pytest.fail("Wrapper should not raise exceptions due to @dont_throw decorator")

    mock_wrapped.assert_called_once_with(*args, **kwargs)
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1, f"Expected 1 span (created before exception), got {len(spans)}"

    span = spans[0]
    assert span.name == "MCP_Tool_Response"
    assert span.status.status_code == StatusCode.OK
