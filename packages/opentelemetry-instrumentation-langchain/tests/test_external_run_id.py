"""Tests for handling external run_ids from systems like LangSmith."""

import logging
from unittest.mock import MagicMock, Mock, patch
from uuid import uuid4

import pytest
from opentelemetry.instrumentation.langchain.callback_handler import (
    TraceloopCallbackHandler,
)


@pytest.fixture
def callback_handler(tracer_provider):
    """
    Create a TraceloopCallbackHandler bound to the provided tracer provider.
    
    Parameters:
    	tracer_provider: The OpenTelemetry TracerProvider used to initialize the callback handler.
    
    Returns:
    	TraceloopCallbackHandler: An initialized callback handler associated with the given tracer_provider.
    """
    return TraceloopCallbackHandler(tracer_provider=tracer_provider)


@pytest.fixture
def mock_run_manager():
    """
    Create a mock run manager with a UUID `run_id` attribute.
    
    Returns:
        manager (Mock): A unittest.mock.Mock instance with a `run_id` attribute set to a generated UUID.
    """
    manager = Mock()
    manager.run_id = uuid4()
    return manager


def test_external_run_id_no_keyerror(
    callback_handler, mock_run_manager, instrument_legacy, caplog
):
    """Test that external run_ids (e.g., from LangSmith) don't cause KeyError."""
    from opentelemetry.instrumentation.langchain import _OpenAITracingWrapper

    # Create the wrapper
    wrapper = _OpenAITracingWrapper(callback_handler)

    # Mock the wrapped function
    mock_wrapped = Mock(return_value="test_result")

    # Ensure the run_id is NOT in the spans dictionary
    # (simulating an external system like LangSmith creating the run_id)
    assert mock_run_manager.run_id not in callback_handler.spans

    # Create kwargs with the external run_manager
    kwargs = {
        "run_manager": mock_run_manager,
        "extra_headers": {},
    }

    # Capture debug logs
    with caplog.at_level(logging.DEBUG):
        # Call the wrapper - should NOT raise KeyError
        result = wrapper(mock_wrapped, None, [], kwargs)

    # Verify the function was called successfully
    assert result == "test_result"
    mock_wrapped.assert_called_once()

    # Verify debug log was generated
    assert any(
        "No span found for run_id" in record.message
        and "skipping header injection" in record.message
        for record in caplog.records
    )

    # Verify extra_headers were not modified since there's no span
    # (the original empty dict should still be there)
    assert "traceparent" not in kwargs["extra_headers"]


def test_internal_run_id_injects_headers(
    callback_handler, mock_run_manager, instrument_legacy
):
    """Test that internal run_ids (created by OTEL) get headers injected."""
    from opentelemetry.instrumentation.langchain import _OpenAITracingWrapper
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.trace import SpanKind

    # Create a real span and add it to the callback handler's spans
    tracer = TracerProvider().get_tracer(__name__)
    span = tracer.start_span("test_span", kind=SpanKind.CLIENT)

    # Create a span holder (mimicking what the callback handler does)
    span_holder = Mock()
    span_holder.span = span

    # Add the span to the callback handler's spans dictionary
    callback_handler.spans[mock_run_manager.run_id] = span_holder

    # Create the wrapper
    wrapper = _OpenAITracingWrapper(callback_handler)

    # Mock the wrapped function
    mock_wrapped = Mock(return_value="test_result")

    # Create kwargs with the internal run_manager
    kwargs = {
        "run_manager": mock_run_manager,
        "extra_headers": {},
    }

    # Call the wrapper
    result = wrapper(mock_wrapped, None, [], kwargs)

    # Verify the function was called successfully
    assert result == "test_result"
    mock_wrapped.assert_called_once()

    # Verify headers were injected (traceparent should exist)
    assert "traceparent" in kwargs["extra_headers"]
    assert kwargs["extra_headers"]["traceparent"]  # Should have a value

    # Clean up
    span.end()


def test_no_run_manager_continues_normally(callback_handler, instrument_legacy):
    """
    Ensure the tracing wrapper executes the wrapped function and does not inject trace headers when no `run_manager` is provided in `kwargs`.
    
    Verifies that the wrapped callable is invoked and that `extra_headers` remains without a `traceparent` entry when `kwargs` does not contain a `run_manager`.
    """
    from opentelemetry.instrumentation.langchain import _OpenAITracingWrapper

    # Create the wrapper
    wrapper = _OpenAITracingWrapper(callback_handler)

    # Mock the wrapped function
    mock_wrapped = Mock(return_value="test_result")

    # Create kwargs without run_manager
    kwargs = {"extra_headers": {}}

    # Call the wrapper - should work fine
    result = wrapper(mock_wrapped, None, [], kwargs)

    # Verify the function was called successfully
    assert result == "test_result"
    mock_wrapped.assert_called_once()

    # Verify no headers were injected
    assert "traceparent" not in kwargs["extra_headers"]
