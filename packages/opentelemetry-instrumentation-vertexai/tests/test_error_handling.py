"""Tests for error handling in the VertexAI instrumentation.

These tests verify that when a VertexAI API call raises an exception,
the OpenTelemetry span correctly records the error, sets the status
to ERROR, and re-raises the exception.
"""

import pytest
from unittest.mock import MagicMock, AsyncMock

from opentelemetry.instrumentation.vertexai import _wrap, _awrap
from opentelemetry.trace.status import StatusCode


class TestSyncErrorHandling:
    """Test error handling in the synchronous _wrap function."""

    def test_records_exception_on_api_error(self, tracer_provider, span_exporter):
        """When the wrapped function raises, the span should record the exception."""
        tracer = tracer_provider.get_tracer("test")
        event_logger = None
        to_wrap = {
            "span_name": "vertexai.generate_content",
            "package": "vertexai.generative_models",
            "object": "GenerativeModel",
            "method": "generate_content",
        }

        # Create the wrapper
        wrapper_fn = _wrap(tracer, event_logger, to_wrap)

        # Create a mock "wrapped" function that raises
        mock_wrapped = MagicMock(side_effect=Exception("HTTP 401 Unauthorized"))

        # Create a mock instance (simulates GenerativeModel)
        mock_instance = MagicMock()
        mock_instance._model_name = "publishers/google/models/gemini-pro"

        with pytest.raises(Exception, match="HTTP 401 Unauthorized"):
            wrapper_fn(mock_wrapped, mock_instance, ("Hello",), {})

        spans = span_exporter.get_finished_spans()
        assert len(spans) == 1

        span = spans[0]
        assert span.status.status_code == StatusCode.ERROR
        assert "HTTP 401 Unauthorized" in span.status.description

        # Verify exception event was recorded on the span
        exception_events = [e for e in span.events if e.name == "exception"]
        assert len(exception_events) == 1

    def test_exception_is_reraised(self, tracer_provider, span_exporter):
        """The original exception type and message must be preserved."""
        tracer = tracer_provider.get_tracer("test")
        to_wrap = {
            "span_name": "vertexai.generate_content",
            "package": "vertexai.generative_models",
            "object": "GenerativeModel",
            "method": "generate_content",
        }

        wrapper_fn = _wrap(tracer, None, to_wrap)
        mock_wrapped = MagicMock(side_effect=ValueError("Bad input"))
        mock_instance = MagicMock()
        mock_instance._model_name = "publishers/google/models/gemini-pro"

        with pytest.raises(ValueError, match="Bad input"):
            wrapper_fn(mock_wrapped, mock_instance, ("Hello",), {})

    def test_successful_call_has_no_error(self, tracer_provider, span_exporter):
        """A successful API call should NOT have error status."""
        tracer = tracer_provider.get_tracer("test")
        to_wrap = {
            "span_name": "vertexai.generate_content",
            "package": "vertexai.generative_models",
            "object": "GenerativeModel",
            "method": "generate_content",
        }

        wrapper_fn = _wrap(tracer, None, to_wrap)
        mock_wrapped = MagicMock(return_value=None)
        mock_instance = MagicMock()
        mock_instance._model_name = "publishers/google/models/gemini-pro"

        wrapper_fn(mock_wrapped, mock_instance, ("Hello",), {})

        spans = span_exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].status.status_code != StatusCode.ERROR


class TestAsyncErrorHandling:
    """Test error handling in the asynchronous _awrap function."""

    @pytest.mark.asyncio
    async def test_records_exception_on_api_error(self, tracer_provider, span_exporter):
        """When the async wrapped function raises, the span should record the exception."""
        tracer = tracer_provider.get_tracer("test")
        to_wrap = {
            "span_name": "vertexai.generate_content_async",
            "package": "vertexai.generative_models",
            "object": "GenerativeModel",
            "method": "generate_content_async",
        }

        wrapper_fn = _awrap(tracer, None, to_wrap)
        mock_wrapped = AsyncMock(side_effect=Exception("HTTP 429 Too Many Requests"))
        mock_instance = MagicMock()
        mock_instance._model_name = "publishers/google/models/gemini-pro"

        with pytest.raises(Exception, match="HTTP 429 Too Many Requests"):
            await wrapper_fn(mock_wrapped, mock_instance, ("Hello",), {})

        spans = span_exporter.get_finished_spans()
        assert len(spans) == 1

        span = spans[0]
        assert span.status.status_code == StatusCode.ERROR
        assert "HTTP 429 Too Many Requests" in span.status.description

        exception_events = [e for e in span.events if e.name == "exception"]
        assert len(exception_events) == 1

    @pytest.mark.asyncio
    async def test_async_exception_is_reraised(self, tracer_provider, span_exporter):
        """The original exception type and message must be preserved in async calls."""
        tracer = tracer_provider.get_tracer("test")
        to_wrap = {
            "span_name": "vertexai.generate_content_async",
            "package": "vertexai.generative_models",
            "object": "GenerativeModel",
            "method": "generate_content_async",
        }

        wrapper_fn = _awrap(tracer, None, to_wrap)
        mock_wrapped = AsyncMock(side_effect=RuntimeError("Connection refused"))
        mock_instance = MagicMock()
        mock_instance._model_name = "publishers/google/models/gemini-pro"

        with pytest.raises(RuntimeError, match="Connection refused"):
            await wrapper_fn(mock_wrapped, mock_instance, ("Hello",), {})
