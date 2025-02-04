import json
from unittest.mock import Mock, patch

import pytest
from opentelemetry.instrumentation.haystack.utils import process_response
from opentelemetry.semconv_ai import SpanAttributes

# Mocking the should_send_prompts function to control its behavior during tests


@pytest.fixture
def mock_should_send_prompts():
    with patch('opentelemetry.instrumentation.haystack.utils.should_send_prompts') as mock:
        yield mock

# Describe block for process_response tests


@pytest.mark.describe("process_response function")
class TestProcessResponse:

    @pytest.mark.happy_path
    def test_process_response_happy_path(self, mock_should_send_prompts):
        """
        Test that process_response sets the correct attribute on the span
        when should_send_prompts returns True.
        """
        mock_should_send_prompts.return_value = True
        span = Mock()
        response = {"key": "value"}

        process_response(span, response)

        span.set_attribute.assert_called_once_with(
            SpanAttributes.TRACELOOP_ENTITY_OUTPUT,
            json.dumps(response)
        )

    @pytest.mark.happy_path
    def test_process_response_no_attribute_set_when_prompts_disabled(self, mock_should_send_prompts):
        """
        Test that process_response does not set any attribute on the span
        when should_send_prompts returns False.
        """
        mock_should_send_prompts.return_value = False
        span = Mock()
        response = {"key": "value"}

        process_response(span, response)

        span.set_attribute.assert_not_called()

    @pytest.mark.edge_case
    def test_process_response_with_empty_response(self, mock_should_send_prompts):
        """
        Test that process_response handles an empty response correctly.
        """
        mock_should_send_prompts.return_value = True
        span = Mock()
        response = {}

        process_response(span, response)

        span.set_attribute.assert_called_once_with(
            SpanAttributes.TRACELOOP_ENTITY_OUTPUT,
            json.dumps(response)
        )

    @pytest.mark.edge_case
    def test_process_response_with_none_response(self, mock_should_send_prompts):
        """
        Test that process_response handles a None response gracefully.
        """
        mock_should_send_prompts.return_value = True
        span = Mock()
        response = None

        process_response(span, response)

        span.set_attribute.assert_called_once_with(
            SpanAttributes.TRACELOOP_ENTITY_OUTPUT,
            json.dumps(response)
        )

    @pytest.mark.edge_case
    def test_process_response_with_complex_response(self, mock_should_send_prompts):
        """
        Test that process_response can handle complex nested response objects.
        """
        mock_should_send_prompts.return_value = True
        span = Mock()
        response = {"key": {"nested_key": "nested_value"}}

        process_response(span, response)

        span.set_attribute.assert_called_once_with(
            SpanAttributes.TRACELOOP_ENTITY_OUTPUT,
            json.dumps(response)
        )


# Run the tests
if __name__ == "__main__":
    pytest.main()
