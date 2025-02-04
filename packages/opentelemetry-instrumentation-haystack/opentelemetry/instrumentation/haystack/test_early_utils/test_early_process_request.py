import json
from unittest.mock import Mock, patch

import pytest
from opentelemetry.instrumentation.haystack.utils import process_request
from opentelemetry.semconv_ai import SpanAttributes

# Mocking the context API and Config for testing


@pytest.fixture(autouse=True)
def mock_context_api(monkeypatch):
    mock_context = Mock()
    monkeypatch.setattr("opentelemetry.instrumentation.haystack.utils.context_api", mock_context)
    return mock_context


@pytest.fixture(autouse=True)
def mock_config(monkeypatch):
    mock_config = Mock()
    monkeypatch.setattr("opentelemetry.instrumentation.haystack.utils.Config", mock_config)
    return mock_config


@pytest.fixture
def mock_span():
    return Mock()


@pytest.mark.describe("process_request function")
class TestProcessRequest:

    @pytest.mark.happy_path
    def test_process_request_with_empty_args_and_kwargs(self, mock_span):
        """
        Test that process_request handles empty args and kwargs gracefully.
        """
        args = ()
        kwargs = {}

        with patch("opentelemetry.instrumentation.haystack.utils.should_send_prompts", return_value=True):
            process_request(mock_span, args, kwargs)

        expected_input_entity = {
            "args": [],
            "kwargs": {}
        }
        mock_span.set_attribute.assert_called_once_with(
            SpanAttributes.TRACELOOP_ENTITY_INPUT,
            json.dumps(expected_input_entity)
        )

    @pytest.mark.edge_case
    def test_process_request_with_non_dict_args(self, mock_span):
        """
        Test that process_request correctly handles non-dict args.
        """
        args = ("arg1", 123, 45.6)
        kwargs = {"kwarg1": "value1"}

        with patch("opentelemetry.instrumentation.haystack.utils.should_send_prompts", return_value=True):
            process_request(mock_span, args, kwargs)

        expected_input_entity = {
            "args": ["arg1", 123, 45.6],
            "kwargs": {"kwarg1": "value1"}
        }
        mock_span.set_attribute.assert_called_once_with(
            SpanAttributes.TRACELOOP_ENTITY_INPUT,
            json.dumps(expected_input_entity)
        )

    @pytest.mark.edge_case
    def test_process_request_with_should_send_prompts_false(self, mock_span):
        """
        Test that process_request does not set attributes when should_send_prompts is False.
        """
        args = ({"key1": "value1"},)
        kwargs = {"kwarg1": "value2"}

        with patch("opentelemetry.instrumentation.haystack.utils.should_send_prompts", return_value=False):
            process_request(mock_span, args, kwargs)

        mock_span.set_attribute.assert_not_called()

    @pytest.mark.edge_case
    def test_process_request_with_exception_handling(self, mock_span, mock_config):
        """
        Test that process_request handles exceptions and logs them without throwing.
        """
        args = ({"key1": "value1"},)
        kwargs = {"kwarg1": "value2"}

        # Simulate an exception in set_attribute
        mock_span.set_attribute.side_effect = Exception("Test exception")

        with patch("opentelemetry.instrumentation.haystack.utils.should_send_prompts", return_value=True):
            process_request(mock_span, args, kwargs)

        # Ensure exception logger is called
        mock_config.exception_logger.assert_called_once()
