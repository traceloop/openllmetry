from unittest.mock import MagicMock

import pytest
from opentelemetry.instrumentation.anthropic.config import Config
from opentelemetry.instrumentation.anthropic.utils import \
    shared_metrics_attributes
from opentelemetry.semconv_ai import SpanAttributes

# Mock configuration for common metrics attributes


@pytest.fixture(autouse=True)
def mock_config():
    Config.get_common_metrics_attributes = MagicMock(return_value={"common_attr": "value"})


@pytest.mark.describe("shared_metrics_attributes")
class TestSharedMetricsAttributes:

    @pytest.mark.happy_path
    def test_shared_metrics_attributes_with_valid_response(self):
        """
        Test that shared_metrics_attributes returns the correct attributes
        when given a valid response dictionary.
        """
        response = {"model": "test-model"}
        expected_attributes = {
            "common_attr": "value",
            "gen_ai.system": "anthropic",
            SpanAttributes.LLM_RESPONSE_MODEL: "test-model",
        }
        assert shared_metrics_attributes(response) == expected_attributes

    @pytest.mark.happy_path
    def test_shared_metrics_attributes_with_empty_response(self):
        """
        Test that shared_metrics_attributes returns the correct attributes
        when given an empty response dictionary.
        """
        response = {}
        expected_attributes = {
            "common_attr": "value",
            "gen_ai.system": "anthropic",
            SpanAttributes.LLM_RESPONSE_MODEL: None,
        }
        assert shared_metrics_attributes(response) == expected_attributes

    @pytest.mark.edge_case
    def test_shared_metrics_attributes_with_non_dict_response(self):
        """
        Test that shared_metrics_attributes correctly handles a non-dict response
        by converting it to a dictionary using __dict__.
        """
        class ResponseObject:
            def __init__(self):
                self.model = "object-model"

        response = ResponseObject()
        expected_attributes = {
            "common_attr": "value",
            "gen_ai.system": "anthropic",
            SpanAttributes.LLM_RESPONSE_MODEL: "object-model",
        }
        assert shared_metrics_attributes(response) == expected_attributes

    @pytest.mark.edge_case
    def test_shared_metrics_attributes_with_none_response(self):
        """
        Test that shared_metrics_attributes handles a None response gracefully.
        """
        response = None
        expected_attributes = {
            "common_attr": "value",
            "gen_ai.system": "anthropic",
            SpanAttributes.LLM_RESPONSE_MODEL: None,
        }
        assert shared_metrics_attributes(response) == expected_attributes

    @pytest.mark.edge_case
    def test_shared_metrics_attributes_with_unexpected_attributes(self):
        """
        Test that shared_metrics_attributes ignores unexpected attributes in the response.
        """
        response = {"unexpected": "value", "model": "test-model"}
        expected_attributes = {
            "common_attr": "value",
            "gen_ai.system": "anthropic",
            SpanAttributes.LLM_RESPONSE_MODEL: "test-model",
        }
        assert shared_metrics_attributes(response) == expected_attributes
