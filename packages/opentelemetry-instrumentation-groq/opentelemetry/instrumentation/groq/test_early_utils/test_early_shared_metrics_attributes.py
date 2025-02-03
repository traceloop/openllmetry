import pytest
from unittest.mock import MagicMock, patch
from opentelemetry.instrumentation.groq.utils import shared_metrics_attributes
from opentelemetry.instrumentation.groq.config import Config
from opentelemetry.semconv_ai import SpanAttributes

# Mocking the Config class to control the behavior of get_common_metrics_attributes
@pytest.fixture
def mock_config():
    with patch('opentelemetry.instrumentation.groq.config.Config.get_common_metrics_attributes') as mock:
        yield mock

# Mocking the model_as_dict function
@pytest.fixture
def mock_model_as_dict():
    with patch('opentelemetry.instrumentation.groq.utils.model_as_dict') as mock:
        yield mock

@pytest.mark.describe("shared_metrics_attributes")
class TestSharedMetricsAttributes:

    @pytest.mark.happy_path
    def test_shared_metrics_attributes_with_valid_response(self, mock_config, mock_model_as_dict):
        """
        Test that shared_metrics_attributes returns the correct attributes
        when given a valid response object.
        """
        # Arrange
        mock_config.return_value = {"common_attr": "value"}
        mock_model_as_dict.return_value = {"model": "test_model"}
        response = MagicMock()

        # Act
        result = shared_metrics_attributes(response)

        # Assert
        assert result == {
            "common_attr": "value",
            "gen_ai.system": "groq",
            SpanAttributes.LLM_RESPONSE_MODEL: "test_model"
        }

    @pytest.mark.happy_path
    def test_shared_metrics_attributes_with_empty_common_attributes(self, mock_config, mock_model_as_dict):
        """
        Test that shared_metrics_attributes handles empty common attributes correctly.
        """
        # Arrange
        mock_config.return_value = {}
        mock_model_as_dict.return_value = {"model": "test_model"}
        response = MagicMock()

        # Act
        result = shared_metrics_attributes(response)

        # Assert
        assert result == {
            "gen_ai.system": "groq",
            SpanAttributes.LLM_RESPONSE_MODEL: "test_model"
        }

    @pytest.mark.edge_case
    def test_shared_metrics_attributes_with_no_model_in_response(self, mock_config, mock_model_as_dict):
        """
        Test that shared_metrics_attributes handles a response with no model attribute.
        """
        # Arrange
        mock_config.return_value = {"common_attr": "value"}
        mock_model_as_dict.return_value = {}
        response = MagicMock()

        # Act
        result = shared_metrics_attributes(response)

        # Assert
        assert result == {
            "common_attr": "value",
            "gen_ai.system": "groq",
            SpanAttributes.LLM_RESPONSE_MODEL: None
        }

    @pytest.mark.edge_case
    def test_shared_metrics_attributes_with_exception_in_model_as_dict(self, mock_config, mock_model_as_dict):
        """
        Test that shared_metrics_attributes handles exceptions in model_as_dict gracefully.
        """
        # Arrange
        mock_config.return_value = {"common_attr": "value"}
        mock_model_as_dict.side_effect = Exception("Test Exception")
        response = MagicMock()

        # Act
        result = shared_metrics_attributes(response)

        # Assert
        assert result is None  # Since the function is decorated with @dont_throw, it should return None on exception