import logging
from unittest.mock import Mock, patch

import pytest
from opentelemetry.instrumentation.groq.utils import dont_throw

# Configure logging to capture log messages for assertions
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Describe block for dont_throw tests


@pytest.mark.describe("dont_throw")
class TestDontThrow:

    @pytest.mark.happy_path
    def test_happy_path_function_execution(self):
        """
        Test that a function wrapped with dont_throw executes successfully without exceptions.
        """
        @dont_throw
        def sample_function(x, y):
            return x + y

        result = sample_function(2, 3)
        assert result == 5, "The function should return the sum of the inputs."

    @pytest.mark.happy_path
    def test_happy_path_no_exception_logging(self):
        """
        Test that no exception is logged when the wrapped function executes without errors.
        """
        @dont_throw
        def sample_function(x, y):
            return x + y

        with patch.object(logger, 'debug') as mock_debug:
            sample_function(2, 3)
            mock_debug.assert_not_called()

    @pytest.mark.edge_case
    def test_edge_case_custom_exception_logger(self):
        """
        Test that a custom exception logger is called when an exception occurs.
        """
        custom_logger = Mock()

        @dont_throw
        def sample_function(x, y):
            return x / y

        with patch('opentelemetry.instrumentation.groq.config.Config.exception_logger', custom_logger):
            sample_function(2, 0)
            custom_logger.assert_called_once()

    @pytest.mark.edge_case
    def test_edge_case_function_with_no_arguments(self):
        """
        Test that a function with no arguments wrapped with dont_throw executes correctly.
        """
        @dont_throw
        def sample_function():
            return "No args"

        result = sample_function()
        assert result == "No args", "The function should return the expected string."

    @pytest.mark.edge_case
    def test_edge_case_function_with_kwargs(self):
        """
        Test that a function with keyword arguments wrapped with dont_throw executes correctly.
        """
        @dont_throw
        def sample_function(x, y=10):
            return x + y

        result = sample_function(5, y=15)
        assert result == 20, "The function should correctly handle keyword arguments."


# Run the tests
if __name__ == "__main__":
    pytest.main()
