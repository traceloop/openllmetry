import pytest
from opentelemetry.instrumentation.groq.utils import error_metrics_attributes

# Describe block for all tests related to error_metrics_attributes
@pytest.mark.describe("Tests for error_metrics_attributes function")
class TestErrorMetricsAttributes:

    @pytest.mark.happy_path
    def test_error_metrics_attributes_with_standard_exception(self):
        """
        Test that error_metrics_attributes correctly extracts the class name
        of a standard exception and returns the expected dictionary.
        """
        exception = ValueError("An error occurred")
        result = error_metrics_attributes(exception)
        assert result == {
            "gen_ai.system": "groq",
            "error.type": "ValueError",
        }

    @pytest.mark.happy_path
    def test_error_metrics_attributes_with_custom_exception(self):
        """
        Test that error_metrics_attributes correctly extracts the class name
        of a custom exception and returns the expected dictionary.
        """
        class CustomException(Exception):
            pass

        exception = CustomException("A custom error occurred")
        result = error_metrics_attributes(exception)
        assert result == {
            "gen_ai.system": "groq",
            "error.type": "CustomException",
        }

    @pytest.mark.edge_case
    def test_error_metrics_attributes_with_no_message_exception(self):
        """
        Test that error_metrics_attributes handles an exception with no message
        and returns the expected dictionary.
        """
        exception = Exception()
        result = error_metrics_attributes(exception)
        assert result == {
            "gen_ai.system": "groq",
            "error.type": "Exception",
        }

    @pytest.mark.edge_case
    def test_error_metrics_attributes_with_non_standard_exception(self):
        """
        Test that error_metrics_attributes handles a non-standard exception
        (not derived from Exception) and returns the expected dictionary.
        """
        class NonStandardException:
            pass

        exception = NonStandardException()
        result = error_metrics_attributes(exception)
        assert result == {
            "gen_ai.system": "groq",
            "error.type": "NonStandardException",
        }

# Run the tests
if __name__ == "__main__":
    pytest.main()