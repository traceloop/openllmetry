import pytest
from opentelemetry.instrumentation.anthropic.utils import error_metrics_attributes

# Describe block for all tests related to error_metrics_attributes
@pytest.mark.describe("Tests for error_metrics_attributes function")
class TestErrorMetricsAttributes:

    @pytest.mark.happy_path
    def test_happy_path_standard_exception(self):
        """
        Test that error_metrics_attributes correctly extracts the error type
        from a standard exception.
        """
        exception = ValueError("An error occurred")
        result = error_metrics_attributes(exception)
        assert result == {
            "gen_ai.system": "anthropic",
            "error.type": "ValueError"
        }

    @pytest.mark.happy_path
    def test_happy_path_custom_exception(self):
        """
        Test that error_metrics_attributes correctly extracts the error type
        from a custom exception.
        """
        class CustomException(Exception):
            pass

        exception = CustomException("A custom error occurred")
        result = error_metrics_attributes(exception)
        assert result == {
            "gen_ai.system": "anthropic",
            "error.type": "CustomException"
        }

    @pytest.mark.edge_case
    def test_edge_case_empty_exception(self):
        """
        Test that error_metrics_attributes handles an exception with no message.
        """
        exception = Exception()
        result = error_metrics_attributes(exception)
        assert result == {
            "gen_ai.system": "anthropic",
            "error.type": "Exception"
        }

    @pytest.mark.edge_case
    def test_edge_case_non_standard_exception(self):
        """
        Test that error_metrics_attributes handles a non-standard exception object.
        """
        class NonStandardException:
            __class__ = type("NonStandardException", (), {})

        exception = NonStandardException()
        result = error_metrics_attributes(exception)
        assert result == {
            "gen_ai.system": "anthropic",
            "error.type": "NonStandardException"
        }

    @pytest.mark.edge_case
    def test_edge_case_subclass_exception(self):
        """
        Test that error_metrics_attributes correctly identifies a subclassed exception.
        """
        class BaseException(Exception):
            pass

        class SubclassException(BaseException):
            pass

        exception = SubclassException("Subclass error")
        result = error_metrics_attributes(exception)
        assert result == {
            "gen_ai.system": "anthropic",
            "error.type": "SubclassException"
        }

# To run these tests, you would typically use the command: pytest -v