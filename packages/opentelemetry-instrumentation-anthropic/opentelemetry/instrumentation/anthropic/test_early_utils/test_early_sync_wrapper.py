import pytest
import logging
from unittest.mock import Mock, patch
from opentelemetry.instrumentation.anthropic.utils import dont_throw

# Mock Config to avoid side effects during testing
class MockConfig:
    exception_logger = Mock()

# Sample function to be wrapped
def sample_function(x, y):
    return x + y

# Sample function to raise an exception
def exception_function(x, y):
    raise ValueError("An error occurred")

@pytest.mark.describe("sync_wrapper")
class TestSyncWrapper:
    
    @pytest.mark.happy_path
    def test_sync_wrapper_happy_path(self):
        """
        Test that sync_wrapper correctly returns the result of a function without exceptions.
        """
        wrapped_function = dont_throw(sample_function)
        result = wrapped_function(2, 3)
        assert result == 5, "Expected the wrapped function to return the sum of 2 and 3"

    @pytest.mark.edge_case
    def test_sync_wrapper_with_no_arguments(self):
        """
        Test that sync_wrapper works with functions that take no arguments.
        """
        def no_arg_function():
            return "no args"

        wrapped_function = dont_throw(no_arg_function)
        result = wrapped_function()
        assert result == "no args", "Expected the wrapped function to return 'no args'"

    @pytest.mark.edge_case
    def test_sync_wrapper_with_none_return(self):
        """
        Test that sync_wrapper correctly handles functions that return None.
        """
        def none_return_function():
            return None

        wrapped_function = dont_throw(none_return_function)
        result = wrapped_function()
        assert result is None, "Expected the wrapped function to return None"