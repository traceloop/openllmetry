import pytest
import logging
from unittest.mock import Mock, patch
from opentelemetry.instrumentation.pinecone.utils import dont_throw

# Create a mock logger to capture log outputs
class MockLogger:
    def __init__(self):
        self.messages = []

    def debug(self, msg, *args):
        self.messages.append(msg % args)

@pytest.mark.describe("dont_throw")
class TestDontThrow:
    
    @pytest.mark.happy_path
    def test_function_executes_without_exception(self):
        """Test that the wrapped function executes successfully without exceptions."""
        mock_func = Mock(return_value="success")
        wrapped_func = dont_throw(mock_func)
        
        result = wrapped_func()
        
        assert result == "success"
        mock_func.assert_called_once()

    @pytest.mark.happy_path
    def test_function_with_arguments(self):
        """Test that the wrapped function executes successfully with arguments."""
        mock_func = Mock(return_value="success")
        wrapped_func = dont_throw(mock_func)
        
        result = wrapped_func(1, 2, key="value")
        
        assert result == "success"
        mock_func.assert_called_once_with(1, 2, key="value")

    @pytest.mark.edge_case
    def test_function_with_no_return_value(self):
        """Test that the wrapped function handles functions with no return value."""
        mock_func = Mock(return_value=None)
        wrapped_func = dont_throw(mock_func)
        
        result = wrapped_func()
        
        assert result is None
        mock_func.assert_called_once()