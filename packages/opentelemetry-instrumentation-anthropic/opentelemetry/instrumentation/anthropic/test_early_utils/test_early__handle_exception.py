import pytest
import logging
from unittest.mock import Mock, patch
from opentelemetry.instrumentation.anthropic.utils import dont_throw

# Mock Config to control the behavior of exception_logger
class MockConfig:
    exception_logger = None

# Patch the Config used in the module with our MockConfig
@pytest.fixture(autouse=True)
def patch_config():
    with patch('opentelemetry.instrumentation.anthropic.utils.Config', MockConfig):
        yield

# Describe block for _handle_exception related tests
@pytest.mark.describe("_handle_exception")
class TestHandleException:

    @pytest.mark.happy_path
    def test_sync_function_no_exception(self):
        """Test that a synchronous function runs without exceptions."""
        @dont_throw
        def no_exception_func():
            return "success"

        assert no_exception_func() == "success"

    @pytest.mark.happy_path
    @pytest.mark.asyncio
    async def test_async_function_no_exception(self):
        """Test that an asynchronous function runs without exceptions."""
        @dont_throw
        async def no_exception_func():
            return "success"

        assert await no_exception_func() == "success"

    @pytest.mark.edge_case
    def test_sync_function_with_exception(self, caplog):
        """Test that a synchronous function logs an exception without raising it."""
        @dont_throw
        def exception_func():
            raise ValueError("Test exception")

        with caplog.at_level(logging.DEBUG):
            exception_func()
            assert "OpenLLMetry failed to trace in exception_func, error:" in caplog.text

    @pytest.mark.edge_case
    @pytest.mark.asyncio
    async def test_async_function_with_exception(self, caplog):
        """Test that an asynchronous function logs an exception without raising it."""
        @dont_throw
        async def exception_func():
            raise ValueError("Test exception")

        with caplog.at_level(logging.DEBUG):
            await exception_func()
            assert "OpenLLMetry failed to trace in exception_func, error:" in caplog.text

    @pytest.mark.edge_case
    def test_no_exception_logger(self):
        """Test that no error occurs if exception_logger is None."""
        MockConfig.exception_logger = None

        @dont_throw
        def exception_func():
            raise ValueError("Test exception")

        exception_func()  # Should not raise any error