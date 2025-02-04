import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from opentelemetry.instrumentation.anthropic.utils import dont_throw

# Mock Config to avoid side effects during testing


class MockConfig:
    exception_logger = None


@pytest.fixture(autouse=True)
def mock_config():
    with patch('opentelemetry.instrumentation.anthropic.utils.Config', new=MockConfig):
        yield


@pytest.mark.describe("Tests for async_wrapper")
class TestAsyncWrapper:

    @pytest.mark.happy_path
    @pytest.mark.asyncio
    async def test_async_wrapper_happy_path(self):
        """Test async_wrapper with a coroutine that succeeds."""
        async def successful_coroutine():
            return "success"

        wrapped_func = dont_throw(successful_coroutine)
        result = await wrapped_func()
        assert result == "success"

    @pytest.mark.happy_path
    @pytest.mark.asyncio
    async def test_async_wrapper_with_args(self):
        """Test async_wrapper with a coroutine that takes arguments."""
        async def coroutine_with_args(x, y):
            return x + y

        wrapped_func = dont_throw(coroutine_with_args)
        result = await wrapped_func(2, 3)
        assert result == 5

    @pytest.mark.edge_case
    @pytest.mark.asyncio
    async def test_async_wrapper_exception_handling(self, caplog):
        """Test async_wrapper with a coroutine that raises an exception."""
        async def failing_coroutine():
            raise ValueError("Test exception")

        wrapped_func = dont_throw(failing_coroutine)

        with caplog.at_level(logging.DEBUG):
            result = await wrapped_func()
            assert result is None
            assert "OpenLLMetry failed to trace in failing_coroutine" in caplog.text

    @pytest.mark.edge_case
    @pytest.mark.asyncio
    async def test_async_wrapper_no_exception_logger(self, caplog):
        """Test async_wrapper with a coroutine that raises an exception without an exception logger."""
        async def failing_coroutine():
            raise ValueError("Test exception")

        MockConfig.exception_logger = None

        wrapped_func = dont_throw(failing_coroutine)

        with caplog.at_level(logging.DEBUG):
            result = await wrapped_func()
            assert result is None
            assert "OpenLLMetry failed to trace in failing_coroutine" in caplog.text
            assert MockConfig.exception_logger is None

# Note: The `@pytest.mark.asyncio` decorator is used to run async tests with pytest.
