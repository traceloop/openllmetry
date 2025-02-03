import pytest
import asyncio
import unittest


# Import the run_async function from the specified path
from opentelemetry.instrumentation.anthropic.utils import run_async

@pytest.mark.describe("run_async function tests")
class TestRunAsync:

    @pytest.mark.happy_path
    def test_run_async_with_running_loop(self):
        """
        Test that run_async executes a coroutine when an event loop is already running.
        """
        async def sample_coroutine():
            return "success"

        # Mock asyncio.get_running_loop to simulate a running loop
        with unittest.mock.patch('asyncio.get_running_loop', return_value=asyncio.get_event_loop()):
            result = run_async(sample_coroutine())
            assert result is None  # Since the function doesn't return anything

    @pytest.mark.happy_path
    def test_run_async_without_running_loop(self):
        """
        Test that run_async executes a coroutine when no event loop is running.
        """
        async def sample_coroutine():
            return "success"

        # Mock asyncio.get_running_loop to raise RuntimeError, simulating no running loop
        with unittest.mock.patch('asyncio.get_running_loop', side_effect=RuntimeError):
            result = run_async(sample_coroutine())
            assert result is None  # Since the function doesn't return anything


    @pytest.mark.edge_case
    def test_run_async_with_exception_in_coroutine(self):
        """
        Test that run_async handles exceptions raised within the coroutine.
        """
        async def failing_coroutine():
            raise ValueError("Intentional error")

        # Mock asyncio.run to capture the exception
        with unittest.mock.patch('asyncio.run', side_effect=ValueError("Intentional error")):
            with pytest.raises(ValueError):
                run_async(failing_coroutine())
