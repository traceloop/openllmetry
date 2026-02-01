"""
Unit tests for the aguardrail decorator.

Tests the standalone @aguardrail decorator from traceloop.sdk.decorators.
"""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from traceloop.sdk.decorators import aguardrail
from traceloop.sdk.guardrail.on_failure import OnFailure


class TestAguardrailDecoratorBasic:
    """Tests for basic aguardrail decorator functionality."""

    @pytest.mark.asyncio
    async def test_decorator_passes_through_result_when_guards_pass(self):
        """Decorator returns function result when all guards pass."""
        mock_guardrails = MagicMock()
        mock_guardrails.create.return_value = mock_guardrails
        mock_guardrails.run = AsyncMock(return_value="guarded result")

        mock_client = MagicMock()
        mock_client.guardrails = mock_guardrails

        with patch("traceloop.sdk.Traceloop") as mock_traceloop:
            mock_traceloop.get.return_value = mock_client

            @aguardrail(
                guards=[lambda z: True],
                on_failure=OnFailure.raise_exception(),
            )
            async def my_function(prompt: str) -> str:
                return f"Response to: {prompt}"

            result = await my_function("Hello")

        assert result == "guarded result"
        mock_guardrails.create.assert_called_once()
        mock_guardrails.run.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_decorator_passes_function_arguments_correctly(self):
        """Decorator passes through function arguments to the wrapped function."""
        captured_args = []
        captured_kwargs = []

        mock_guardrails = MagicMock()
        mock_guardrails.create.return_value = mock_guardrails

        # Capture the func_to_guard and execute it
        async def run_side_effect(func_to_guard, input_mapper=None):
            result = await func_to_guard()
            return result

        mock_guardrails.run = AsyncMock(side_effect=run_side_effect)

        mock_client = MagicMock()
        mock_client.guardrails = mock_guardrails

        with patch("traceloop.sdk.Traceloop") as mock_traceloop:
            mock_traceloop.get.return_value = mock_client

            @aguardrail(
                guards=[lambda z: True],
                on_failure=OnFailure.noop(),
            )
            async def my_function(arg1: str, arg2: int, kwarg1: str = "default") -> str:
                captured_args.append((arg1, arg2))
                captured_kwargs.append(kwarg1)
                return f"{arg1}-{arg2}-{kwarg1}"

            result = await my_function("hello", 42, kwarg1="custom")

        assert captured_args == [("hello", 42)]
        assert captured_kwargs == ["custom"]
        assert result == "hello-42-custom"

    @pytest.mark.asyncio
    async def test_decorator_uses_default_function_name(self):
        """Decorator uses function name when name not provided."""
        mock_guardrails = MagicMock()
        mock_guardrails.create.return_value = mock_guardrails
        mock_guardrails.run = AsyncMock(return_value="result")

        mock_client = MagicMock()
        mock_client.guardrails = mock_guardrails

        with patch("traceloop.sdk.Traceloop") as mock_traceloop:
            mock_traceloop.get.return_value = mock_client

            @aguardrail(
                guards=[lambda z: True],
                on_failure=OnFailure.noop(),
            )
            async def my_named_function() -> str:
                return "result"

            await my_named_function()

        # Check that create was called with the function name
        create_call = mock_guardrails.create.call_args
        assert create_call.kwargs.get("name") == "my_named_function"

    @pytest.mark.asyncio
    async def test_decorator_uses_custom_name_when_provided(self):
        """Decorator uses custom name when provided."""
        mock_guardrails = MagicMock()
        mock_guardrails.create.return_value = mock_guardrails
        mock_guardrails.run = AsyncMock(return_value="result")

        mock_client = MagicMock()
        mock_client.guardrails = mock_guardrails

        with patch("traceloop.sdk.Traceloop") as mock_traceloop:
            mock_traceloop.get.return_value = mock_client

            @aguardrail(
                guards=[lambda z: True],
                on_failure=OnFailure.noop(),
                name="custom-guardrail-name",
            )
            async def my_function() -> str:
                return "result"

            await my_function()

        create_call = mock_guardrails.create.call_args
        assert create_call.kwargs.get("name") == "custom-guardrail-name"


class TestAguardrailDecoratorGuards:
    """Tests for guard configuration in aguardrail decorator."""

    @pytest.mark.asyncio
    async def test_decorator_passes_guards_to_create(self):
        """Decorator passes guards list to guardrails.create()."""
        guard1 = lambda z: z["score"] > 0.5
        guard2 = lambda z: "bad" not in z["text"]

        mock_guardrails = MagicMock()
        mock_guardrails.create.return_value = mock_guardrails
        mock_guardrails.run = AsyncMock(return_value="result")

        mock_client = MagicMock()
        mock_client.guardrails = mock_guardrails

        with patch("traceloop.sdk.Traceloop") as mock_traceloop:
            mock_traceloop.get.return_value = mock_client

            @aguardrail(
                guards=[guard1, guard2],
                on_failure=OnFailure.noop(),
            )
            async def my_function() -> str:
                return "result"

            await my_function()

        create_call = mock_guardrails.create.call_args
        assert create_call.kwargs.get("guards") == [guard1, guard2]

    @pytest.mark.asyncio
    async def test_decorator_passes_input_mapper_to_run(self):
        """Decorator passes input_mapper to guardrails.run()."""
        input_mapper = lambda r: [{"text": r}]

        mock_guardrails = MagicMock()
        mock_guardrails.create.return_value = mock_guardrails
        mock_guardrails.run = AsyncMock(return_value="result")

        mock_client = MagicMock()
        mock_client.guardrails = mock_guardrails

        with patch("traceloop.sdk.Traceloop") as mock_traceloop:
            mock_traceloop.get.return_value = mock_client

            @aguardrail(
                guards=[lambda z: True],
                input_mapper=input_mapper,
                on_failure=OnFailure.noop(),
            )
            async def my_function() -> str:
                return "result"

            await my_function()

        run_call = mock_guardrails.run.call_args
        assert run_call.kwargs.get("input_mapper") == input_mapper


class TestAguardrailDecoratorOnFailure:
    """Tests for on_failure behavior in aguardrail decorator."""

    @pytest.mark.asyncio
    async def test_decorator_passes_on_failure_to_create(self):
        """Decorator passes on_failure handler to guardrails.create()."""
        on_failure_handler = OnFailure.return_value("fallback")

        mock_guardrails = MagicMock()
        mock_guardrails.create.return_value = mock_guardrails
        mock_guardrails.run = AsyncMock(return_value="result")

        mock_client = MagicMock()
        mock_client.guardrails = mock_guardrails

        with patch("traceloop.sdk.Traceloop") as mock_traceloop:
            mock_traceloop.get.return_value = mock_client

            @aguardrail(
                guards=[lambda z: True],
                on_failure=on_failure_handler,
            )
            async def my_function() -> str:
                return "result"

            await my_function()

        create_call = mock_guardrails.create.call_args
        assert create_call.kwargs.get("on_failure") == on_failure_handler

    @pytest.mark.asyncio
    async def test_decorator_uses_default_raise_exception_on_failure(self):
        """Decorator uses raise_exception as default on_failure."""
        mock_guardrails = MagicMock()
        mock_guardrails.create.return_value = mock_guardrails
        mock_guardrails.run = AsyncMock(return_value="result")

        mock_client = MagicMock()
        mock_client.guardrails = mock_guardrails

        with patch("traceloop.sdk.Traceloop") as mock_traceloop:
            mock_traceloop.get.return_value = mock_client

            @aguardrail(guards=[lambda z: True])
            async def my_function() -> str:
                return "result"

            await my_function()

        create_call = mock_guardrails.create.call_args
        on_failure = create_call.kwargs.get("on_failure")
        # Default should be a callable (OnFailure.raise_exception())
        assert callable(on_failure)


class TestAguardrailDecoratorErrors:
    """Tests for error handling in aguardrail decorator."""

    @pytest.mark.asyncio
    async def test_decorator_raises_when_traceloop_not_initialized(self):
        """Decorator raises exception when Traceloop.get() fails."""
        with patch("traceloop.sdk.Traceloop") as mock_traceloop:
            mock_traceloop.get.side_effect = Exception(
                "Client not initialized, you should call Traceloop.init() first."
            )

            @aguardrail(
                guards=[lambda z: True],
                on_failure=OnFailure.noop(),
            )
            async def my_function() -> str:
                return "result"

            with pytest.raises(Exception) as exc_info:
                await my_function()

            assert "Client not initialized" in str(exc_info.value)


class TestAguardrailDecoratorPreservesMetadata:
    """Tests for function metadata preservation."""

    def test_decorator_preserves_function_name(self):
        """Decorator preserves the original function name."""

        @aguardrail(
            guards=[lambda z: True],
            on_failure=OnFailure.noop(),
        )
        async def original_function_name() -> str:
            return "result"

        # Check that function name is preserved
        assert original_function_name.__name__ == "original_function_name"

    def test_decorator_preserves_function_docstring(self):
        """Decorator preserves the original function docstring."""

        @aguardrail(
            guards=[lambda z: True],
            on_failure=OnFailure.noop(),
        )
        async def documented_function() -> str:
            """This is the original docstring."""
            return "result"

        # Check that docstring is preserved
        assert documented_function.__doc__ == "This is the original docstring."
