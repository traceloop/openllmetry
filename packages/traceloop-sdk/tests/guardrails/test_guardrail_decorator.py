"""
Unit tests for the guardrail decorator.

Tests the standalone @guardrail decorator from traceloop.sdk.decorators.
"""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from traceloop.sdk.decorators import guardrail
from traceloop.sdk.guardrail.on_failure import OnFailure


class TestGuardrailDecoratorBasic:
    """Tests for basic guardrail decorator functionality."""

    @pytest.mark.asyncio
    async def test_decorator_passes_through_result_when_guards_pass(self):
        """Decorator returns function result when all guards pass."""
        mock_guardrails_instance = MagicMock()
        mock_guardrails_instance.run = AsyncMock(return_value="guarded result")

        with patch("traceloop.sdk.guardrail.Guardrails", return_value=mock_guardrails_instance) as mock_guardrails_cls:

            @guardrail(lambda z: True, on_failure=OnFailure.raise_exception())
            async def my_function(prompt: str) -> str:
                return f"Response to: {prompt}"

            result = await my_function("Hello")

        assert result == "guarded result"
        mock_guardrails_cls.assert_called_once()
        mock_guardrails_instance.run.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_decorator_passes_function_arguments_correctly(self):
        """Decorator passes through function arguments to the wrapped function."""
        captured_args = []
        captured_kwargs = []

        mock_guardrails_instance = MagicMock()

        # Capture the func_to_guard and execute it
        async def run_side_effect(func_to_guard, *args, input_mapper=None, **kwargs):
            result = await func_to_guard(*args, **kwargs)
            return result

        mock_guardrails_instance.run = AsyncMock(side_effect=run_side_effect)

        with patch("traceloop.sdk.guardrail.Guardrails", return_value=mock_guardrails_instance):

            @guardrail(lambda z: True, on_failure=OnFailure.noop())
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
        mock_guardrails_instance = MagicMock()
        mock_guardrails_instance.run = AsyncMock(return_value="result")

        with patch("traceloop.sdk.guardrail.Guardrails", return_value=mock_guardrails_instance) as mock_guardrails_cls:

            @guardrail(lambda z: True, on_failure=OnFailure.noop())
            async def my_named_function() -> str:
                return "result"

            await my_named_function()

        # Check that constructor was called with the function name
        create_call = mock_guardrails_cls.call_args
        assert create_call.kwargs.get("name") == "my_named_function"

    @pytest.mark.asyncio
    async def test_decorator_uses_custom_name_when_provided(self):
        """Decorator uses custom name when provided."""
        mock_guardrails_instance = MagicMock()
        mock_guardrails_instance.run = AsyncMock(return_value="result")

        with patch("traceloop.sdk.guardrail.Guardrails", return_value=mock_guardrails_instance) as mock_guardrails_cls:

            @guardrail(
                lambda z: True,
                on_failure=OnFailure.noop(),
                name="custom-guardrail-name",
            )
            async def my_function() -> str:
                return "result"

            await my_function()

        create_call = mock_guardrails_cls.call_args
        assert create_call.kwargs.get("name") == "custom-guardrail-name"


class TestGuardrailDecoratorGuards:
    """Tests for guard configuration in guardrail decorator."""

    @pytest.mark.asyncio
    async def test_decorator_passes_guards_to_create(self):
        """Decorator passes guards to Guardrails constructor."""

        def guard1(z):
            return z["score"] > 0.5

        def guard2(z):
            return "bad" not in z["text"]

        mock_guardrails_instance = MagicMock()
        mock_guardrails_instance.run = AsyncMock(return_value="result")

        with patch("traceloop.sdk.guardrail.Guardrails", return_value=mock_guardrails_instance) as mock_guardrails_cls:

            @guardrail(guard1, guard2, on_failure=OnFailure.noop())
            async def my_function() -> str:
                return "result"

            await my_function()

        create_call = mock_guardrails_cls.call_args
        assert list(create_call.args) == [guard1, guard2]

    @pytest.mark.asyncio
    async def test_decorator_passes_input_mapper_to_run(self):
        """Decorator passes input_mapper to guardrails.run()."""

        def input_mapper(r):
            return [{"text": r}]

        mock_guardrails_instance = MagicMock()
        mock_guardrails_instance.run = AsyncMock(return_value="result")

        with patch("traceloop.sdk.guardrail.Guardrails", return_value=mock_guardrails_instance):

            @guardrail(
                lambda z: True,
                input_mapper=input_mapper,
                on_failure=OnFailure.noop(),
            )
            async def my_function() -> str:
                return "result"

            await my_function()

        run_call = mock_guardrails_instance.run.call_args
        assert run_call.kwargs.get("input_mapper") == input_mapper


class TestGuardrailDecoratorOnFailure:
    """Tests for on_failure behavior in guardrail decorator."""

    @pytest.mark.asyncio
    async def test_decorator_passes_on_failure_to_create(self):
        """Decorator passes on_failure handler to Guardrails constructor."""
        on_failure_handler = OnFailure.return_value("fallback")

        mock_guardrails_instance = MagicMock()
        mock_guardrails_instance.run = AsyncMock(return_value="result")

        with patch("traceloop.sdk.guardrail.Guardrails", return_value=mock_guardrails_instance) as mock_guardrails_cls:

            @guardrail(lambda z: True, on_failure=on_failure_handler)
            async def my_function() -> str:
                return "result"

            await my_function()

        create_call = mock_guardrails_cls.call_args
        assert create_call.kwargs.get("on_failure") == on_failure_handler

    @pytest.mark.asyncio
    async def test_decorator_uses_default_raise_exception_on_failure(self):
        """Decorator uses raise_exception as default on_failure."""
        mock_guardrails_instance = MagicMock()
        mock_guardrails_instance.run = AsyncMock(return_value="result")

        with patch("traceloop.sdk.guardrail.Guardrails", return_value=mock_guardrails_instance) as mock_guardrails_cls:

            @guardrail(lambda z: True)
            async def my_function() -> str:
                return "result"

            await my_function()

        create_call = mock_guardrails_cls.call_args
        on_failure = create_call.kwargs.get("on_failure")
        # Default should be a callable (OnFailure.raise_exception())
        assert callable(on_failure)

    @pytest.mark.asyncio
    async def test_decorator_converts_string_on_failure_to_return_value(self):
        """Decorator converts string on_failure to OnFailure.return_value()."""
        mock_guardrails_instance = MagicMock()
        mock_guardrails_instance.run = AsyncMock(return_value="result")

        with patch("traceloop.sdk.guardrail.Guardrails", return_value=mock_guardrails_instance) as mock_guardrails_cls:

            @guardrail(lambda z: True, on_failure="Blocked content")
            async def my_function() -> str:
                return "result"

            await my_function()

        create_call = mock_guardrails_cls.call_args
        on_failure = create_call.kwargs.get("on_failure")
        # Should be a callable that returns the string
        assert callable(on_failure)


class TestGuardrailDecoratorErrors:
    """Tests for error handling in guardrail decorator."""

    @pytest.mark.asyncio
    async def test_decorator_raises_when_guardrails_constructor_fails(self):
        """Decorator raises exception when Guardrails constructor fails."""
        with patch("traceloop.sdk.guardrail.Guardrails", side_effect=Exception(
            "Guardrails initialization failed"
        )):

            @guardrail(lambda z: True, on_failure=OnFailure.noop())
            async def my_function() -> str:
                return "result"

            with pytest.raises(Exception) as exc_info:
                await my_function()

            assert "Guardrails initialization failed" in str(exc_info.value)


class TestGuardrailDecoratorPreservesMetadata:
    """Tests for function metadata preservation."""

    def test_decorator_preserves_function_name(self):
        """Decorator preserves the original function name."""

        @guardrail(lambda z: True, on_failure=OnFailure.noop())
        async def original_function_name() -> str:
            return "result"

        # Check that function name is preserved
        assert original_function_name.__name__ == "original_function_name"

    def test_decorator_preserves_function_docstring(self):
        """Decorator preserves the original function docstring."""

        @guardrail(lambda z: True, on_failure=OnFailure.noop())
        async def documented_function() -> str:
            """This is the original docstring."""
            return "result"

        # Check that docstring is preserved
        assert documented_function.__doc__ == "This is the original docstring."


class TestGuardrailDecoratorSyncSupport:
    """Tests for sync function support in guardrail decorator."""

    def test_decorator_preserves_sync_function_name(self):
        """Decorator preserves the original sync function name."""

        @guardrail(lambda z: True, on_failure=OnFailure.noop())
        def sync_function_name() -> str:
            return "result"

        # Check that function name is preserved
        assert sync_function_name.__name__ == "sync_function_name"

    def test_decorator_preserves_sync_function_docstring(self):
        """Decorator preserves the original sync function docstring."""

        @guardrail(lambda z: True, on_failure=OnFailure.noop())
        def sync_documented_function() -> str:
            """This is a sync function docstring."""
            return "result"

        # Check that docstring is preserved
        assert sync_documented_function.__doc__ == "This is a sync function docstring."

    def test_sync_wrapper_runs_without_event_loop(self):
        """Sync wrapper works when no event loop is running."""
        mock_guardrails_instance = MagicMock()
        mock_guardrails_instance.run = AsyncMock(return_value="sync result")

        with patch("traceloop.sdk.guardrail.Guardrails", return_value=mock_guardrails_instance):

            @guardrail(lambda z: True, on_failure=OnFailure.noop())
            def my_sync_function(x: int) -> str:
                return f"value: {x}"

            result = my_sync_function(42)

        assert result == "sync result"
        mock_guardrails_instance.run.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_sync_wrapper_runs_inside_existing_event_loop(self):
        """Sync wrapper works when called from within a running event loop."""
        mock_guardrails_instance = MagicMock()
        mock_guardrails_instance.run = AsyncMock(return_value="sync result from loop")

        with patch("traceloop.sdk.guardrail.Guardrails", return_value=mock_guardrails_instance):

            @guardrail(lambda z: True, on_failure=OnFailure.noop())
            def my_sync_function(x: int) -> str:
                return f"value: {x}"

            # We're inside an async test, so there IS a running event loop.
            # This would crash with the old asyncio.run() approach.
            result = my_sync_function(42)

        assert result == "sync result from loop"
        mock_guardrails_instance.run.assert_awaited_once()
