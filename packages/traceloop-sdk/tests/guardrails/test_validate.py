"""
Unit tests for guardrail validate method.

Tests the validate method which runs guards directly on inputs.
"""
import pytest
from unittest.mock import MagicMock, AsyncMock

from traceloop.sdk.guardrail.guardrail import Guardrails
from traceloop.sdk.guardrail.model import GuardExecutionError
from traceloop.sdk.guardrail.on_failure import OnFailure


def create_guardrails_with_guards(guards: list, on_failure=None) -> Guardrails:
    """Helper to create a Guardrails instance with specified guards."""
    mock_client = MagicMock()
    guardrails = Guardrails(mock_client)
    guardrails._guards = guards
    guardrails._on_failure = on_failure or OnFailure.noop()
    return guardrails


class TestValidateReturnsTrue:
    """Tests for validate returning True when guards pass."""

    @pytest.mark.asyncio
    async def test_single_guard_passes(self):
        """Validate returns True when single guard passes."""
        guardrails = create_guardrails_with_guards(
            guards=[lambda z: z["score"] > 0.5]
        )
        result = await guardrails.validate([{"score": 0.8}])
        assert result is True

    @pytest.mark.asyncio
    async def test_multiple_guards_all_pass(self):
        """Validate returns True when all guards pass."""
        guardrails = create_guardrails_with_guards(
            guards=[
                lambda z: z["score"] > 0.5,
                lambda z: z["valid"] is True,
            ]
        )
        result = await guardrails.validate([
            {"score": 0.8},
            {"valid": True},
        ])
        assert result is True

    @pytest.mark.asyncio
    async def test_async_guard_passes(self):
        """Validate returns True when async guard passes."""
        async def async_guard(data: dict) -> bool:
            return data.get("score", 0) > 0.5

        guardrails = create_guardrails_with_guards(guards=[async_guard])
        result = await guardrails.validate([{"score": 0.8}])
        assert result is True


class TestValidateReturnsFalse:
    """Tests for validate returning False when guards fail."""

    @pytest.mark.asyncio
    async def test_single_guard_fails(self):
        """Validate returns False when single guard fails."""
        guardrails = create_guardrails_with_guards(
            guards=[lambda z: z["score"] > 0.5]
        )
        result = await guardrails.validate([{"score": 0.3}])
        assert result is False

    @pytest.mark.asyncio
    async def test_one_of_multiple_guards_fails(self):
        """Validate returns False when any guard fails."""
        guardrails = create_guardrails_with_guards(
            guards=[
                lambda z: z["score"] > 0.5,
                lambda z: z["valid"] is True,
            ]
        )
        result = await guardrails.validate([
            {"score": 0.8},  # passes
            {"valid": False},  # fails
        ])
        assert result is False

    @pytest.mark.asyncio
    async def test_async_guard_fails(self):
        """Validate returns False when async guard fails."""
        async def async_guard(data: dict) -> bool:
            return data.get("score", 0) > 0.5

        guardrails = create_guardrails_with_guards(guards=[async_guard])
        result = await guardrails.validate([{"score": 0.3}])
        assert result is False


class TestValidateOnFailureHandler:
    """Tests for on_failure handler behavior."""

    @pytest.mark.asyncio
    async def test_calls_class_on_failure_when_guard_fails(self):
        """Validate calls the class-configured on_failure handler."""
        on_failure_mock = MagicMock(return_value=None)
        guardrails = create_guardrails_with_guards(
            guards=[lambda z: z["score"] > 0.5],
            on_failure=on_failure_mock,
        )

        result = await guardrails.validate([{"score": 0.3}])

        assert result is False
        on_failure_mock.assert_called_once()

    @pytest.mark.asyncio
    async def test_does_not_call_on_failure_when_guards_pass(self):
        """Validate does not call on_failure when all guards pass."""
        on_failure_mock = MagicMock(return_value=None)
        guardrails = create_guardrails_with_guards(
            guards=[lambda z: z["score"] > 0.5],
            on_failure=on_failure_mock,
        )

        result = await guardrails.validate([{"score": 0.8}])

        assert result is True
        on_failure_mock.assert_not_called()

    @pytest.mark.asyncio
    async def test_override_on_failure_parameter(self):
        """Validate uses overridden on_failure when provided."""
        class_on_failure = MagicMock(return_value=None)
        override_on_failure = MagicMock(return_value=None)

        guardrails = create_guardrails_with_guards(
            guards=[lambda z: z["score"] > 0.5],
            on_failure=class_on_failure,
        )

        result = await guardrails.validate(
            [{"score": 0.3}],
            on_failure=override_on_failure,
        )

        assert result is False
        class_on_failure.assert_not_called()
        override_on_failure.assert_called_once()

    @pytest.mark.asyncio
    async def test_async_on_failure_handler(self):
        """Validate handles async on_failure handlers."""
        async_on_failure = AsyncMock(return_value=None)
        guardrails = create_guardrails_with_guards(
            guards=[lambda z: z["score"] > 0.5],
            on_failure=async_on_failure,
        )

        result = await guardrails.validate([{"score": 0.3}])

        assert result is False
        async_on_failure.assert_awaited_once()


class TestValidateErrors:
    """Tests for error handling in validate."""

    @pytest.mark.asyncio
    async def test_raises_value_error_without_create(self):
        """Validate raises ValueError if create() was not called."""
        mock_client = MagicMock()
        guardrails = Guardrails(mock_client)
        # _guards is empty since create() was not called

        with pytest.raises(ValueError) as exc_info:
            await guardrails.validate([{"score": 0.8}])

        assert "Must call create() before validate()" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_raises_value_error_on_input_length_mismatch(self):
        """Validate raises ValueError when input count doesn't match guards."""
        guardrails = create_guardrails_with_guards(
            guards=[lambda z: True, lambda z: True]
        )

        with pytest.raises(ValueError) as exc_info:
            await guardrails.validate([{"score": 0.8}])  # 1 input for 2 guards

        assert "Number of guard_inputs (1)" in str(exc_info.value)
        assert "must match number of guards (2)" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_raises_guard_execution_error_when_guard_throws(self):
        """Validate raises GuardExecutionError when guard raises exception."""
        def failing_guard(data):
            raise RuntimeError("Guard crashed")

        guardrails = create_guardrails_with_guards(guards=[failing_guard])

        with pytest.raises(GuardExecutionError) as exc_info:
            await guardrails.validate([{"score": 0.8}])

        assert "Guard execution failed" in str(exc_info.value)
        assert exc_info.value.guard_index == 0
        assert isinstance(exc_info.value.original_exception, RuntimeError)


class TestValidateParallelSequential:
    """Tests for parallel vs sequential guard execution."""

    @pytest.mark.asyncio
    async def test_parallel_execution(self):
        """Validate runs guards in parallel by default."""
        call_order = []

        async def guard1(data):
            call_order.append(1)
            return True

        async def guard2(data):
            call_order.append(2)
            return True

        guardrails = create_guardrails_with_guards(guards=[guard1, guard2])
        guardrails._parallel = True

        result = await guardrails.validate([{"a": 1}, {"b": 2}])

        assert result is True
        assert len(call_order) == 2

    @pytest.mark.asyncio
    async def test_sequential_execution(self):
        """Validate runs guards sequentially when parallel=False."""
        guardrails = create_guardrails_with_guards(
            guards=[
                lambda z: z["pass"],
                lambda z: z["pass"],
            ]
        )
        guardrails._parallel = False

        result = await guardrails.validate([
            {"pass": True},
            {"pass": True},
        ])

        assert result is True

    @pytest.mark.asyncio
    async def test_sequential_stops_at_first_failure_when_run_all_false(self):
        """Sequential execution stops at first failure when run_all=False."""
        call_count = [0]

        def counting_guard(data):
            call_count[0] += 1
            return data.get("pass", False)

        guardrails = create_guardrails_with_guards(
            guards=[counting_guard, counting_guard, counting_guard]
        )
        guardrails._parallel = False
        guardrails._run_all = False

        result = await guardrails.validate([
            {"pass": False},  # First guard fails
            {"pass": True},
            {"pass": True},
        ])

        assert result is False
        assert call_count[0] == 1  # Only first guard was called
