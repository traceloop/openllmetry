"""
Built-in conditions for evaluating guard results.

Conditions are functions that take an evaluator result and return a boolean
indicating whether the guard should pass (True) or fail (False).
"""
from typing import Any, Callable


def _get_field(result: Any, field: str, default: Any = None) -> Any:
    """
    Get a field from result, supporting both dict and object attribute access.

    Args:
        result: The result object or dict
        field: The field name to access
        default: Default value if field not found

    Returns:
        The field value, or default if not found
    """
    # Try dict access first
    if isinstance(result, dict):
        return result.get(field, default)
    # Fall back to attribute access
    return getattr(result, field, default)


class Condition:
    """Built-in conditions for common evaluator result patterns."""

    @staticmethod
    def success() -> Callable[[Any], bool]:
        """
        Pass if result.success is True.

        Example:
            guard=EvaluatorMadeByTraceloop.pii_detector().as_guard(
                condition=Condition.success()
            )
        """
        return lambda result: _get_field(result, "success", False) is True

    @staticmethod
    def is_true(field: str) -> Callable[[Any], bool]:
        """
        Pass if specified field is True.

        Args:
            field: The attribute name to check

        Example:
            condition=Condition.is_true("matched")
        """
        return lambda result: _get_field(result, field, None) is True

    @staticmethod
    def is_false(field: str) -> Callable[[Any], bool]:
        """
        Pass if specified field is False.

        Args:
            field: The attribute name to check

        Example:
            condition=Condition.is_false("contains_pii")
        """
        return lambda result: _get_field(result, field, None) is False


    @staticmethod
    def between(
        min_val: float, max_val: float, field: str = "score"
    ) -> Callable[[Any], bool]:
        """
        Pass if min_val <= field <= max_val.

        Args:
            min_val: Minimum acceptable value (inclusive)
            max_val: Maximum acceptable value (inclusive)
            field: The attribute name to check (default: "score")

        Example:
            condition=Condition.between(50, 200, field="count")
        """

        def check(result: Any) -> bool:
            value = _get_field(result, field, None)
            if value is None:
                return False
            return bool(min_val <= value <= max_val)

        return check

    @staticmethod
    def equals(value: Any, field: str) -> Callable[[Any], bool]:
        """
        Pass if field == value.

        Args:
            value: The expected value
            field: The attribute name to check

        Example:
            condition=Condition.equals("approved", field="status")
        """
        return lambda result: _get_field(result, field, None) == value

    @staticmethod
    def greater_than(value: float, field: str = "score") -> Callable[[Any], bool]:
        """
        Pass if field > value.

        Args:
            value: The threshold (exclusive)
            field: The attribute name to check (default: "score")

        Example:
            condition=Condition.greater_than(10, field="count")
        """
        return lambda result: _get_field(result, field, 0) > value

    @staticmethod
    def less_than(value: float, field: str = "score") -> Callable[[Any], bool]:
        """
        Pass if field < value.

        Args:
            value: The threshold (exclusive)
            field: The attribute name to check (default: "score")

        Example:
            condition=Condition.less_than(1000, field="latency_ms")
        """
        return lambda result: _get_field(result, field, float("inf")) < value

    @staticmethod
    def greater_than_or_equal(
        value: float, field: str = "score"
    ) -> Callable[[Any], bool]:
        """
        Pass if field >= value.

        Args:
            value: The threshold (inclusive)
            field: The attribute name to check (default: "score")

        Example:
            condition=Condition.greater_than_or_equal(0.8, field="confidence")
        """
        return lambda result: _get_field(result, field, 0) >= value

    @staticmethod
    def less_than_or_equal(value: float, field: str = "score") -> Callable[[Any], bool]:
        """
        Pass if field <= value.

        Args:
            value: The threshold (inclusive)
            field: The attribute name to check (default: "score")

        Example:
            condition=Condition.less_than_or_equal(0.5, field="toxicity")
        """
        return lambda result: _get_field(result, field, float("inf")) <= value
