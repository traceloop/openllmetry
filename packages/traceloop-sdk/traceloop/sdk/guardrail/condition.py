"""
Built-in conditions for evaluating guard results.

Conditions are functions that take an evaluator result value and return a boolean
indicating whether the guard should pass (True) or fail (False).

The condition_field is extracted by the guard function before passing to these
conditions, so conditions receive the field value directly.
"""
from typing import Any, Callable


class Condition:
    """Built-in conditions for common evaluator result patterns."""

    @staticmethod
    def is_true() -> Callable[[Any], bool]:
        """
        Pass if value is exactly True (must be a bool).

        Rejects non-boolean types like 1 or "yes".
        For truthy checking, use is_truthy() instead.

        Example:
            toxicity_guard(condition=Condition.is_true())
        """

        def check(value: bool) -> bool:
            return isinstance(value, bool) and value is True

        return check

    @staticmethod
    def is_false() -> Callable[[Any], bool]:
        """
        Pass if value is exactly False (must be a bool).

        Rejects non-boolean types like 0 or "".
        For falsy checking, use is_falsy() instead.

        Example:
            pii_guard(condition=Condition.is_false())
        """

        def check(value: bool) -> bool:
            return isinstance(value, bool) and value is False

        return check

    @staticmethod
    def is_truthy() -> Callable[[Any], bool]:
        """
        Pass if value is truthy (bool(value) is True).

        Unlike is_true(), this accepts truthy values like 1, "yes",
        non-empty lists.

        Example:
            guard(condition=Condition.is_truthy())
        """

        def check(value: Any) -> bool:
            return bool(value)

        return check

    @staticmethod
    def is_falsy() -> Callable[[Any], bool]:
        """
        Pass if value is falsy (bool(value) is False).

        Unlike is_false(), this accepts falsy values like 0, "",
        empty lists.

        Example:
            guard(condition=Condition.is_falsy())
        """

        def check(value: Any) -> bool:
            return not bool(value)

        return check

    @staticmethod
    def between(min_val: float, max_val: float) -> Callable[[Any], bool]:
        """
        Pass if min_val <= value <= max_val.

        Args:
            min_val: Minimum acceptable value (inclusive)
            max_val: Maximum acceptable value (inclusive)

        Example:
            condition=Condition.between(50, 200)
        """

        def check(value: Any) -> bool:
            if value is None:
                return False
            return bool(min_val <= value <= max_val)

        return check

    @staticmethod
    def equals(expected: Any) -> Callable[[Any], bool]:
        """
        Pass if value == expected.

        Args:
            expected: The expected value

        Example:
            condition=Condition.equals("approved")
        """

        def check(value: Any) -> bool:
            return bool(value == expected)

        return check

    @staticmethod
    def greater_than(threshold: float) -> Callable[[Any], bool]:
        """
        Pass if value > threshold.

        Args:
            threshold: The threshold (exclusive)

        Example:
            condition=Condition.greater_than(0.8)
        """

        def check(value: Any) -> bool:
            if value is None:
                return False
            return bool(value > threshold)

        return check

    @staticmethod
    def less_than(threshold: float) -> Callable[[Any], bool]:
        """
        Pass if value < threshold.

        Args:
            threshold: The threshold (exclusive)

        Example:
            condition=Condition.less_than(0.5)
        """

        def check(value: Any) -> bool:
            if value is None:
                return False
            return bool(value < threshold)

        return check

    @staticmethod
    def greater_than_or_equal(threshold: float) -> Callable[[Any], bool]:
        """
        Pass if value >= threshold.

        Args:
            threshold: The threshold (inclusive)

        Example:
            condition=Condition.greater_than_or_equal(0.8)
        """

        def check(value: Any) -> bool:
            if value is None:
                return False
            return bool(value >= threshold)

        return check

    @staticmethod
    def less_than_or_equal(threshold: float) -> Callable[[Any], bool]:
        """
        Pass if value <= threshold.

        Args:
            threshold: The threshold (inclusive)

        Example:
            condition=Condition.less_than_or_equal(0.5)
        """

        def check(value: Any) -> bool:
            if value is None:
                return False
            return bool(value <= threshold)

        return check
