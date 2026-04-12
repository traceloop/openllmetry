"""
Module-level condition helpers for evaluating guard results.

These are short, Pythonic alternatives to the Condition class.
Each function returns a Callable[[Any], bool] suitable for use as a guard condition.

Example:
    from traceloop.sdk.guardrail.conditions import gt, is_true

    toxicity_guard(condition=is_true())
    answer_correctness_guard(condition=gt(0.9))

    # Or use plain lambdas:
    toxicity_guard(condition=lambda v: v is True)
    answer_correctness_guard(condition=lambda v: v > 0.9)
"""

from typing import Any, Callable


def is_true() -> Callable[[Any], bool]:
    """Pass if value is exactly True (strict bool check)."""

    def check(value: Any) -> bool:
        return isinstance(value, bool) and value is True

    return check


def is_false() -> Callable[[Any], bool]:
    """Pass if value is exactly False (strict bool check)."""

    def check(value: Any) -> bool:
        return isinstance(value, bool) and value is False

    return check


def is_truthy() -> Callable[[Any], bool]:
    """Pass if bool(value) is True."""

    def check(value: Any) -> bool:
        return bool(value)

    return check


def is_falsy() -> Callable[[Any], bool]:
    """Pass if bool(value) is False."""

    def check(value: Any) -> bool:
        return not bool(value)

    return check


def gt(threshold: float) -> Callable[[Any], bool]:
    """Pass if value > threshold. Returns False for None."""

    def check(value: Any) -> bool:
        if value is None:
            return False
        return bool(value > threshold)

    return check


def lt(threshold: float) -> Callable[[Any], bool]:
    """Pass if value < threshold. Returns False for None."""

    def check(value: Any) -> bool:
        if value is None:
            return False
        return bool(value < threshold)

    return check


def gte(threshold: float) -> Callable[[Any], bool]:
    """Pass if value >= threshold. Returns False for None."""

    def check(value: Any) -> bool:
        if value is None:
            return False
        return bool(value >= threshold)

    return check


def lte(threshold: float) -> Callable[[Any], bool]:
    """Pass if value <= threshold. Returns False for None."""

    def check(value: Any) -> bool:
        if value is None:
            return False
        return bool(value <= threshold)

    return check


def between(min_val: float, max_val: float) -> Callable[[Any], bool]:
    """Pass if min_val <= value <= max_val (inclusive). Returns False for None."""

    def check(value: Any) -> bool:
        if value is None:
            return False
        return bool(min_val <= value <= max_val)

    return check


def eq(expected: Any) -> Callable[[Any], bool]:
    """Pass if value == expected."""

    def check(value: Any) -> bool:
        return bool(value == expected)

    return check
