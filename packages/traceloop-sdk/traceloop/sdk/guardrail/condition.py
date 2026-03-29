"""
Deprecated: Use the module-level helpers in traceloop.sdk.guardrail.conditions instead.

This module is kept for backward compatibility. All methods emit DeprecationWarning
and delegate to the new helpers.
"""

import warnings
from typing import Any, Callable

from . import conditions


class Condition:
    """
    Deprecated: Use module-level helpers from traceloop.sdk.guardrail.conditions instead.

    Example migration:
        # Before
        from traceloop.sdk.guardrail import Condition
        condition = Condition.greater_than(0.8)

        # After
        from traceloop.sdk.guardrail.conditions import gt
        condition = gt(0.8)

        # Or use a plain lambda
        condition = lambda v: v > 0.8
    """

    @staticmethod
    def is_true() -> Callable[[Any], bool]:
        warnings.warn(
            "Condition.is_true() is deprecated. Use is_true() from "
            "traceloop.sdk.guardrail.conditions or pass a lambda instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return conditions.is_true()

    @staticmethod
    def is_false() -> Callable[[Any], bool]:
        warnings.warn(
            "Condition.is_false() is deprecated. Use is_false() from "
            "traceloop.sdk.guardrail.conditions or pass a lambda instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return conditions.is_false()

    @staticmethod
    def is_truthy() -> Callable[[Any], bool]:
        warnings.warn(
            "Condition.is_truthy() is deprecated. Use is_truthy() from "
            "traceloop.sdk.guardrail.conditions or pass a lambda instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return conditions.is_truthy()

    @staticmethod
    def is_falsy() -> Callable[[Any], bool]:
        warnings.warn(
            "Condition.is_falsy() is deprecated. Use is_falsy() from "
            "traceloop.sdk.guardrail.conditions or pass a lambda instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return conditions.is_falsy()

    @staticmethod
    def between(min_val: float, max_val: float) -> Callable[[Any], bool]:
        warnings.warn(
            "Condition.between() is deprecated. Use between() from "
            "traceloop.sdk.guardrail.conditions or pass a lambda instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return conditions.between(min_val, max_val)

    @staticmethod
    def equals(expected: Any) -> Callable[[Any], bool]:
        warnings.warn(
            "Condition.equals() is deprecated. Use eq() from "
            "traceloop.sdk.guardrail.conditions or pass a lambda instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return conditions.eq(expected)

    @staticmethod
    def greater_than(threshold: float) -> Callable[[Any], bool]:
        warnings.warn(
            "Condition.greater_than() is deprecated. Use gt() from "
            "traceloop.sdk.guardrail.conditions or pass a lambda instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return conditions.gt(threshold)

    @staticmethod
    def less_than(threshold: float) -> Callable[[Any], bool]:
        warnings.warn(
            "Condition.less_than() is deprecated. Use lt() from "
            "traceloop.sdk.guardrail.conditions or pass a lambda instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return conditions.lt(threshold)

    @staticmethod
    def greater_than_or_equal(threshold: float) -> Callable[[Any], bool]:
        warnings.warn(
            "Condition.greater_than_or_equal() is deprecated. Use gte() from "
            "traceloop.sdk.guardrail.conditions or pass a lambda instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return conditions.gte(threshold)

    @staticmethod
    def less_than_or_equal(threshold: float) -> Callable[[Any], bool]:
        warnings.warn(
            "Condition.less_than_or_equal() is deprecated. Use lte() from "
            "traceloop.sdk.guardrail.conditions or pass a lambda instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return conditions.lte(threshold)
