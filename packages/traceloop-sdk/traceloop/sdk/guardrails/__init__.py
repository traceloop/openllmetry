"""
Deprecated module - use traceloop.sdk.guardrail instead.

This module provides backwards compatibility for the old import path.
"""
import warnings

warnings.warn(
    "traceloop.sdk.guardrails is deprecated. Use traceloop.sdk.guardrail instead.",
    DeprecationWarning,
    stacklevel=2
)

# Re-export everything from the new module for backwards compatibility
from traceloop.sdk.guardrail import (  # noqa: E402
    Guardrails,
    GuardedResult,
    GuardrailError,
    GuardValidationError,
    GuardExecutionError,
    GuardInputTypeError,
    Guard,
    OnFailureHandler,
    InputMapper,
    GuardInput,
    GuardedFunctionResult,
    Condition,
    OnFailure,
    Guards,
    guard,
    default_input_mapper,
)

__all__ = [
    "Guardrails",
    "GuardedResult",
    "GuardrailError",
    "GuardValidationError",
    "GuardExecutionError",
    "GuardInputTypeError",
    "Guard",
    "OnFailureHandler",
    "InputMapper",
    "GuardInput",
    "GuardedFunctionResult",
    "Condition",
    "OnFailure",
    "Guards",
    "guard",
    "default_input_mapper",
]
