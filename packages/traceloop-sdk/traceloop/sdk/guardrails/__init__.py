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
    default_input_mapper,
    # Guard functions
    custom_evaluator_guard,
    toxicity_guard,
    profanity_guard,
    sexism_guard,
    pii_guard,
    secrets_guard,
    prompt_injection_guard,
    json_validator_guard,
    sql_validator_guard,
    regex_validator_guard,
    instruction_adherence_guard,
    semantic_similarity_guard,
    prompt_perplexity_guard,
    uncertainty_guard,
    tone_detection_guard,
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
    "default_input_mapper",
    # Guard functions
    "custom_evaluator_guard",
    "toxicity_guard",
    "profanity_guard",
    "sexism_guard",
    "pii_guard",
    "secrets_guard",
    "prompt_injection_guard",
    "json_validator_guard",
    "sql_validator_guard",
    "regex_validator_guard",
    "instruction_adherence_guard",
    "semantic_similarity_guard",
    "prompt_perplexity_guard",
    "uncertainty_guard",
    "tone_detection_guard",
]
