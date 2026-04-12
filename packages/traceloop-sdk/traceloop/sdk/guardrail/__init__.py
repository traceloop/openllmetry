"""
Guardrail module for the Traceloop SDK.

Provides a simple function-based guardrail system for running protected operations
with evaluation and failure handling.

Example:
    from traceloop.sdk.guardrail import Guardrails, pii_guard

    async def generate_email() -> str:
        return await llm.complete("Write a customer email...")

    guardrail = Guardrails(
        pii_guard(),
        on_failure="raise",
    )
    result = await guardrail.run(generate_email)

    # With arguments
    result = await guardrail.run(generate_response, user_prompt)

    # With custom input mapper
    result = await guardrail.run(
        generate_email,
        input_mapper=lambda text: [{"text": text}]
    )
"""

from .guardrail import Guardrails
from .model import (
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
)
from .condition import Condition
from .conditions import (
    gt,
    lt,
    gte,
    lte,
    between,
    eq,
    is_true,
    is_false,
    is_truthy,
    is_falsy,
)
from .on_failure import OnFailure, OnFailureInput, resolve_on_failure
from .guards import (
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
from .default_mapper import default_input_mapper

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
    # Condition helpers
    "gt",
    "lt",
    "gte",
    "lte",
    "between",
    "eq",
    "is_true",
    "is_false",
    "is_truthy",
    "is_falsy",
    "OnFailure",
    "OnFailureInput",
    "resolve_on_failure",
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
