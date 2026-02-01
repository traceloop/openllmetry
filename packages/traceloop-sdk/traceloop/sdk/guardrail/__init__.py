"""
Guardrail module for the Traceloop SDK.

Provides a simple function-based guardrail system for running protected operations
with evaluation and failure handling.

Example:
    from traceloop.sdk import Traceloop
    from traceloop.sdk.guardrail import Guards, OnFailure

    # Initialize and get client
    client = Traceloop.init(api_key="...")

    async def generate_email() -> str:
        return await llm.complete("Write a customer email...")

    guardrail = client.guardrails.create(
        guards=[Guards.pii_detector()],
        on_failure=OnFailure.raise_exception("PII detected in response"),
    )
    result = await guardrail.run(generate_email)

    # With custom input mapper
    result = await guardrail.run(
        generate_email,
        input_mapper=lambda text: [{"text": text}]
    )
"""

from .guardrail import Guardrails
from .model import (
    GuardedResult,
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
from .on_failure import OnFailure
from .guards import Guards, guard
from .default_mapper import default_input_mapper

__all__ = [
    "Guardrails",
    "GuardedResult",
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
