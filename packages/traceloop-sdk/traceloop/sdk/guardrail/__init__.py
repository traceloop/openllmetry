"""
Guardrail module for the Traceloop SDK.

Provides a simple function-based guardrail system for running protected operations
with evaluation and failure handling.

Example:
    from traceloop.sdk import Traceloop
    from traceloop.sdk.guardrail import GuardedOutput, Guards, OnFailure

    # Initialize and get client
    client = Traceloop.init(api_key="...")

    async def generate_email() -> GuardedOutput[str, dict]:
        text = await llm.complete("Write a customer email...")
        return GuardedOutput(
            result=text,
            guard_inputs=[{"text": text}],
        )

    guardrail = client.guardrails.create(
        guards=[Guards.pii_detector()],
        on_failure=OnFailure.raise_exception("PII detected in response"),
    )
    result = await guardrail.run(generate_email)
"""

from .guardrail import Guardrails
from .model import (
    GuardedOutput,
    GuardValidationError,
    GuardExecutionError,
    GuardInputTypeError,
    Guard,
    OnFailureHandler,
)
from .condition import Condition
from .on_failure import OnFailure
from .guards import Guards, guard

__all__ = [
    "Guardrails",
    "GuardedOutput",
    "GuardValidationError",
    "GuardExecutionError",
    "GuardInputTypeError",
    "Guard",
    "OnFailureHandler",
    "Condition",
    "OnFailure",
    "Guards",
    "guard",
]
