"""
Guardrail module for the Traceloop SDK.

Provides a simple function-based guardrail system for running protected operations
with evaluation and failure handling.

Example:
    from traceloop.sdk import Traceloop
    from traceloop.sdk.guardrail import GuardedFunctionOutput, Condition, OnFailure
    from traceloop.sdk.evaluator import EvaluatorMadeByTraceloop

    # Initialize and get client
    client = Traceloop.init(api_key="...")

    async def generate_email() -> GuardedFunctionOutput[str, dict]:
        text = await llm.complete("Write a customer email...")
        return GuardedFunctionOutput(
            result=text,
            guard_input={"text": text},
        )

    result = await client.guardrails.run(
        func_to_guard=generate_email,
        guard=EvaluatorMadeByTraceloop.pii_detector().as_guard(
            condition=Condition.is_false("has_pii")
        ),
        on_failure=OnFailure.raise_exception("PII detected in response"),
    )
"""

from .guardrail import Guardrails
from .model import (
    GuardedFunctionOutput,
    GuardValidationError,
    GuardExecutionError,
    Guard,
    OnFailureHandler,
)
from .condition import Condition
from .on_failure import OnFailure

__all__ = [
    "Guardrails",
    "GuardedFunctionOutput",
    "GuardValidationError",
    "GuardExecutionError",
    "Guard",
    "OnFailureHandler",
    "Condition",
    "OnFailure",
]
