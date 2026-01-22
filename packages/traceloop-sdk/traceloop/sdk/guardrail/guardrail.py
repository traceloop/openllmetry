"""
Guardrails class for running guarded operations through the Traceloop client.
"""
import asyncio
import time
from typing import Callable, Awaitable, cast

import httpx
from opentelemetry.trace.status import Status, StatusCode

from traceloop.sdk.tracing import get_tracer
from traceloop.sdk.evaluator.evaluator import Evaluator

from .model import (
    GuardedOutput,
    Guard,
    OnFailureHandler,
    GuardedFunctionResult,
    GuardInput,
    FailureResult,
    GuardExecutionError,
)


class Guardrails:
    """
    Guardrails class for running guarded operations.

    Access via the Traceloop client:
        client = Traceloop.init(api_key="...")
        result = await client.guardrails.run(
            func_to_guard=my_agent,
            guard=lambda z: z["score"] > 0.8,
            on_failure=OnFailure.raise_exception("Quality check failed"),
        )
    """

    _evaluator: Evaluator
    _async_http: httpx.AsyncClient

    def __init__(self, async_http_client: httpx.AsyncClient):
        self._async_http = async_http_client
        self._evaluator = Evaluator(async_http_client)

    async def run(
        self,
        func_to_guard: Callable[
            [], Awaitable[GuardedOutput[GuardedFunctionResult, GuardInput]]
        ],
        guard: Guard,
        on_failure: OnFailureHandler,
    ) -> GuardedFunctionResult | FailureResult:
        """
        Execute a function with guardrail protection.

        Args:
            func_to_guard: Async function that returns GuardedOutput[T, Z].
                           Executed immediately inside run().

            guard: Function that receives Z (guard_input) and returns bool.
                   True = pass, False = fail.
                   Can be a lambda, custom function, or EvaluatorDetails.as_guard().

            on_failure: Called when guard returns False.
                        Receives the full GuardedFunctionOutput[T, Z].
                        Can raise, return a fallback value, log, or perform custom actions.
                        If it returns a value, that value is returned instead of output.result.

        Returns:
            T | F: The result from GuardedFunctionOutput.result, or the on_failure return value

        Raises:
            GuardValidationError: If guard returns False and on_failure raises
            GuardExecutionError: If the guard function throws an exception during execution

        Example:
            result = await client.guardrails.run(
                func_to_guard=generate_email,
                guard=EvaluatorMadeByTraceloop.pii_detector().as_guard(
                    condition=Condition.is_false("has_pii")
                ),
                on_failure=OnFailure.raise_exception("PII detected"),
            )
        """
        with get_tracer() as tracer:
            with tracer.start_as_current_span("guardrail.run") as span:
                start_time = time.perf_counter()

                print(f"NOMI - In the guard run function")

                # 1. Execute func_to_guard
                output: GuardedOutput[
                    GuardedFunctionResult, GuardInput
                ] = await func_to_guard()

                # 2. Run guard
                try:
                    guard_result = guard(output.guard_input)
                    if asyncio.iscoroutine(guard_result):
                        guard_result = await guard_result
                    guard_passed = bool(guard_result)
                    print(f"NOMI - Guard passed: {guard_passed}")
                except Exception as e:
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.record_exception(e)
                    raise GuardExecutionError(
                        message="Guard execution failed",
                        original_exception=e,
                        guard_input=output.guard_input,
                    ) from e

                duration_ms = (time.perf_counter() - start_time) * 1000

                span.set_attribute("guardrail.passed", guard_passed)
                span.set_attribute("guardrail.duration_ms", duration_ms)

                # 3. Handle failure
                if not guard_passed:
                    failure_result = on_failure(output)
                    if asyncio.iscoroutine(failure_result):
                        failure_result = await failure_result
                    return cast(FailureResult, failure_result)

                # 4. Guard passed, return result
                return output.result
